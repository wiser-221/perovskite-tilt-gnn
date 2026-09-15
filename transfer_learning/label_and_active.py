"""CHGNet single-point labels, budget-matched active learning and acceptance.

One SQLite file stores labels, queries, predictions, metrics and GPU logs.
Run: python transfer_learning/label_and_active.py all | label | learn | report
"""
from __future__ import annotations
import argparse, hashlib, json, math, sqlite3, subprocess, sys, threading, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'transfer_learning'))
import train_formation as common
DB=common.DB

def connect():
    db=sqlite3.connect(DB,timeout=60)
    db.execute('CREATE TABLE IF NOT EXISTS labels(id TEXT PRIMARY KEY, energy REAL, max_force REAL, model TEXT, geometry_hash TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS rounds(run TEXT, budget INTEGER, selected TEXT, metrics TEXT, PRIMARY KEY(run,budget))')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(run TEXT,budget INTEGER,id TEXT,split TEXT,values_json TEXT,PRIMARY KEY(run,budget,id))')
    return db

def geometry_hash(item):
    return hashlib.sha256(json.dumps(item['structure'],sort_keys=True).encode()).hexdigest()

def label():
    from chgnet.model import CHGNet
    from pymatgen.core import Structure
    cfg=common.config();common.setup();data=common.load_data()
    model=CHGNet.load(model_name=cfg['chgnet_model'],use_device='cuda')
    with connect() as db:
        existing={r[0]:(r[1],r[2]) for r in db.execute('SELECT id,model,geometry_hash FROM labels')}
    for i,item in enumerate(data['candidates']):
        cid=item['candidate_id'];digest=geometry_hash(item)
        if cid in existing:
            if existing[cid]!=(cfg['chgnet_model'],digest):raise ValueError('stale label '+cid)
            continue
        t=time.perf_counter()
        # predict_structure returns intensive energy (eV/atom); no optimizer.
        result=model.predict_structure(Structure.from_dict(item['structure']),task='ef')
        energy=float(result['e']);force=float(np.linalg.norm(result['f'],axis=1).max())
        if not math.isfinite(energy) or not math.isfinite(force):raise ValueError('nonfinite CHGNet label')
        with connect() as db:db.execute('INSERT INTO labels VALUES(?,?,?,?,?)',(cid,energy,force,cfg['chgnet_model'],digest))
        if i%20==0:common.event('label_progress',done=i+1,total=len(data['candidates']),seconds=time.perf_counter()-t)
    common.event('label_complete',count=len(data['candidates']),source='CHGNet_single_point_NOT_DFT')

def proxy_labels(data):
    with connect() as db:raw={r[0]:r[1] for r in db.execute('SELECT id,energy FROM labels')}
    if set(raw)!={c['candidate_id'] for c in data['candidates']}:raise ValueError('labels incomplete or stale')
    parents={c['material_id']:c for c in data['candidates'] if c['pattern']=='parent'}
    labels={c['candidate_id']:c['parent_formation_ev_atom']+raw[c['candidate_id']]-raw[parents[c['material_id']]['candidate_id']]
            for c in data['candidates']}
    for p in parents.values():
        assert abs(labels[p['candidate_id']]-p['parent_formation_ev_atom'])<1e-12
    return labels

def predict_bundle(bundle,candidates,labels,device):
    matrix=[];cfg=common.config()
    # Prediction/acquisition does not even read unqueried reference values.
    dl,_=common.loader(common.ProxyDataset(candidates,{c['candidate_id']:0. for c in candidates}),cfg['batch_size'])
    for seed in cfg['seeds']:
        saved=bundle[seed];model=common.make_model(saved,device)
        pred,_=common.predict(model,dl,device,saved['target_mean'],saved['target_std'])
        matrix.append(pred);del model
    torch.cuda.empty_cache()
    return np.array(matrix)

def summarize(candidates,labels,matrix):
    truth=np.array([labels[c['candidate_id']] for c in candidates]);mean=matrix.mean(axis=0)
    uncertainty=matrix.std(axis=0,ddof=1);groups=defaultdict(list)
    for i,c in enumerate(candidates):groups[c['composition']].append(i)
    pair_errors=[];regrets=[];pair_u=[];pair_abs=[];tops=[]
    for raw_ix in groups.values():
        # Angle accuracy is measured only inside the fixed-cell generated scan.
        # The dataset parent has a different cell/order and is a formation-energy bridge,
        # not a clean one-variable angle comparison.
        ix=[i for i in raw_ix if candidates[i].get('pattern')!='parent']
        if len(ix)<2:continue
        ix=np.array(ix);a,b=np.triu_indices(len(ix),1);left,right=ix[a],ix[b]
        error=abs((mean[left]-mean[right])-(truth[left]-truth[right]))
        pair_errors.append(float(error.mean()));pair_abs.extend(error.tolist())
        pair_u.extend((matrix[:,left]-matrix[:,right]).std(axis=0,ddof=1).tolist())
        regret=truth[ix[np.argmin(mean[ix])]]-truth[ix].min()
        regrets.append(float(regret));tops.append(float(regret<1e-6))
    order=np.argsort(pair_u);n=len(order)
    # Half-coverage calibration fixed as an operational requirement in config.
    # Each entry here is a pair; acceptance MAE remains composition-balanced.
    take=order[:max(1,math.ceil(n*common.config()['uncertainty_min_coverage']))]
    corr=spearmanr(pair_u,pair_abs).statistic if np.std(pair_u)>0 and np.std(pair_abs)>0 else float('nan')
    return {'proxy_formation_mae':float(abs(mean-truth).mean()),'pair_mae':float(np.mean(pair_errors)),
            'regret':float(np.mean(regrets)),'top1':float(np.mean(tops)),
            'pair_error_p90':float(np.quantile(pair_abs,.9)),
            'pair_uncertainty_spearman':float(corr) if math.isfinite(corr) else None,
            'uncertainty_threshold':float(np.asarray(pair_u)[take].max()),
            'accepted_pair_mae':float(np.asarray(pair_abs)[take].mean()),'coverage':len(take)/n,
            'mean_std':float(uncertainty.mean())}

def formation_mae(bundle, dataset, device):
    dl,_=common.loader(dataset,common.config()['batch_size']);pred=[]
    for saved in bundle.values():
        model=common.make_model(saved,device)
        p,y=common.predict(model,dl,device,saved['target_mean'],saved['target_std']);pred.append(p);del model
    return float(np.mean(abs(np.mean(pred,axis=0)-y)))

def acquisition(candidates,selected,matrix,budget,strategy,seed):
    selected=set(selected);remaining=[i for i,c in enumerate(candidates) if c['candidate_id'] not in selected]
    if not remaining:return sorted(selected)
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    for i,c in enumerate(candidates):groups[c['composition']].append(i)
    uncertainty=matrix.std(axis=0,ddof=1)
    for _ in range(min(budget-len(selected),len(remaining))):
        if strategy=='random':chosen=int(rng.choice(remaining))
        else:
            # Rank uncertainty within each composition to reduce composition-offset dominance.
            def score(i):
                item=candidates[i];ix=groups[item['composition']]
                known=[j for j in ix if candidates[j]['candidate_id'] in selected]
                rank=sum(uncertainty[j]<=uncertainty[i] for j in ix)/len(ix)
                same=[j for j in known if candidates[j]['pattern']==item['pattern']]
                diversity=1. if not same else min(abs(candidates[j]['amplitude_deg']-item['amplitude_deg']) for j in same)/14.
                return rank+diversity+1/(1+len(known))
            chosen=max(remaining,key=lambda i:(score(i),candidates[i]['candidate_id']))
        selected.add(candidates[chosen]['candidate_id']);remaining.remove(chosen)
    return sorted(selected)

def accepted(metrics):
    cfg=common.config()
    return (metrics['pair_mae']<=cfg['pair_mae_limit'] and metrics['regret']<=cfg['regret_limit']
            and metrics['retention_ratio']<=cfg['retention_ratio_limit']
            and metrics['accepted_pair_mae']<=cfg['pair_mae_limit']
            and metrics['pair_uncertainty_spearman'] is not None and metrics['pair_uncertainty_spearman']>0)

def learn():
    cfg=common.config();device=common.setup();data=common.load_data();labels=proxy_labels(data)
    baseline=torch.load(ROOT/'transfer_learning/base_models.pt',map_location='cpu',weights_only=False)
    if set(baseline)!=set(cfg['seeds']):raise ValueError('need all five formation models')
    train=[c for c in data['candidates'] if c['split']=='train']
    val=[c for c in data['candidates'] if c['split']=='validation']
    valset=common.base_dataset([r for r in data['rows'] if r['formation_split']=='validation'])
    replay_rows=[r for r in data['rows'] if r['formation_split']=='train']
    replay_rows=sorted(replay_rows,key=lambda r:hashlib.sha256(('replay'+r['material_id']).encode()).hexdigest())[:cfg['replay_samples']]
    replay=common.base_dataset(replay_rows)
    base_mae=formation_mae(baseline,valset,device)
    baseline_matrix=predict_bundle(baseline,train,labels,device)
    best_score=float('inf');best_metrics=None;best_run=None
    final_path=ROOT/'transfer_learning/final_models.pt';work_path=ROOT/'transfer_learning/working.pt'
    # Only validation used for selecting the final strategy/round; all test labels remain hidden.
    for sampling_seed in cfg['sampling_seeds']:
        initial=[]
        for composition in sorted({c['composition'] for c in train}):
            group=[c for c in train if c['composition']==composition]
            group.sort(key=lambda c:(c['pattern']!='parent',hashlib.sha256((str(sampling_seed)+c['candidate_id']).encode()).hexdigest()))
            initial.extend(c['candidate_id'] for c in group[:4])
        for strategy in ('active','random'):
            run=f'anchored_v7_{strategy}_{sampling_seed}';selected=initial;bundle=baseline;matrix=baseline_matrix
            for requested in cfg['budgets']:
                budget=min(requested,len(train))
                with connect() as db:record=db.execute('SELECT selected,metrics FROM rounds WHERE run=? AND budget=?',(run,budget)).fetchone()
                # Completed run rounds need their state to continue. The working checkpoint
                # persists the latest round of the current run; finished older runs are skipped below.
                if record:
                    m=json.loads(record[1])
                    m['accepted']=accepted(m)
                    m['selection_score']=float(not m['accepted'])+m['pair_mae']+max(
                        0,m['retention_ratio']-cfg['retention_ratio_limit'])
                    with connect() as db:
                        db.execute('UPDATE rounds SET metrics=? WHERE run=? AND budget=?',
                                   (json.dumps(m),run,budget))
                    if m['selection_score']<best_score:best_score=m['selection_score'];best_metrics=m;best_run=run
                    selected=json.loads(record[0])
                    if work_path.exists():
                        working=torch.load(work_path,map_location='cpu',weights_only=False)
                        if working['run']==run and working['budget']==budget:bundle=working['models'];matrix=predict_bundle(bundle,train,labels,device)
                    continue
                if budget>len(selected):selected=acquisition(train,selected,matrix,budget,strategy,sampling_seed+budget)
                # Every round starts from baseline for comparable budgets and resumable controls.
                adapted={}
                if work_path.exists():
                    working=torch.load(work_path,map_location='cpu',weights_only=False)
                    if working['run']==run and working['budget']==budget:adapted=working['models'];selected=working['selected']
                samples=[c for c in train if c['candidate_id'] in set(selected)]
                proxy=common.ProxyDataset(samples,labels)
                # Persist query choices BEFORE the first seed, including when it is interrupted.
                common.save_atomic({'run':run,'budget':budget,'selected':selected,'models':adapted},work_path)
                for seed in cfg['seeds']:
                    if seed in adapted:continue
                    adapted[seed]=common.fit(baseline[seed],replay,valset,device,f'{run}_{budget}_{seed}',
                                             cfg['adapt_epochs'],cfg['adapt_patience'],cfg['adapt_lr'],proxy,val,labels)
                    common.save_atomic({'run':run,'budget':budget,'selected':selected,'models':adapted},work_path)
                bundle=adapted;matrix=predict_bundle(bundle,train,labels,device)
                vm=predict_bundle(bundle,val,labels,device);m=summarize(val,labels,vm)
                m['formation_val_mae']=formation_mae(bundle,valset,device);m['retention_ratio']=m['formation_val_mae']/base_mae
                m['accepted']=accepted(m);m['labels']=len(selected)
                m['selection_score']=float(not m['accepted'])+m['pair_mae']+max(0,m['retention_ratio']-cfg['retention_ratio_limit'])
                with connect() as db:
                    db.execute('INSERT INTO rounds VALUES(?,?,?,?)',(run,budget,json.dumps(selected),json.dumps(m)))
                    for j,c in enumerate(val):db.execute('INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?)',
                        (run,budget,c['candidate_id'],'validation',json.dumps(vm[:,j].tolist())))
                if m['selection_score']<best_score:
                    best_score=m['selection_score'];best_metrics=m;best_run=run;common.save_atomic(bundle,final_path)
                common.event('round_complete',run=run,budget=budget,**m);report()
                if budget==len(train):break
    # Final test is evaluated once after all validation-based choices are frozen.
    with connect() as db:done=db.execute("SELECT 1 FROM events WHERE kind='final_test'").fetchone()
    if best_metrics and accepted(best_metrics) and not done:
        bundle=torch.load(final_path,map_location='cpu',weights_only=False)
        test=[c for c in data['candidates'] if c['split']=='test']
        tm=predict_bundle(bundle,test,labels,device);m=summarize(test,labels,tm)
        testset=common.base_dataset([r for r in data['rows'] if r['formation_split']=='test'])
        m['formation_test_mae']=formation_mae(bundle,testset,device)
        m['retention_ratio']=m['formation_test_mae']/formation_mae(baseline,testset,device)
        # Use frozen validation uncertainty threshold (not test-derived quantiles).
        truth=np.array([labels[c['candidate_id']] for c in test]);groups=defaultdict(list)
        for i,c in enumerate(test):groups[c['composition']].append(i)
        errors=[];us=[]
        for raw_ix in groups.values():
            ix=[i for i in raw_ix if test[i]['pattern']!='parent']
            if len(ix)<2:continue
            a,b=np.triu_indices(len(ix),1);left=np.array(ix)[a];right=np.array(ix)[b]
            errors.extend(abs((tm.mean(0)[left]-tm.mean(0)[right])-(truth[left]-truth[right])))
            us.extend((tm[:,left]-tm[:,right]).std(0,ddof=1))
        mask=np.array(us)<=best_metrics['uncertainty_threshold']
        m['coverage']=float(mask.mean());m['accepted_pair_mae']=float(np.array(errors)[mask].mean()) if mask.any() else 1e9
        m['uncertainty_threshold']=best_metrics['uncertainty_threshold']
        m['accepted']=accepted(m) and m['coverage']>=cfg['uncertainty_min_coverage']
        common.event('final_test',run=best_run,**m)
    else:
        common.event('budget_complete',validation_accepted=bool(best_metrics and accepted(best_metrics)),
                     action='inspect_report_before_expanding_budget',test_opened=bool(done))
    report()

def report():
    with connect() as db:
        n=db.execute('SELECT COUNT(*) FROM labels').fetchone()[0]
        rows=list(db.execute('SELECT run,budget,metrics FROM rounds ORDER BY run,budget'))
    lines=['# Step 4—6 运行与结果报告','','目标：给定角度畸变结构的形成能。新增标签为CHGNet代理值，不是DFT真值。',
           '',f'已完成单点标签：{n}。已完成主动学习预算实验：{len(rows)}。','',
           '|策略|标签数|成对MAE eV/atom|regret|原形成能退化比|验证达标|',
           '|---|---:|---:|---:|---:|---|']
    for run,budget,raw in rows:
        m=json.loads(raw);lines.append(f"|{run}|{budget}|{m['pair_mae']:.6f}|{m['regret']:.6f}|{m['retention_ratio']:.4f}|{m['accepted']}|")
    lines+=['','三项能量门槛：成对MAE≤0.01、regret≤0.01 eV/atom，原形成能MAE退化比≤1.05。',
            '不确定性要求：验证集低分歧50%结构对误差≤0.01，且分歧与误差正相关；最终测试使用冻结阈值并检查覆盖率。',
            '运行细节、每轮选样、标签与GPU记录统一保存在results.sqlite；完成预算不代表达到目标。',
            '角度生成器采用共享氧位置平均的原生Glazer构造，不是PySPuDS；会伴随键长变化。',
            '每条生成扫描固定体积与rocksalt排序；与数据集母体的差还可包含晶胞形状和排序差，不能全归因于角度。']
    path=ROOT/'transfer_learning/README.md'
    existing=path.read_text(encoding='utf-8') if path.exists() else ''
    marker='\n<!-- generated-results -->\n'
    guide=existing.split(marker)[0]
    path.write_text(guide+marker+'\n'.join(lines)+'\n',encoding='utf-8')

def monitor(stop):
    while not stop.is_set():
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                   '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            if result.returncode==0:
                with sqlite3.connect(DB,timeout=60) as db:
                    db.execute('CREATE TABLE IF NOT EXISTS gpu(time REAL, values_csv TEXT)')
                    db.execute('INSERT INTO gpu VALUES(?,?)',(time.time(),result.stdout.strip()))
        except Exception:pass
        stop.wait(10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['all','label','learn','report']);args=parser.parse_args()
    stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,),daemon=True);thread.start()
    try:
        if args.action=='all':common.train();label();learn()
        elif args.action=='label':label()
        elif args.action=='learn':learn()
        else:report()
    finally:stop.set();thread.join(timeout=6)
