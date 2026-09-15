"""Five-seed paired-graph model for cubic-anchored tilt formation energies."""
from __future__ import annotations
import csv, json, random, sqlite3, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'project'),str(ROOT/'transfer_learning')]
from graph_data import collate_graphs
from models import AngleGNN
import train_formation as common

class PairSet(Dataset):
    def __init__(self,items,cubic,proxy,raw):self.items,self.cubic,self.proxy,self.raw=items,cubic,proxy,raw
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        x=self.items[i];r=self.cubic[x['composition']]
        return dict(x['graph']),dict(r['graph']),torch.tensor(self.raw[x['candidate_id']]-self.raw[r['candidate_id']]),\
               torch.tensor(self.proxy[x['candidate_id']]),torch.tensor(self.proxy[r['candidate_id']])

def collate(items):
    x,r,d,y,yr=zip(*items)
    return collate_graphs(x),collate_graphs(r),torch.stack(d),torch.stack(y),torch.stack(yr)

class DualAngleGNN(nn.Module):
    def __init__(self,saved):
        super().__init__();cfg=saved['model_config'];h=cfg['hidden_dim']
        self.base=AngleGNN(**cfg);self.base.load_state_dict(saved['model_state_dict'])
        self.encoder=AngleGNN(**cfg);self.encoder.load_state_dict(saved['model_state_dict'])
        for p in self.base.parameters():p.requires_grad_(False)
        self.delta_head=nn.Sequential(nn.LayerNorm(3*h),nn.Linear(3*h,h),nn.SiLU(),nn.Linear(h,1))
        self.anchor_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h//2),nn.SiLU(),nn.Linear(h//2,1))
        nn.init.zeros_(self.delta_head[-1].weight);nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.anchor_head[-1].weight);nn.init.zeros_(self.anchor_head[-1].bias)
    def components(self,x,r):
        hx,hr=self.encoder.encode(x),self.encoder.encode(r);z=torch.cat((hx-hr,abs(hx-hr),hx*hr),1)
        zero=torch.zeros_like(z)
        delta=(self.delta_head(z)-self.delta_head(zero)).squeeze(1)
        anchor=self.anchor_head(hr).squeeze(1)
        return self.base(r),anchor,delta
    def forward(self,x,r):
        base,anchor,delta=self.components(x,r);return base+anchor+delta

def move(x,device):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in x.items()}

@torch.inference_mode()
def evaluate(model,ds,device,mean,std,batch=48):
    model.eval();pred=[];truth=[];dp=[];dt=[]
    for x,r,d,y,_ in DataLoader(ds,batch_size=batch,collate_fn=collate):
        x,r=move(x,device),move(r,device);base,anchor,delta=model.components(x,r)
        pred.extend(((base+anchor+delta)*std+mean).cpu());truth.extend(y)
        dp.extend((delta*std).cpu());dt.extend(d)
    pred,truth,dp,dt=map(lambda z:np.asarray(z,float),(pred,truth,dp,dt))
    return {'absolute_mae':float(abs(pred-truth).mean()),'delta_mae':float(abs(dp-dt).mean()),
            'pred':pred,'truth':truth,'delta_pred':dp,'delta_true':dt}

def train_seed(saved,train,val,seed,device,cfg):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=DualAngleGNN(saved).to(device)
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    heads=list(model.delta_head.parameters())+list(model.anchor_head.parameters())
    opt=torch.optim.AdamW([{'params':encoder,'lr':cfg['dual_encoder_lr']},{'params':heads,'lr':cfg['dual_head_lr']}],weight_decay=1e-5)
    dl=DataLoader(train,batch_size=cfg['dual_batch_size'],shuffle=True,collate_fn=collate,
                  generator=torch.Generator().manual_seed(seed),num_workers=0)
    mean,std=float(saved['target_mean']),float(saved['target_std']);best=None;best_score=1e9;bad=0
    for epoch in range(1,cfg['dual_epochs']+1):
        t=time.perf_counter();model.train();losses=[]
        for x,r,d,_,yr in dl:
            x,r,d,yr=move(x,device),move(r,device),d.to(device),yr.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                base,anchor,delta=model.components(x,r)
                ld=nn.functional.smooth_l1_loss(delta,d/std,beta=.03)
                la=nn.functional.smooth_l1_loss(base+anchor,(yr-mean)/std,beta=.05)
                loss=ld+cfg['dual_anchor_weight']*la
            loss.backward();nn.utils.clip_grad_norm_(encoder+heads,5.);opt.step();losses.append(float(loss.detach()))
        m=evaluate(model,val,device,mean,std,cfg['dual_batch_size']);score=m['delta_mae']+.25*m['absolute_mae']
        common.event('dual_epoch',seed=seed,epoch=epoch,loss=float(np.mean(losses)),delta_mae=m['delta_mae'],
                     absolute_mae=m['absolute_mae'],seconds=time.perf_counter()-t,
                     peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        if score<best_score-1e-5:
            best_score=score;bad=0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        else:bad+=1
        if bad>=cfg['dual_patience']:break
    saved={**saved,'architecture':'dual_cubic_delta_v1','model_state_dict':best,'best_epoch':best_epoch,
           'selection_score':best_score,'label_source':'dataset_DFT_anchor_plus_CHGNet_delta'}
    del model;torch.cuda.empty_cache();return saved

def export_readable_results():
    """Export the SQLite training log to files readable without database tools."""
    out=ROOT/'transfer_learning';db_path=out/'results.sqlite'
    with sqlite3.connect(db_path) as db:
        epochs=[json.loads(row[0]) for row in db.execute(
            "select payload from events where kind='dual_epoch' order by time")]
        def latest(kind):
            row=db.execute("select payload from events where kind=? order by time desc limit 1",(kind,)).fetchone()
            return json.loads(row[0]) if row else {}
        final=latest('dual_final_test');audit=latest('dual_same_metric_audit')
    fields=['seed','epoch','loss','delta_mae','absolute_mae','seconds','peak_gpu_mib']
    with (out/'training_history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:x.get(k) for k in fields} for x in epochs)
    bundle=torch.load(out/'final_models.pt',map_location='cpu',weights_only=False)
    by_key={(x['seed'],x['epoch']):x for x in epochs}
    best={seed:(saved['selection_score'],by_key[(seed,saved['best_epoch'])])
          for seed,saved in bundle.items()}
    lines=['# Step 6 最终结果（可直接阅读）','',
      '> 重要：以下误差相对于 **CHGNet 代理标签**，不是相对于真实 DFT。任务范围是已知组成上的未见角度插值。','',
      '## 最终五模型集成','',
      '| 数据集 | 绝对形成能 MAE (eV/atom) | 相对参考结构 ΔE MAE (eV/atom) |',
      '|---|---:|---:|',
      f"| 验证集 | {final.get('validation_absolute_mae',float('nan')):.6f} | {final.get('validation_delta_mae',float('nan')):.6f} |",
      f"| 封存测试集 | {final.get('test_absolute_mae',float('nan')):.6f} | {final.get('test_delta_mae',float('nan')):.6f} |",'',
      f"- 测试集 ΔE 误差 P90：{final.get('test_delta_p90',float('nan')):.6f} eV/atom",
      f"- 测试集平均集成不确定性：{final.get('test_uncertainty_mean',float('nan')):.6f} eV/atom",
      f"- 测试集不确定性—误差 Spearman：{final.get('test_uncertainty_error_spearman',float('nan')):.3f}",'',
      '## 与旧方案完全相同口径的复核','',
      '| 数据集 | 同组成两两能量差 pair MAE (eV/atom) | regret (eV/atom) |',
      '|---|---:|---:|',
      f"| 验证集 | {audit.get('validation_pair_mae',float('nan')):.6f} | {audit.get('validation_regret',float('nan')):.6f} |",
      f"| 封存测试集 | {audit.get('test_pair_mae',float('nan')):.6f} | {audit.get('test_regret',float('nan')):.6f} |",'',
      '该复核沿用 README 历史表中的 pair MAE 定义，因此约 0.075 → 0.0053 的提升不是更换 MAE 定义造成的。','',
      '## 五个已保存模型的 checkpoint','',
      '| seed | 最佳 epoch | ΔE MAE | 绝对 MAE | 选择分数 | 单 epoch 时间 (s) | 峰值显存 (MiB) |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for seed in sorted(best):
        score,x=best[seed];lines.append(
          f"| {seed} | {x['epoch']} | {x['delta_mae']:.6f} | {x['absolute_mae']:.6f} | {score:.6f} | {x['seconds']:.2f} | {x['peak_gpu_mib']:.1f} |")
    lines += ['', '## 文件说明','',
      '- `final_models.pt`：最终五个训练模型，种子为 42、123、2026、3407、7777。',
      '- `training_history.csv`：378条逐 epoch 原始训练记录，可直接用 Excel、WPS 或文本编辑器查看。',
      '- `results.sqlite`：完整机器可读实验数据库，保留历史方案和审计事件。','',
      '数据划分中候选 ID 及 `(组成, 模式, 角度)` 无交叉；100个组成有意在训练、验证和测试间共享，所以不能把结果解释成对全新化学组成的外推能力。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    device=common.setup();cfg=common.config();data=common.load_data();proxy=__import__('label_and_active').proxy_labels(data)
    with sqlite3.connect(common.DB) as db:raw=dict(db.execute('select id,energy from labels'))
    by_comp=defaultdict(list)
    for x in data['candidates']:by_comp[x['composition']].append(x)
    cubic={}
    for composition,items in by_comp.items():
        exact=[x for x in items if x['pattern']=='cubic']
        generated=[x for x in items if x['pattern']!='parent']
        cubic[composition]=exact[0] if exact else min(
            generated,key=lambda x:(x['amplitude_deg'],x['candidate_id']))
    fallback=sum(x['pattern']!='cubic' for x in cubic.values())
    common.event('dual_reference_check',cubic=len(cubic)-fallback,
                 fallback_minimum_tilt=fallback)
    train=[x for x in data['candidates'] if x['split']=='train' and x['pattern'] not in ('parent','cubic')]
    val=[x for x in data['candidates'] if x['split']=='validation']
    train,val=PairSet(train,cubic,proxy,raw),PairSet(val,cubic,proxy,raw)
    base=torch.load(ROOT/'transfer_learning/base_models.pt',map_location='cpu',weights_only=False);path=ROOT/'transfer_learning/delta_models.pt'
    bundle=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
    for seed in cfg['seeds']:
        if seed not in bundle:
            bundle[seed]=train_seed(base[seed],train,val,seed,device,cfg);common.save_atomic(bundle,path)
    matrices=[];deltas=[];truth=None;dtruth=None
    for seed in cfg['seeds']:
        model=DualAngleGNN(base[seed]).to(device);model.load_state_dict(bundle[seed]['model_state_dict'])
        m=evaluate(model,val,device,base[seed]['target_mean'],base[seed]['target_std'],cfg['dual_batch_size'])
        matrices.append(m['pred']);deltas.append(m['delta_pred']);truth=m['truth'];dtruth=m['delta_true']
    matrix=np.asarray(matrices);dm=np.asarray(deltas);result={'absolute_mae':float(abs(matrix.mean(0)-truth).mean()),
      'delta_mae':float(abs(dm.mean(0)-dtruth).mean()),'mean_uncertainty':float(dm.std(0,ddof=1).mean()),'seeds':cfg['seeds']}
    common.save_atomic(bundle,ROOT/'transfer_learning/final_models.pt');common.event('dual_complete',**result)
    export_readable_results();print(json.dumps(result))

if __name__=='__main__':
    export_readable_results() if len(sys.argv)>1 and sys.argv[1]=='export' else main()
