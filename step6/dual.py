"""Five-seed paired-graph model for cubic-anchored tilt formation energies."""
from __future__ import annotations
import json, random, sqlite3, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'project'),str(ROOT/'step5')]
from graph_data import collate_graphs
from models import AngleGNN
import step5 as common

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

def main():
    device=common.setup();cfg=common.config();data=common.load_data();proxy=__import__('step6').proxy_labels(data)
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
    base=torch.load(ROOT/'step5/models.pt',map_location='cpu',weights_only=False);path=ROOT/'step6/dual_models.pt'
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
    common.save_atomic(bundle,ROOT/'step6/models.pt');common.event('dual_complete',**result);print(json.dumps(result))

if __name__=='__main__':main()
