"""Five-seed single-graph absolute proxy-energy ablation."""
from __future__ import annotations
import csv, json, random, sqlite3, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "transfer_learning"), str(ROOT / "project")]
import train_formation as common
from graph_data import collate_graphs
from models import AngleGNN


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate(model, items, labels, device, mean, std, batch_size):
    model.eval(); pred=[]; truth=[]
    ds = common.ProxyDataset(items, labels)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_graphs)
    with torch.inference_mode():
        for batch in dl:
            target=batch["target"].numpy(); out=model(move(batch,device)).float()*std+mean
            pred.extend(out.cpu().numpy()); truth.extend(target)
    pred=np.asarray(pred); truth=np.asarray(truth)
    return float(np.abs(pred-truth).mean()), pred, truth


def train_seed(saved, train_items, val_items, proxy, seed, device, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=AngleGNN(**saved["model_config"]); model.load_state_dict(saved["model_state_dict"]); model.to(device)
    ds=common.ProxyDataset(train_items,proxy)
    dl=DataLoader(ds,batch_size=cfg["dual_batch_size"]*2,shuffle=True,collate_fn=collate_graphs,
                  generator=torch.Generator().manual_seed(seed),num_workers=0)
    mean,std=float(saved["target_mean"]),float(saved["target_std"])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["dual_encoder_lr"],weight_decay=1e-5)
    best=None; best_score=1e9; bad=0; best_epoch=0
    for epoch in range(1,cfg["dual_epochs"]+1):
        started=time.perf_counter(); model.train(); losses=[]
        for batch in dl:
            batch=move(batch,device); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=nn.functional.smooth_l1_loss(model(batch),(batch["target"]-mean)/std,beta=.05)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); losses.append(float(loss.detach()))
        val_mae,_,_=evaluate(model,val_items,proxy,device,mean,std,cfg["dual_batch_size"]*2)
        row={"seed":seed,"epoch":epoch,"loss":float(np.mean(losses)),"val_mae":val_mae,
             "seconds":time.perf_counter()-started,"peak_gpu_mib":torch.cuda.max_memory_allocated()/2**20}
        with (OUT/"training_history.csv").open("a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=row.keys());
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
        if val_mae < best_score-1e-5:
            best_score=val_mae; best_epoch=epoch; bad=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        if bad>=cfg["dual_patience"]: break
    out={**saved,"architecture":"single_graph_absolute_proxy_v1","model_state_dict":best,
         "best_epoch":best_epoch,"selection_score":best_score,
         "label_source":"CHGNet_proxy_absolute_formation_energy"}
    del model; torch.cuda.empty_cache(); return out


def main():
    device=common.setup(); cfg=common.config(); data=common.load_data()
    proxy=__import__("label_and_active").proxy_labels(data)
    train=[x for x in data["candidates"] if x["split"]=="train" and x["pattern"] not in ("parent","cubic")]
    val=[x for x in data["candidates"] if x["split"]=="validation"]
    test=[x for x in data["candidates"] if x["split"]=="test"]
    base=torch.load(ROOT/"transfer_learning/base_models.pt",map_location="cpu",weights_only=False)
    path=OUT/"single_graph_models.pt"; bundle=torch.load(path,map_location="cpu",weights_only=False) if path.exists() else {}
    for seed in cfg["seeds"]:
        if seed not in bundle:
            bundle[seed]=train_seed(base[seed],train,val,proxy,seed,device,cfg); common.save_atomic(bundle,path)
    preds=[]; val_preds=[]; test_truth=None
    for seed in cfg["seeds"]:
        model=AngleGNN(**base[seed]["model_config"]); model.load_state_dict(bundle[seed]["model_state_dict"]); model.to(device)
        _,vp,vt=evaluate(model,val,proxy,device,base[seed]["target_mean"],base[seed]["target_std"],cfg["dual_batch_size"]*2)
        _,tp,tt=evaluate(model,test,proxy,device,base[seed]["target_mean"],base[seed]["target_std"],cfg["dual_batch_size"]*2)
        val_preds.append(vp); preds.append(tp); test_truth=tt
        del model
    val_preds=np.asarray(val_preds); preds=np.asarray(preds)
    result={"experiment":"single_graph_absolute_proxy","seeds":cfg["seeds"],
            "train_structures":len(train),"validation_structures":len(val),"test_structures":len(test),
            "validation_absolute_mae":float(np.abs(val_preds.mean(0)-vt).mean()),
            "test_absolute_mae":float(np.abs(preds.mean(0)-test_truth).mean()),
            "test_mean_uncertainty":float(preds.std(0,ddof=1).mean()),
            "parameters_total":sum(p.numel() for p in AngleGNN(**base[cfg["seeds"][0]]["model_config"]).parameters())}
    (OUT/"single_graph_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    common.save_atomic(bundle,path); print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
