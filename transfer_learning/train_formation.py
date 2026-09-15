"""Formation-energy training, shared evaluation, and CIF inference.

Commands: train, infer --cif PATH. Data and weights stay in compact bundles.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sqlite3, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from graph_data import collate_graphs, GraphConfig, structure_to_graph
from models import AngleGNN
from graph_data import TRIPLET_B_O_B, ROLE_B, ROLE_O
from train_angle_ensemble import CachedGraphDataset, ShardBatchSampler
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
DB = HERE / "results.sqlite"

def config():
    return json.loads(CONFIG.read_text())

def load_data():
    return torch.load(HERE / "data.pt", map_location="cpu", weights_only=False)

def save_atomic(obj, path):
    path = Path(path); temp = path.with_suffix(".tmp")
    torch.save(obj, temp); temp.replace(path)

def event(kind, **values):
    with sqlite3.connect(DB, timeout=60) as db:
        db.execute("CREATE TABLE IF NOT EXISTS events(time REAL, kind TEXT, payload TEXT)")
        db.execute("INSERT INTO events VALUES(?,?,?)", (time.time(), kind, json.dumps(values)))
    print(kind, json.dumps(values), flush=True)

def setup():
    cfg=config()
    torch.set_num_threads(cfg["threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU training fallback")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")

class FormationDataset(CachedGraphDataset):
    def __getitem__(self, i):
        graph = dict(super().__getitem__(i))
        graph["target"] = torch.tensor(float(self.rows[i]["formation_energy_per_atom"]), dtype=torch.float32)
        return graph

class ProxyDataset(Dataset):
    def __init__(self, candidates, labels):
        self.candidates, self.labels = candidates, labels
    def __len__(self): return len(self.candidates)
    def __getitem__(self,i):
        item=self.candidates[i]; graph=dict(item["graph"])
        graph["target"]=torch.tensor(self.labels[item["candidate_id"]], dtype=torch.float32)
        return graph

class PairDataset(Dataset):
    """Pairs from one fixed-cell generated scan; the dataset parent is excluded."""
    def __init__(self, candidates, labels):
        groups=defaultdict(list)
        for item in candidates:
            if item["pattern"] != "parent": groups[item["composition"]].append(item)
        self.pairs=[]
        for group in groups.values():
            group.sort(key=lambda x:(x["pattern"]!="cubic",x["amplitude_deg"],x["candidate_id"]))
            if len(group)>1:
                anchor=group[0]
                self.pairs.extend((anchor,item) for item in group[1:])
        self.labels=labels
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        left,right=self.pairs[i]
        return dict(left["graph"]),dict(right["graph"]),torch.tensor(
            self.labels[left["candidate_id"]]-self.labels[right["candidate_id"]],dtype=torch.float32)

def collate_pairs(items):
    left,right,target=zip(*items)
    return collate_graphs(left),collate_graphs(right),torch.stack(target)

def base_dataset(rows):
    with (ROOT/"dataset/processed/structure_index.csv").open() as f:
        index={r["structure_id"]:r for r in csv.DictReader(f)}
    return FormationDataset(rows, index)

def loader(dataset, batch_size, seed=0, shuffle=False):
    if isinstance(dataset, CachedGraphDataset):
        sampler=ShardBatchSampler(dataset,batch_size,seed,shuffle)
        return DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_graphs,num_workers=0),sampler
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_graphs,
                      generator=torch.Generator().manual_seed(seed),num_workers=0),None

def move(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

def make_model(saved,device):
    architecture=saved.get("architecture")
    if architecture=="frozen_base_angle_residual":model=ResidualAngleGNN(saved["model_config"])
    elif architecture=="anchored_geometry_residual_v2":model=AnchoredGeometryResidual(saved["model_config"])
    else:model=AngleGNN(**saved["model_config"])
    model=model.to(device)
    model.load_state_dict(saved["model_state_dict"])
    return model

class ResidualAngleGNN(torch.nn.Module):
    """Frozen formation model plus a zero-initialized geometry correction."""
    def __init__(self, model_config, base_state=None):
        super().__init__();self.base=AngleGNN(**model_config);self.correction=AngleGNN(**model_config)
        if base_state is not None:self.base.load_state_dict(base_state)
        for parameter in self.base.parameters():parameter.requires_grad_(False)
        last=[x for x in self.correction.head if isinstance(x,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight);torch.nn.init.zeros_(last.bias)
    def forward(self,batch):return self.base(batch)+self.correction(batch)

def geometry_features(batch):
    """Invariant per-crystal [B-O-B tilt mean/std, B-O length mean/std]."""
    n=int(batch["batch"].max())+1
    def stats(values,index):
        out=values.new_zeros((n,));out.index_add_(0,index,values)
        count=torch.bincount(index,minlength=n).clamp_min(1).to(values.dtype)
        mean=out/count;sq=values.new_zeros((n,));sq.index_add_(0,index,values.square())
        return mean,(sq/count-mean.square()).clamp_min(0).sqrt()
    tj=batch["triplet_edge_index"][0]
    tgraph=batch["batch"][batch["edge_index"][0,tj]]
    mask=batch["triplet_type"]==TRIPLET_B_O_B
    angle=torch.rad2deg(torch.acos(batch["triplet_cosine"][mask].clamp(-1,1)))
    amean,astd=stats(angle,tgraph[mask])
    center,neighbor=batch["edge_index"]
    roles=batch["role"];bo=((roles[center]==ROLE_B)&(roles[neighbor]==ROLE_O))
    length=batch["distance"][bo];lmean,lstd=stats(length,batch["batch"][center[bo]])
    return torch.stack(((180.-amean)/60.,astd/30.,lmean/3.,lstd),dim=1)

class AnchoredGeometryResidual(torch.nn.Module):
    """Frozen formation baseline; explicit geometry correction is exactly zero at 180°."""
    def __init__(self,model_config,base_state=None):
        super().__init__();self.base=AngleGNN(**model_config);self.encoder=AngleGNN(**model_config)
        if base_state is not None:self.base.load_state_dict(base_state)
        for p in self.base.parameters():p.requires_grad_(False)
        self.geometry_head=torch.nn.Sequential(torch.nn.LayerNorm(15),torch.nn.Linear(15,64),
                                               torch.nn.SiLU(),torch.nn.Linear(64,1))
        last=self.geometry_head[-1];torch.nn.init.zeros_(last.weight);torch.nn.init.zeros_(last.bias)
        last_encoder=[x for x in self.encoder.head if isinstance(x,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last_encoder.weight);torch.nn.init.zeros_(last_encoder.bias)
    def residual(self,batch):
        geo=geometry_features(batch);gate=geo[:,0].clamp_min(0).clamp_max(1)
        explicit=torch.cat((geo,gate[:,None],batch["composition_features"]),1)
        return gate*(self.encoder(batch)+self.geometry_head(explicit).squeeze(1))
    def forward(self,batch):return self.base(batch)+self.residual(batch)

@torch.inference_mode()
def predict(model, iterable, device, mean, std):
    model.eval(); values=[]; target=[]
    for batch in iterable:
        batch=move(batch,device)
        values.extend((model(batch).float()*std+mean).cpu().tolist())
        target.extend(batch["target"].cpu().tolist())
    return np.array(values),np.array(target)

def _scan_pair_mae(pred,candidates,labels):
    groups=defaultdict(list)
    for i,item in enumerate(candidates):
        if item["pattern"] != "parent": groups[item["composition"]].append(i)
    errors=[]
    truth=np.array([labels[x["candidate_id"]] for x in candidates])
    for ix in groups.values():
        if len(ix)<2: continue
        a,b=np.triu_indices(len(ix),1);left=np.asarray(ix)[a];right=np.asarray(ix)[b]
        errors.extend(abs((pred[left]-pred[right])-(truth[left]-truth[right])))
    return float(np.mean(errors)) if errors else float("inf")

def fit(saved, train_set, val_set, device, tag, epochs, patience, lr, proxy=None, val_candidates=None, val_labels=None):
    cfg=config(); seed=int(saved["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if proxy is not None and saved.get("architecture")!="anchored_geometry_residual_v2":
        model=AnchoredGeometryResidual(saved["model_config"],saved["model_state_dict"]).to(device)
        saved={**saved,"architecture":"anchored_geometry_residual_v2"}
    else:model=make_model(saved,device)
    mean,std=float(saved["target_mean"]),float(saved["target_std"])
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=.5)
    train_loader,sampler=loader(train_set,cfg["batch_size"],seed,True)
    val_loader,_=loader(val_set,cfg["batch_size"])
    proxy_loader=pair_loader=None
    if proxy is not None:
        proxy_loader,_=loader(proxy,cfg["batch_size"],seed,True)
        pairs=PairDataset(proxy.candidates,proxy.labels)
        if len(pairs):
            pair_loader=DataLoader(pairs,batch_size=cfg["batch_size"]//2,shuffle=True,
                collate_fn=collate_pairs,generator=torch.Generator().manual_seed(seed),num_workers=0)
    resume=HERE/"progress.pt"
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_pred,initial_true=predict(model,val_loader,device,mean,std)
    initial_formation_mae=float(np.mean(abs(initial_pred-initial_true)))
    best_loss=initial_formation_mae
    if val_candidates is not None:
        vp,_=predict(model,loader(ProxyDataset(val_candidates,val_labels),cfg["batch_size"])[0],device,mean,std)
        best_loss = _scan_pair_mae(vp,val_candidates,val_labels)
    bad=0; start=0; best_epoch=0
    if resume.exists():
        state=torch.load(resume,map_location="cpu",weights_only=False)
        if state["tag"]==tag:
            model.load_state_dict(state["current"]);opt.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]);best=state["best"];best_loss=state["best_loss"]
            bad=state["bad"];start=state["epoch"];best_epoch=state["best_epoch"]
            torch.set_rng_state(state["rng"]);torch.cuda.set_rng_state(state["cuda_rng"])
    for epoch in range(start+1,epochs+1):
        t=time.perf_counter(); model.train()
        if sampler: sampler.set_epoch(epoch)
        losses=[]
        # Separate mean losses in common normalized units; average proxy gradients
        # per replay batch, so datasets of different sizes don't silently change weight.
        pit=iter(proxy_loader) if proxy_loader is not None else None
        qit=iter(pair_loader) if pair_loader is not None else None
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            batch=move(batch,device)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if isinstance(model,AnchoredGeometryResidual):
                    loss=cfg["residual_replay_weight"]*torch.nn.functional.smooth_l1_loss(
                        model.residual(batch),torch.zeros_like(batch["target"]),beta=.05)
                elif isinstance(model,ResidualAngleGNN):
                    loss=torch.nn.functional.smooth_l1_loss(model.correction(batch),torch.zeros_like(batch["target"]),beta=.05)
                else:
                    loss=torch.nn.functional.smooth_l1_loss(model(batch),(batch["target"]-mean)/std,beta=.1)
                if pit is not None:
                    try: pb=next(pit)
                    except StopIteration: pit=iter(proxy_loader);pb=next(pit)
                    pb=move(pb,device)
                    loss=loss+cfg["proxy_absolute_weight"]*torch.nn.functional.smooth_l1_loss(
                        model(pb),(pb["target"]-mean)/std,beta=.1)
                if qit is not None:
                    try: left,right,delta=next(qit)
                    except StopIteration: qit=iter(pair_loader);left,right,delta=next(qit)
                    left,right,delta=move(left,device),move(right,device),delta.to(device)
                    loss=loss+cfg["pair_weight"]*torch.nn.functional.smooth_l1_loss(
                        model(left)-model(right),delta/std,beta=.05)
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.)
            opt.step();losses.append(loss.detach().float().item())
        pv,tv=predict(model,val_loader,device,mean,std)
        val_mae=float(np.mean(abs(pv-tv)));score=val_mae;proxy_mae=None;pair_mae=None
        if val_candidates is not None:
            vp,_=predict(model,loader(ProxyDataset(val_candidates,val_labels),cfg["batch_size"])[0],device,mean,std)
            proxy_mae=float(np.mean(abs(vp-np.array([val_labels[x["candidate_id"]] for x in val_candidates]))))
            pair_mae=_scan_pair_mae(vp,val_candidates,val_labels)
            score=pair_mae+5.*max(0.,val_mae-initial_formation_mae*cfg["retention_ratio_limit"])
        scheduler.step(score)
        if score<best_loss-1e-6:
            best_loss=score;best_epoch=epoch;bad=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        save_atomic({"tag":tag,"epoch":epoch,"current":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                     "optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"best":best,"best_loss":best_loss,
                     "best_epoch":best_epoch,"bad":bad,"rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()},resume)
        event("epoch",tag=tag,epoch=epoch,train_loss=float(np.mean(losses)),formation_val_mae=val_mae,
              proxy_val_mae=proxy_mae,scan_pair_val_mae=pair_mae,seconds=time.perf_counter()-t,
              peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        if bad>=patience: break
    result={k:v for k,v in saved.items() if k not in ("validation_metrics","test_metrics","epoch")}
    result.update(model_state_dict=best,best_epoch=best_epoch,selection_score=best_loss,target="formation_energy_per_atom",
                  label_source="dataset_DFT" if proxy is None else "dataset_DFT_plus_CHGNet_proxy",unit="eV/atom")
    del model,opt;torch.cuda.empty_cache()
    return result

def train():
    device=setup();cfg=config();data=load_data()
    graph_manifest=json.loads((ROOT/"dataset/processed/angle_graph_shards/manifest.json").read_text())
    if graph_manifest["config"]!=data["graph_config"]:raise ValueError("cache graph config mismatch")
    train_rows=[r for r in data["rows"] if r["formation_split"]=="train"]
    val_rows=[r for r in data["rows"] if r["formation_split"]=="validation"]
    train_set=base_dataset(train_rows);val_set=base_dataset(val_rows)
    y=np.array([float(r["formation_energy_per_atom"]) for r in train_rows])
    path=HERE/"base_models.pt"
    bundle=torch.load(path,map_location="cpu",weights_only=False) if path.exists() else {}
    for seed in cfg["seeds"]:
        if seed in bundle: continue
        saved=torch.load(ROOT/f"outputs/angle_ensemble/seed_{seed}/best_model.pt",map_location="cpu",weights_only=False)
        torch.manual_seed(seed)
        model=make_model(saved,torch.device("cpu"))
        for layer in model.head:
            if isinstance(layer,torch.nn.Linear):layer.reset_parameters()
        saved={**saved,"model_state_dict":model.state_dict(),"target_mean":float(y.mean()),"target_std":float(y.std()),
               "target":"formation_energy_per_atom","graph_config":data["graph_config"],"split_policy":"composition_disjoint_v1"}
        del model
        event("baseline_start",seed=seed,train=len(train_set),validation=len(val_set))
        bundle[seed]=fit(saved,train_set,val_set,device,f"baseline_{seed}",cfg["baseline_epochs"],
                         cfg["baseline_patience"],cfg["baseline_lr"])
        save_atomic(bundle,path)
    event("baseline_complete",seeds=list(bundle))

def infer(path, a_elements=None, b_elements=None, reference_path=None):
    from pymatgen.core import Structure
    device=setup();data=load_data();s=Structure.from_file(path);s.remove_oxidation_states()
    matching=[r for r in data["rows"] if s.composition.reduced_composition==
              __import__("pymatgen.core",fromlist=["Composition"]).Composition(r["formula"]).reduced_composition]
    if a_elements or b_elements:
        if not a_elements or not b_elements:raise ValueError('provide both --a-elements and --b-elements')
        av=a_elements.split(',');bv=b_elements.split(',')
        if not 1<=len(av)<=2 or not 1<=len(bv)<=2:raise ValueError('one or two elements per sublattice')
        row={'material_id':'user_cif','a1_element':av[0],'a2_element':av[-1],
             'b1_element':bv[0],'b2_element':bv[-1],'goldschmidt_t':0.,'bartel_tau':0.}
        for site in ('a1','a2','b1','b2'):row[site+'_oxidation_state']=0.
        # These unused composition-baseline features are placeholders; AngleGNN uses z/distance/angles.
    else:
        if not matching:raise ValueError('unknown composition: provide --a-elements and --b-elements')
        roles={(tuple(sorted({r['a1_element'],r['a2_element']})),tuple(sorted({r['b1_element'],r['b2_element']}))) for r in matching}
        if len(roles)>1:raise ValueError('ambiguous A/B assignment: provide --a-elements and --b-elements')
        row=matching[0]
    g=structure_to_graph(s,row,GraphConfig(**data["graph_config"]))
    model_path=HERE/"final_models.pt"
    if not model_path.exists():model_path=HERE/"base_models.pt"
    bundle=torch.load(model_path,map_location="cpu",weights_only=False);values=[]
    architecture=next(iter(bundle.values())).get("architecture")
    if architecture=="dual_cubic_delta_v1":
        if not reference_path:raise ValueError("dual model requires --reference-cif (normally the same-composition cubic structure)")
        reference=Structure.from_file(reference_path);reference.remove_oxidation_states()
        rg=structure_to_graph(reference,row,GraphConfig(**data["graph_config"]))
        sys.path.insert(0,str(HERE));from train_delta import DualAngleGNN
        bases=torch.load(HERE/'base_models.pt',map_location='cpu',weights_only=False);deltas=[]
        xb,rb=collate_graphs([g]),collate_graphs([rg])
        for seed,saved in bundle.items():
            model=DualAngleGNN(bases[seed]).to(device);model.load_state_dict(saved['model_state_dict'])
            xb2,rb2=move(xb,device),move(rb,device)
            with torch.inference_mode():
                base,anchor,delta=model.components(xb2,rb2)
                values.append(float(((base+anchor+delta)*saved['target_std']+saved['target_mean']).cpu()[0]))
                deltas.append(float((delta*saved['target_std']).cpu()[0]))
        print(json.dumps({"formation_ev_atom":float(np.mean(values)),"ensemble_std_ev_atom":float(np.std(values,ddof=1)),
                          "delta_from_reference_ev_atom":float(np.mean(deltas)),
                          "delta_ensemble_std_ev_atom":float(np.std(deltas,ddof=1)),"seeds":values,
                          "label_source":"dataset_DFT_anchor_plus_CHGNet_delta"}))
        return
    for saved in bundle.values():
        model=make_model(saved,device)
        p,_=predict(model,[collate_graphs([g])],device,saved["target_mean"],saved["target_std"]);values.append(float(p[0]))
    print(json.dumps({"formation_ev_atom":float(np.mean(values)),"ensemble_std_ev_atom":float(np.std(values,ddof=1)),
                      "seeds":values,"label_source":next(iter(bundle.values()))["label_source"]}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["train","infer"])
    parser.add_argument("--cif");parser.add_argument('--reference-cif');parser.add_argument('--a-elements');parser.add_argument('--b-elements');args=parser.parse_args()
    train() if args.action=="train" else infer(args.cif,args.a_elements,args.b_elements,args.reference_cif)
