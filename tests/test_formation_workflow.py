"""Regression tests for target replacement, pair metrics and acquisition isolation."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'transfer_learning'))
import train_formation
import label_and_active


def test_proxy_dataset_does_not_mutate_original_graph():
    graph={'target':torch.tensor(.1)}
    item={'candidate_id':'a','graph':graph}
    ds=train_formation.ProxyDataset([item],{'a':-2.5})
    assert ds[0]['target'].item()==-2.5
    assert abs(graph['target'].item()-.1)<1e-6


def test_pair_error_is_offset_invariant_and_composition_balanced():
    items=[{'candidate_id':str(i),'composition':'A' if i<2 else 'B'} for i in range(5)]
    labels={str(i):float(i) for i in range(5)}
    matrix=np.array([[10.,11.2,12.,13.,14.]]*5)
    m=label_and_active.summarize(items,labels,matrix)
    assert abs(m['pair_mae']-.1)<1e-10
    assert abs(m['regret'])<1e-10
    assert m['proxy_formation_mae']>9


def test_acquisition_unique_deterministic_budget():
    items=[{'candidate_id':str(i),'composition':'A','pattern':'x','amplitude_deg':i} for i in range(10)]
    matrix=np.array([np.arange(10)*i for i in range(5)])
    for strategy in ('active','random'):
        first=label_and_active.acquisition(items,['0'],matrix,5,strategy,19)
        assert len(first)==len(set(first))==5 and '0' in first
        assert first==label_and_active.acquisition(items,['0'],matrix,5,strategy,19)


def test_small_uncertainty_does_not_certify_large_error():
    m={'pair_mae':.1,'regret':0.,'retention_ratio':1.,'accepted_pair_mae':.1,
       'pair_uncertainty_spearman':.8}
    assert not label_and_active.accepted(m)
