#!/usr/bin/env python3
"""Profile all eight architectures under a common pure-forward protocol.

Reports parameters, FP32 parameter memory, approximate GFLOPs (fvcore), and
batch-1 inference latency on the same local device without disk I/O.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import brain_tumor_publication_train_resume as base

MODELS=['resnet50','inceptionv3','efficientnetb0','densenet121','vgg16','vgg19','convnext_tiny','swin_tiny']

def build(name,cfg):
    if name in {'resnet50','inceptionv3','efficientnetb0','densenet121','vgg16','vgg19'}:
        return base.build_model(name,4,cfg,pretrained=False)[:2]
    if name=='convnext_tiny':
        m=models.convnext_tiny(weights=None); m.classifier[2]=nn.Linear(m.classifier[2].in_features,4); return m,224
    if name=='swin_tiny':
        m=models.swin_t(weights=None); m.head=nn.Linear(m.head.in_features,4); return m,224
    raise ValueError(name)

def ckpt_path(name,seed,old_root,new_root):
    root=new_root if name in {'convnext_tiny','swin_tiny'} else old_root
    return Path(root)/f'benchmark_{name}'/f'seed_{seed}'/'final_model.pth'

def load_strict(model,path,device):
    obj=torch.load(path,map_location=device)
    if isinstance(obj,dict):
        for k in ('model_state_dict','state_dict','model'):
            if k in obj and isinstance(obj[k],dict): obj=obj[k]; break
    obj={k[7:] if k.startswith('module.') else k:v for k,v in obj.items()}
    model.load_state_dict(obj,strict=True); return model

def sync(device):
    if device.type=='mps': torch.mps.synchronize()
    elif device.type=='cuda': torch.cuda.synchronize()

def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def flop_count(model,input_size):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        raise RuntimeError('Install fvcore first: python3 -m pip install fvcore') from e
    model=model.cpu().eval(); x=torch.randn(1,3,input_size,input_size)
    fl=FlopCountAnalysis(model,x)
    fl.unsupported_ops_warnings(False); fl.uncalled_modules_warnings(False)
    total=float(fl.total())
    unsupported={str(k):int(v) for k,v in fl.unsupported_ops().items()}
    return total/1e9,unsupported

def latency(model,input_size,device,warmup,repeats):
    model=model.to(device).eval(); x=torch.randn(1,3,input_size,input_size,device=device)
    with torch.inference_mode():
        for _ in range(warmup): _=model(x)
        sync(device)
        vals=[]
        for _ in range(repeats):
            sync(device); t0=time.perf_counter(); _=model(x); sync(device)
            vals.append((time.perf_counter()-t0)*1000.0)
    return vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old-root',default='publication_outputs')
    ap.add_argument('--new-root',default='publication_outputs_modern')
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--output-dir',default='publication_outputs_ipas/efficiency')
    ap.add_argument('--warmup',type=int,default=15)
    ap.add_argument('--repeats',type=int,default=60)
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dev=get_device(); print('Latency device:',dev)
    cfg=base.ExperimentConfig(name='profile',model_name='vgg16')
    rows=[]
    for name in MODELS:
        path=ckpt_path(name,args.seed,args.old_root,args.new_root)
        if not path.exists(): raise FileNotFoundError(f'Missing checkpoint for {name}: {path}')
        print('\nProfiling',name)
        m,input_size=build(name,cfg)
        m=load_strict(m,path,torch.device('cpu'))
        params=sum(p.numel() for p in m.parameters())
        param_mb=sum(p.numel()*p.element_size() for p in m.parameters())/(1024**2)
        checkpoint_mb=path.stat().st_size/(1024**2)
        gflops,unsupported=flop_count(m,input_size)
        # reload to ensure clean device state after CPU FLOPs analysis
        m,input_size=build(name,cfg); m=load_strict(m,path,torch.device('cpu'))
        vals=latency(m,input_size,dev,args.warmup,args.repeats)
        row={'model':name,'seed':args.seed,'input_size':input_size,'parameters':params,
             'parameters_millions':params/1e6,'fp32_parameter_memory_mb':param_mb,
             'checkpoint_size_mb':checkpoint_mb,'gflops_approx':gflops,
             'latency_device':str(dev),'batch_size':1,'warmup_runs':args.warmup,'timed_runs':args.repeats,
             'latency_mean_ms':float(np.mean(vals)),'latency_sd_ms':float(np.std(vals,ddof=1)),
             'latency_median_ms':float(np.median(vals)),'latency_p95_ms':float(np.percentile(vals,95)),
             'fvcore_unsupported_ops':json.dumps(unsupported,sort_keys=True)}
        rows.append(row); print(row)
        del m
        if dev.type=='mps': torch.mps.empty_cache()
        elif dev.type=='cuda': torch.cuda.empty_cache()
    df=pd.DataFrame(rows); df.to_csv(out/'efficiency_profile.csv',index=False)
    (out/'efficiency_protocol.txt').write_text(
        'Pure forward-pass latency, batch size 1, in-memory synthetic input, model.eval(), torch.inference_mode().\n'
        'Warm-up precedes timed repetitions. Device synchronization is used before/after each timed forward.\n'
        'GFLOPs are approximate fvcore forward-pass FLOPs; unsupported operations are recorded per model.\n'
        'Memory column is exact FP32 parameter storage, not peak accelerator runtime memory.\n')
    print('\nSaved:',out/'efficiency_profile.csv')

if __name__=='__main__': main()
