#!/usr/bin/env python3
"""Train ConvNeXt-Tiny and Swin-Tiny using the exact benchmark protocol.

Requires brain_tumor_publication_train_resume.py in the same project directory.
Outputs are isolated under publication_outputs_modern/ by default.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import brain_tumor_publication_train_resume as base

MODERN_MODELS=['convnext_tiny','swin_tiny']

def build_model_extended(model_name:str,num_classes:int,cfg,pretrained:bool=True):
    n=model_name.lower()
    if n not in MODERN_MODELS:
        return base._original_build_model(model_name,num_classes,cfg,pretrained)
    if n=='convnext_tiny':
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m=models.convnext_tiny(weights=weights)
        in_features=m.classifier[2].in_features
        m.classifier[2]=nn.Linear(in_features,num_classes)
        return m,224,'classifier'
    if n=='swin_tiny':
        weights=models.Swin_T_Weights.DEFAULT if pretrained else None
        m=models.swin_t(weights=weights)
        in_features=m.head.in_features
        m.head=nn.Linear(in_features,num_classes)
        return m,224,'head'
    raise ValueError(n)

def freeze_extended(model:nn.Module,model_name:str):
    n=model_name.lower()
    if n not in MODERN_MODELS: return base._original_freeze(model,model_name)
    for p in model.parameters(): p.requires_grad=False
    module=model.classifier if n=='convnext_tiny' else model.head
    for p in module.parameters(): p.requires_grad=True

def unfreeze_extended(model:nn.Module,model_name:str):
    n=model_name.lower()
    if n not in MODERN_MODELS: return base._original_unfreeze(model,model_name)
    for p in model.parameters(): p.requires_grad=False
    modules=[model.features[-2],model.features[-1],model.classifier] if n=='convnext_tiny' else [model.features[-2],model.features[-1],model.head]
    for module in modules:
        for p in module.parameters(): p.requires_grad=True

def config(name,args):
    return base.ExperimentConfig(
        name=f'benchmark_{name}', model_name=name,
        val_split=0.20,batch_size=args.batch_size,num_workers=args.num_workers,
        epochs_phase1=args.epochs_phase1,epochs_phase2=args.epochs_phase2,
        lr_phase1=args.lr_phase1,lr_phase2=args.lr_phase2,
        early_stop_patience=6,use_clahe_median=False,use_weighted_sampler=False,
        label_smoothing=0.0,optimizer_name='adam',scheduler_name='plateau',use_tta=False,
        rotation_degrees=15,jitter_brightness=0.15,jitter_contrast=0.15)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-dir',default='DatasetNew/Training')
    ap.add_argument('--test-dir',default='DatasetNew/Testing')
    ap.add_argument('--output-dir',default='publication_outputs_modern')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026])
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--num-workers',type=int,default=0)
    ap.add_argument('--epochs-phase1',type=int,default=15)
    ap.add_argument('--epochs-phase2',type=int,default=10)
    ap.add_argument('--lr-phase1',type=float,default=1e-3)
    ap.add_argument('--lr-phase2',type=float,default=1e-5)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--skip-integrity-check',action='store_true')
    args=ap.parse_args()

    train=Path(args.train_dir); test=Path(args.test_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not train.exists() or not test.exists(): raise FileNotFoundError('DatasetNew/Training or DatasetNew/Testing not found.')
    dev=base.get_device(); print('Device:',dev); print('Seeds:',args.seeds)

    if not args.skip_integrity_check:
        integ=base.dataset_integrity_check(train,test,out)
        print('Exact cross-split duplicate groups:',integ['exact_cross_split_duplicate_count'])
        if integ['exact_cross_split_duplicate_count']>0: raise RuntimeError('Cross-split duplicates detected; stop before manuscript use.')

    rows=[]
    for model_name in MODERN_MODELS:
        cfg=config(model_name,args)
        for seed in args.seeds:
            summary=out/cfg.name/f'seed_{seed}'/'run_summary.json'
            if args.resume and summary.exists():
                row=json.loads(summary.read_text()); rows.append(row); print('RESUME:',cfg.name,seed); base.aggregate_results(rows,out); continue
            row=base.run_experiment(cfg,seed,train,test,out,dev)
            rows.append(row); base.aggregate_results(rows,out)
    print('Completed modern baselines. Results:',out.resolve())

if __name__=='__main__':
    # Preserve original functions, then patch the base module used by run_experiment.
    base._original_build_model=base.build_model
    base._original_freeze=base.freeze_for_phase1
    base._original_unfreeze=base.unfreeze_for_phase2
    base.build_model=build_model_extended
    base.freeze_for_phase1=freeze_extended
    base.unfreeze_for_phase2=unfreeze_extended
    main()
