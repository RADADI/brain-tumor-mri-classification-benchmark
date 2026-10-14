#!/usr/bin/env python3
"""Verified VGG16 penultimate-feature t-SNE for the held-out test set.

Uses the same VGG16 architecture and eval preprocessing as the publication
benchmark. Optionally verifies predictions against the saved seed-123 CSV.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def device_auto():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def build_vgg16(num_classes=4):
    m=models.vgg16(weights=None)
    m.classifier[6]=nn.Linear(m.classifier[6].in_features,num_classes)
    return m

def load_state_strict(model,path,device):
    obj=torch.load(path,map_location=device)
    if isinstance(obj,dict):
        for key in ('model_state_dict','state_dict','model'):
            if key in obj and isinstance(obj[key],dict):
                obj=obj[key]; break
    if not isinstance(obj,dict): raise TypeError('Checkpoint does not contain a state_dict.')
    obj={k[7:] if k.startswith('module.') else k:v for k,v in obj.items()}
    model.load_state_dict(obj,strict=True)
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--test-dir', default='DatasetNew/Testing')
    ap.add_argument('--checkpoint', default='publication_outputs/benchmark_vgg16/seed_123/final_model.pth')
    ap.add_argument('--reference-predictions', default='publication_outputs/benchmark_vgg16/seed_123/test_predictions.csv')
    ap.add_argument('--output-dir', default='publication_outputs_ipas/feature_visualization')
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--perplexity',type=float,default=30.0)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tfm=transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    ds=datasets.ImageFolder(args.test_dir,transform=tfm)
    dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    if ds.classes != ['glioma','meningioma','notumor','pituitary']:
        raise RuntimeError(f'Unexpected class order: {ds.classes}')

    dev=device_auto(); print('Device:',dev)
    m=load_state_strict(build_vgg16(len(ds.classes)),args.checkpoint,dev).to(dev).eval()
    print('[PASS] Strict checkpoint load')

    feats=[]; labels=[]; probs=[]
    # VGG classifier: FC-ReLU-Dropout-FC-ReLU-Dropout-FC. Use output of ReLU at index 4.
    cache={}
    def hook(_m,_i,o): cache['feat']=o.detach()
    h=m.classifier[4].register_forward_hook(hook)
    with torch.no_grad():
        for x,y in dl:
            logits=m(x.to(dev))
            p=torch.softmax(logits,1)
            feats.append(cache['feat'].cpu().numpy())
            labels.append(y.numpy())
            probs.append(p.cpu().numpy())
    h.remove()
    X=np.concatenate(feats); y=np.concatenate(labels); P=np.concatenate(probs)
    pred=P.argmax(1)

    ref_path=Path(args.reference_predictions)
    pred_match=None; prob_maxdiff=None
    if ref_path.exists():
        ref=pd.read_csv(ref_path)
        if len(ref)!=len(ds): raise RuntimeError(f'Reference rows={len(ref)} but dataset={len(ds)}')
        pred_match=float(np.mean(ref['pred_index'].to_numpy()==pred))
        cols=[f'prob_{c}' for c in ds.classes]
        if all(c in ref.columns for c in cols):
            prob_maxdiff=float(np.max(np.abs(ref[cols].to_numpy()-P)))
        print(f'[PASS] prediction agreement vs saved CSV: {pred_match:.6f}')
        if pred_match != 1.0: raise RuntimeError('Predictions do not reproduce saved seed-123 run.')
        if prob_maxdiff is not None: print(f'[INFO] max probability difference: {prob_maxdiff:.8f}')

    ncomp=min(50,X.shape[1],X.shape[0]-1)
    Xp=PCA(n_components=ncomp,random_state=args.seed).fit_transform(X)
    Z=TSNE(n_components=2,random_state=args.seed,init='pca',learning_rate='auto',
           perplexity=args.perplexity,max_iter=1500).fit_transform(Xp)

    rows=[]
    for i,(path,label) in enumerate(ds.samples):
        rows.append({'file':Path(path).name,'true_index':int(y[i]),'true_class':ds.classes[y[i]],
                     'pred_index':int(pred[i]),'pred_class':ds.classes[pred[i]],
                     'correct':bool(pred[i]==y[i]),'tsne_1':float(Z[i,0]),'tsne_2':float(Z[i,1])})
    pd.DataFrame(rows).to_csv(out/'vgg16_tsne_points.csv',index=False)

    fig,ax=plt.subplots(figsize=(7.2,6.1))
    markers=['o','s','^','D']
    for c,name in enumerate(ds.classes):
        mask=y==c
        ax.scatter(Z[mask,0],Z[mask,1],s=14,alpha=.62,label=name,marker=markers[c])
    err=pred!=y
    if np.any(err): ax.scatter(Z[err,0],Z[err,1],s=34,facecolors='none',edgecolors='black',linewidths=.7,label='misclassified')
    ax.set_xlabel('t-SNE dimension 1'); ax.set_ylabel('t-SNE dimension 2')
    ax.set_title('VGG16 penultimate-feature representation of held-out MRI test images')
    ax.legend(fontsize=8,markerscale=1.4)
    fig.tight_layout(); fig.savefig(out/'vgg16_feature_tsne.png',dpi=300,bbox_inches='tight'); plt.close(fig)

    summary={
        'test_images':len(ds),'feature_dimension':int(X.shape[1]),'pca_dimension':int(ncomp),
        'tsne_seed':args.seed,'perplexity':args.perplexity,
        'prediction_agreement_with_saved_csv':pred_match,
        'max_probability_abs_difference':prob_maxdiff,
        'tsne_2d_silhouette_exploratory':float(silhouette_score(Z,y)),
        'note':'t-SNE is an exploratory visualization; distances and cluster shapes are not quantitative evidence of clinical separability.'
    }
    (out/'vgg16_feature_tsne_summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved:',out/'vgg16_feature_tsne.png')
    print('Saved:',out/'vgg16_tsne_points.csv')
    print('Saved:',out/'vgg16_feature_tsne_summary.json')

if __name__=='__main__': main()
