#!/usr/bin/env python3
"""External validation of the three VGG16 seed models on BDNeuro-MRI.

Dataset: BDNeuro-MRI (Mendeley Data V7, DOI 10.17632/zwr4ntf94j.7),
Epic & CSCR Hospital, Bangladesh. This script uses ONLY the provided external
TEST split and does no tuning on external labels.

It also checks exact SHA-256 overlap and optional perceptual-hash near overlap
between the internal 7,200-image dataset and the external test set.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

CLASSES=['glioma','meningioma','notumor','pituitary']
ALIASES={
    'glioma':'glioma','glioma_tumor':'glioma',
    'meningioma':'meningioma','meningioma_tumor':'meningioma',
    'pituitary':'pituitary','pituitary_tumor':'pituitary','pituitary_macroadenoma':'pituitary',
    'notumor':'notumor','no_tumor':'notumor','no tumor':'notumor','healthy':'notumor','normal':'notumor'
}
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def device_auto():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def find_external_test(root:Path):
    candidates=[root/'test',root/'Test',root/'Testing',root/'Brain_Tumor_MRI_Dataset_Final'/'test']
    for c in candidates:
        if c.exists(): return c
    # allow user to pass test directory itself
    if any(p.is_dir() for p in root.iterdir()): return root
    raise FileNotFoundError(f'Could not locate external test split under {root}')

def canonical(name):
    k=name.strip().lower().replace('-','_')
    return ALIASES.get(k)

class ExternalDataset(Dataset):
    def __init__(self,root,tfm):
        self.items=[]; self.tfm=tfm
        for folder in sorted(Path(root).iterdir()):
            if not folder.is_dir(): continue
            c=canonical(folder.name)
            if c is None: continue
            label=CLASSES.index(c)
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in EXTS: self.items.append((p,label,c))
        if not self.items: raise RuntimeError(f'No recognized class images found in {root}')
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y,_=self.items[i]
        im=Image.open(p).convert('RGB')
        return self.tfm(im),y,str(p)

def build_vgg16():
    m=models.vgg16(weights=None); m.classifier[6]=nn.Linear(m.classifier[6].in_features,4); return m

def load_strict(model,path,device):
    obj=torch.load(path,map_location=device)
    if isinstance(obj,dict):
        for k in ('model_state_dict','state_dict','model'):
            if k in obj and isinstance(obj[k],dict): obj=obj[k]; break
    obj={k[7:] if k.startswith('module.') else k:v for k,v in obj.items()}
    model.load_state_dict(obj,strict=True); return model

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def iter_images(root):
    return [p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in EXTS]

def overlap_checks(internal_roots,external_files,out):
    print('Checking exact SHA-256 overlap...')
    internal=[]
    for r in internal_roots: internal.extend(iter_images(r))
    ih=defaultdict(list)
    for p in internal: ih[sha256(p)].append(str(p))
    exact=[]
    for p in external_files:
        h=sha256(p)
        if h in ih:
            for q in ih[h]: exact.append({'external':str(p),'internal':q,'sha256':h})
    pd.DataFrame(exact,columns=['external','internal','sha256']).to_csv(out/'external_exact_overlap.csv',index=False)
    print('Exact cross-dataset matches:',len(exact))

    # Optional pHash near-overlap check.
    try:
        import imagehash
    except ImportError:
        print('[INFO] imagehash not installed; skipping pHash check. Install: python3 -m pip install ImageHash')
        return {'internal_images':len(internal),'external_images':len(external_files),'exact_matches':len(exact),'phash_checked':False}
    print('Computing perceptual hashes for near-duplicate check...')
    def ph(p): return int(str(imagehash.phash(Image.open(p).convert('L'))),16)
    int_hash=[(p,ph(p)) for p in internal]
    ext_hash=[(p,ph(p)) for p in external_files]
    near=[]
    for ep,eh in ext_hash:
        best_d=65; best_p=None
        for ip,ihv in int_hash:
            d=(eh^ihv).bit_count()
            if d<best_d: best_d=d; best_p=ip
            if d==0: break
        if best_d<=5: near.append({'external':str(ep),'internal':str(best_p),'phash_hamming':best_d})
    pd.DataFrame(near,columns=['external','internal','phash_hamming']).to_csv(out/'external_near_overlap_phash_le5.csv',index=False)
    print('External images with pHash distance <=5 to internal data:',len(near))
    return {'internal_images':len(internal),'external_images':len(external_files),'exact_matches':len(exact),'phash_checked':True,'near_matches_phash_le5':len(near)}

def evaluate(seed,checkpoint,loader,device,out):
    m=load_strict(build_vgg16(),checkpoint,device).to(device).eval()
    yt=[]; yp=[]; pr=[]; files=[]
    with torch.inference_mode():
        for x,y,p in loader:
            prob=torch.softmax(m(x.to(device)),1).cpu().numpy(); pred=prob.argmax(1)
            yt.extend(y.numpy()); yp.extend(pred); pr.append(prob); files.extend(p)
    y=np.asarray(yt); pred=np.asarray(yp); P=np.concatenate(pr)
    try: auc=float(roc_auc_score(label_binarize(y,classes=np.arange(4)),P,multi_class='ovr',average='macro'))
    except Exception: auc=float('nan')
    metrics={'seed':seed,'accuracy':float(accuracy_score(y,pred)),
             'precision_macro':float(precision_score(y,pred,average='macro',zero_division=0)),
             'recall_macro':float(recall_score(y,pred,average='macro',zero_division=0)),
             'f1_macro':float(f1_score(y,pred,average='macro',zero_division=0)),
             'auc_macro_ovr':auc,'test_samples':int(len(y))}
    df=pd.DataFrame({'file':files,'true_index':y,'true_class':[CLASSES[i] for i in y],
                     'pred_index':pred,'pred_class':[CLASSES[i] for i in pred]})
    for i,c in enumerate(CLASSES): df[f'prob_{c}']=P[:,i]
    df.to_csv(out/f'external_predictions_seed_{seed}.csv',index=False)
    cm=confusion_matrix(y,pred,labels=np.arange(4)); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'external_confusion_seed_{seed}.csv')
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--external-root',required=True,help='Extracted BDNeuro-MRI root or its test directory')
    ap.add_argument('--internal-train',default='DatasetNew/Training')
    ap.add_argument('--internal-test',default='DatasetNew/Testing')
    ap.add_argument('--checkpoint-root',default='publication_outputs/benchmark_vgg16')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026])
    ap.add_argument('--output-dir',default='publication_outputs_ipas/external_validation')
    ap.add_argument('--batch-size',type=int,default=32)
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    test_root=find_external_test(Path(args.external_root)); print('External test split:',test_root)
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=ExternalDataset(test_root,tfm); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    counts=pd.Series([c for _,_,c in ds.items]).value_counts().reindex(CLASSES,fill_value=0)
    print('External class counts:',counts.to_dict())
    counts.rename('count').to_csv(out/'external_class_counts.csv')

    overlap=overlap_checks([Path(args.internal_train),Path(args.internal_test)],[p for p,_,_ in ds.items],out)
    (out/'overlap_summary.json').write_text(json.dumps(overlap,indent=2))
    if overlap['exact_matches']>0:
        raise RuntimeError('Exact internal/external image overlap detected. Do not claim independent external validation.')
    if overlap.get('near_matches_phash_le5',0)>0:
        print('[WARNING] Near-duplicate candidates found. Review CSV before using external-validation results in manuscript.')

    dev=device_auto(); print('Evaluation device:',dev)
    rows=[]
    for seed in args.seeds:
        ck=Path(args.checkpoint_root)/f'seed_{seed}'/'final_model.pth'
        if not ck.exists(): raise FileNotFoundError(ck)
        row=evaluate(seed,ck,loader,dev,out); rows.append(row); print(row)
    df=pd.DataFrame(rows); df.to_csv(out/'external_validation_per_seed.csv',index=False)
    metrics=['accuracy','precision_macro','recall_macro','f1_macro','auc_macro_ovr']
    agg=[]
    for m in metrics:
        agg.append({'metric':m,'mean':float(df[m].mean()),'sample_sd':float(df[m].std(ddof=1))})
    pd.DataFrame(agg).to_csv(out/'external_validation_mean_std.csv',index=False)
    meta={
      'dataset':'BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification',
      'doi':'10.17632/zwr4ntf94j.7','external_test_split':str(test_root),
      'protocol':'zero-shot external evaluation of already-trained VGG16 seed models; no external fine-tuning or threshold tuning',
      'classes':CLASSES,'overlap_check':overlap}
    (out/'external_validation_protocol.json').write_text(json.dumps(meta,indent=2))
    print('Saved external-validation results to',out.resolve())

if __name__=='__main__': main()
