#!/usr/bin/env python3
"""Combine completed old/new benchmark, efficiency, t-SNE and external-validation outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

DISPLAY={
'resnet50':'ResNet50','inceptionv3':'InceptionV3','efficientnetb0':'EfficientNetB0','densenet121':'DenseNet121',
'vgg16':'VGG16','vgg19':'VGG19','convnext_tiny':'ConvNeXt-Tiny','swin_tiny':'Swin-Tiny'}

def aggregate(df):
    rows=[]
    for model,g in df.groupby('model'):
        if not str(model).lower() in DISPLAY: continue
        row={'model':str(model).lower(),'n_seeds':len(g)}
        for col in ['accuracy','f1_macro','auc_macro_ovr']:
            row[col+'_mean']=g[col].mean(); row[col+'_sd']=g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old-runs',default='publication_outputs/all_runs_summary.csv')
    ap.add_argument('--new-runs',default='publication_outputs_modern/all_runs_summary.csv')
    ap.add_argument('--efficiency',default='publication_outputs_ipas/efficiency/efficiency_profile.csv')
    ap.add_argument('--external',default='publication_outputs_ipas/external_validation/external_validation_mean_std.csv')
    ap.add_argument('--tsne-summary',default='publication_outputs_ipas/feature_visualization/vgg16_feature_tsne_summary.json')
    ap.add_argument('--output-dir',default='publication_outputs_ipas/final_tables')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

    old=pd.read_csv(a.old_runs); new=pd.read_csv(a.new_runs)
    # Keep only benchmark rows from old output (exclude D0-D5 ablation rows).
    old=old[old['experiment'].astype(str).str.startswith('benchmark_')]
    comb=pd.concat([old,new],ignore_index=True)
    agg=aggregate(comb)
    missing=set(DISPLAY)-set(agg['model'])
    if missing: raise RuntimeError(f'Missing benchmark results for: {sorted(missing)}')
    if not (agg['n_seeds']==3).all(): raise RuntimeError('Every benchmark model must have exactly three seed runs before final manuscript use.')

    eff=pd.read_csv(a.efficiency)
    merged=agg.merge(eff,on='model',how='left',validate='one_to_one')
    if merged['gflops_approx'].isna().any(): raise RuntimeError('Efficiency table is incomplete.')
    merged['display_model']=merged['model'].map(DISPLAY)
    order=['resnet50','inceptionv3','efficientnetb0','densenet121','vgg16','vgg19','convnext_tiny','swin_tiny']
    merged['order']=merged['model'].map({m:i for i,m in enumerate(order)})
    merged=merged.sort_values('order').drop(columns='order')
    merged.to_csv(out/'eight_model_benchmark_efficiency.csv',index=False)

    # Publication-ready compact table fragment (percentages).
    lines=['\\begin{tabular}{lrrrrrr}','\\toprule','Model & Acc. & Macro-F1 & AUC & Params (M) & GFLOPs & ms/img \\\\','\\midrule']
    for _,r in merged.iterrows():
        lines.append(f"{r['display_model']} & {100*r['accuracy_mean']:.2f}$\\pm${100*r['accuracy_sd']:.2f} & {100*r['f1_macro_mean']:.2f}$\\pm${100*r['f1_macro_sd']:.2f} & {100*r['auc_macro_ovr_mean']:.2f}$\\pm${100*r['auc_macro_ovr_sd']:.2f} & {r['parameters_millions']:.2f} & {r['gflops_approx']:.2f} & {r['latency_mean_ms']:.1f} \\\\")
    lines += ['\\bottomrule','\\end{tabular}']
    (out/'eight_model_table_fragment.tex').write_text('\n'.join(lines))

    ext=pd.read_csv(a.external)
    ext.to_csv(out/'external_validation_summary.csv',index=False)
    tsne=json.loads(Path(a.tsne_summary).read_text())
    (out/'feature_visualization_summary.json').write_text(json.dumps(tsne,indent=2))
    checklist={
      'eight_models_three_seeds':True,
      'efficiency_complete':True,
      'external_validation_present':True,
      'tsne_present':True,
      'ready_for_manuscript_update':True}
    (out/'completion_check.json').write_text(json.dumps(checklist,indent=2))
    print('All requested experiment outputs are complete.')
    print('Final tables:',out.resolve())

if __name__=='__main__': main()
