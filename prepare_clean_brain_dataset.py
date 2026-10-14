#!/usr/bin/env python3
"""Prepare an exact-deduplicated, group-safe four-class MRI dataset for review.

Never changes DatasetNew. Preserves the supplied Testing membership, removes
identical files within each source partition, and splits unique Training images
by SHA-256 groups. Generates a pHash review report if Pillow and ImageHash are
installed. IMPORTANT: pHash matches are only candidates; this does not establish
patient independence or clearance for publication/retraining.

Usage:
  python3 prepare_clean_brain_dataset.py --source DatasetNew --output DatasetNew_Clean --seed 42
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SPLITS = ('Training', 'Testing')

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def enumerate_images(root):
    records = []
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            raise SystemExit(f'Missing required folder: {d}')
        for cls_dir in sorted(d.iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in EXT:
                    records.append({'source_split': split, 'class': cls_dir.name,
                                    'path': str(p.resolve()), 'sha256': file_hash(p)})
    return records

def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='DatasetNew')
    ap.add_argument('--output', default='DatasetNew_Clean')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--validation-fraction', type=float, default=.20)
    ap.add_argument('--phash-threshold', type=int, default=5)
    args = ap.parse_args()
    src = Path(args.source).resolve()
    dst = Path(args.output).resolve()
    if not src.is_dir():
        raise SystemExit(f'Source folder not found: {src}')
    if src == dst or src in dst.parents or dst in src.parents:
        raise SystemExit('Source and output must be separate folders; nested paths are not permitted.')
    if dst.exists():
        raise SystemExit(f'Output exists: {dst}. Choose a NEW output folder; nothing was overwritten.')
    if not 0 < args.validation_fraction < 1:
        raise SystemExit('Validation fraction must be between 0 and 1.')
    records = enumerate_images(src)
    if not records:
        raise SystemExit('No images found.')
    classes = sorted(set(r['class'] for r in records))
    if len(classes) != 4:
        raise SystemExit(f'Expected 4 class folders, found {classes}.')
    hmap = defaultdict(list)
    for r in records: hmap[r['sha256']].append(r)
    conflict = {h: v for h, v in hmap.items() if len({x['source_split'] for x in v}) > 1}
    label_conflict = {h: v for h, v in hmap.items() if len({x['class'] for x in v}) > 1}
    if conflict or label_conflict:
        raise SystemExit(f'STOP: {len(conflict)} exact groups cross supplied Training/Testing and '
                         f'{len(label_conflict)} groups have conflicting labels. Manual review required.')
    groups = defaultdict(list)
    for h, members in hmap.items():
        r = members[0]
        groups[(r['source_split'], r['class'])].append((h, members))
    chosen = []
    for cls in classes:
        train_groups = sorted(groups.get(('Training', cls), []), key=lambda x: x[0])
        random.Random(f'{args.seed}:{cls}').shuffle(train_groups)
        n_val = round(len(train_groups) * args.validation_fraction)
        for index, (h, members) in enumerate(train_groups):
            chosen.append(('Validation' if index < n_val else 'Training', cls, h, members))
        for h, members in groups.get(('Testing', cls), []):
            chosen.append(('Testing', cls, h, members))
    # Create file tree without changing original files. Keep one canonical copy per exact hash.
    dst.mkdir(parents=True)
    rows = []
    for split, cls, h, members in chosen:
        p = Path(sorted(x['path'] for x in members)[0])
        out_dir = dst / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f'{h[:20]}{p.suffix.lower()}'
        shutil.copy2(p, target)
        rows.append({'split': split, 'class': cls, 'sha256': h, 'original_path': str(p),
                     'clean_path': str(target), 'exact_copies_in_original': len(members)})
    write_csv(dst / 'clean_split_manifest.csv', rows,
              ['split', 'class', 'sha256', 'original_path', 'clean_path', 'exact_copies_in_original'])
    partition_hash = defaultdict(set)
    for r in rows: partition_hash[r['sha256']].add(r['split'])
    if any(len(v) > 1 for v in partition_hash.values()):
        raise RuntimeError('Unexpected exact hash crossing partitions; stop.')
    counts = Counter((r['split'], r['class']) for r in rows)
    summary = {
        'status': 'EXACT_DEDUP_COMPLETE__NEAR_DUPLICATE_AND_PATIENT_REVIEW_REQUIRED',
        'source_count': len(records), 'unique_sha256_images': len(hmap),
        'removed_exact_copies': len(records) - len(hmap),
        'cross_original_train_test_exact_groups': len(conflict),
        'cross_clean_split_exact_groups': 0, 'seed': args.seed,
        'validation_fraction': args.validation_fraction,
        'counts': {split: {cls: counts[(split, cls)] for cls in classes}
                   for split in ('Training', 'Validation', 'Testing')},
        'warnings': ['Original held-out Test membership retained, but redundant exact files deduplicated.',
                     'The number of samples changes. Prior results MUST NOT be reused.',
                     'Near duplicates and patient overlap are NOT ruled out.',
                     'This data is not cleared for publication just because exact duplicates are absent.']
    }
    try:
        from PIL import Image
        import imagehash
        fingerprints = []
        for r in rows:
            with Image.open(r['clean_path']) as im:
                ph = int(str(imagehash.phash(im.convert('L'))), 16)
            fingerprints.append((ph, r))
        # Report a single nearest candidate for each image in later splits; avoid huge pair lists.
        suspects = []
        for i, (ha, ra) in enumerate(fingerprints):
            for hb, rb in fingerprints[i+1:]:
                if ra['split'] == rb['split']: continue
                d = (ha ^ hb).bit_count()
                if d <= args.phash_threshold:
                    suspects.append({'split_a': ra['split'], 'split_b': rb['split'],
                                     'class_a': ra['class'], 'class_b': rb['class'],
                                     'file_a': ra['clean_path'], 'file_b': rb['clean_path'],
                                     'phash_distance': d})
        write_csv(dst / 'near_duplicate_candidates_REVIEW.csv', suspects,
                  ['split_a', 'split_b', 'class_a', 'class_b', 'file_a', 'file_b', 'phash_distance'])
        summary['phash_checked'] = True
        summary['phash_candidate_pairs_cross_splits'] = len(suspects)
        summary['phash_threshold'] = args.phash_threshold
    except ImportError:
        summary['phash_checked'] = False
        summary['warnings'].append('Install Pillow and ImageHash; rerun into NEW output folder for candidate pHash audit.')
    (dst / 'integrity_report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nDO NOT TRAIN OR CLAIM NO LEAKAGE until near-duplicate and patient review is completed.')

if __name__ == '__main__':
    main()
