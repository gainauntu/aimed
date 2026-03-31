from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import build_model
from pair_dataset import scan_pair_samples, read_image_unicode, PairAugment, symmetric_fusion_chw, LightROIPreprocessor


def _embed_pair(sample, augment, image_size, swap: bool,
                light_preprocessor=None):
    paths = sample.image_paths
    if len(paths) >= 2:
        pa, pb = random.sample(paths, 2)
    else:
        pa = pb = paths[0]
    a = read_image_unicode(pa)
    b = read_image_unicode(pb)
    # Apply ROI preprocessing if available (Issue 6 fix)
    if light_preprocessor is not None:
        a = light_preprocessor.process(a)
        b = light_preprocessor.process(b)
    if swap:
        a, b = b, a
    a, b = augment(a, b)
    return torch.from_numpy(symmetric_fusion_chw(a, b, size=image_size))


def precompute_centroids(model, samples, class_names, device, image_size,
                         light_preprocessor=None):
    model.eval()
    augment = PairAugment(image_size, train=False)
    by_class = {name: [] for name in class_names}
    with torch.inference_mode():
        for s in samples:
            x = _embed_pair(s, augment, image_size, swap=False,
                            light_preprocessor=light_preprocessor).unsqueeze(0).to(device)
            _, emb = model(x)
            by_class[s.class_name].append(emb[0].detach().cpu().numpy().astype(np.float32))
    centroids = {}
    for cls in class_names:
        arr = np.stack(by_class[cls], axis=0)
        c = arr.mean(axis=0)
        c /= np.clip(np.linalg.norm(c), 1e-8, None)
        centroids[cls] = c
    return centroids


def choose_episode_classes(by_class, class_names, centroids, episode_idx, n_way=5):
    if episode_idx > 0 and episode_idx % 1000 == 0 and centroids is not None and len(class_names) >= n_way:
        seed_cls = random.choice(class_names)
        sims = []
        base = centroids[seed_cls]
        for cls in class_names:
            if cls == seed_cls:
                continue
            sims.append((float(np.dot(base, centroids[cls])), cls))
        sims.sort(reverse=True)
        chosen = [seed_cls] + [cls for _, cls in sims[: n_way - 1]]
        return chosen
    return random.sample(sorted(class_names), min(n_way, len(class_names)))


@torch.no_grad()
def build_library(model, samples, class_names, device, image_size: int,
                  light_preprocessor=None):
    model.eval()
    by_class = {name: [] for name in class_names}
    augment = PairAugment(image_size, train=False)
    for s in samples:
        tensors = [
            _embed_pair(s, augment, image_size, swap=False,
                        light_preprocessor=light_preprocessor),
            _embed_pair(s, augment, image_size, swap=True,
                        light_preprocessor=light_preprocessor),
        ]
        x = torch.stack(tensors).to(device)
        _, emb = model(x)
        mean_emb = emb.mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=0)
        by_class[s.class_name].append(mean_emb.detach().cpu().numpy().astype(np.float32))
    out = {}
    centroids = {}
    for class_name in class_names:
        arr = np.stack(by_class[class_name], axis=0)
        centroid = arr.mean(axis=0)
        centroid /= np.clip(np.linalg.norm(centroid), 1e-8, None)
        centroids[class_name] = centroid
        dists = np.linalg.norm(arr - centroid[None, :], axis=1)
        sims = arr @ centroid
        out[class_name] = {
            'centroid': centroid.astype(np.float32).tolist(),
            'p95_distance': float(np.percentile(dists, 95)),
            'p99_distance': float(np.percentile(dists, 99)),
            'min_similarity': float(np.percentile(sims, 1)),
            'second_class_margin': 0.0,
        }
    for class_name in class_names:
        base = centroids[class_name]
        others = [float(np.linalg.norm(base - centroids[o])) for o in class_names if o != class_name]
        out[class_name]['second_class_margin'] = float(min(others) * 0.20) if others else 0.10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--image-size', type=int, default=224,
                    help='Input resolution for prototype encoder. '
                         '224 is the safe default for T4 (288 OOMs). '
                         'Prototype embeddings are cosine-compared so '
                         'resolution has minimal effect on quality.')
    ap.add_argument('--episodes', type=int, default=50000)
    ap.add_argument('--episodes-per-epoch', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone-init', default='')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--n-way', type=int, default=3,
                    help='Number of classes per episode. '
                         'Reduced from 5 to 3 to halve episode batch size.')
    ap.add_argument('--n-support', type=int, default=3,
                    help='Support samples per class per episode.')
    ap.add_argument('--n-query', type=int, default=3,
                    help='Query samples per class per episode.')
    ns = ap.parse_args()

    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)

    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples, class_names = scan_pair_samples(ns.data_root)
    by_class = {}
    for s in samples:
        by_class.setdefault(s.class_name, []).append(s)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model('prototype', len(class_names), backbone_checkpoint=ns.backbone_init or None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ns.lr, weight_decay=1e-4, fused=torch.cuda.is_available())
    augment = PairAugment(ns.image_size, train=True)

    # Issue 6 fix: always preprocess to ROI crops using black reference
    light_preprocessor = LightROIPreprocessor(target_size=ns.image_size)
    print('LightROIPreprocessor active for prototype training (black reference)')

    n_way     = min(ns.n_way, len(class_names))
    n_support = ns.n_support
    n_query   = ns.n_query
    # Per-episode GPU tensors:
    #   support: n_way * n_support * 2 swaps  =  n_way * n_support * 2
    #   query:   n_way * n_query   * 2 swaps  =  n_way * n_query   * 2
    # At n_way=3, n_support=3, n_query=3, image_size=224:
    #   (3*3*2 + 3*3*2) = 36 tensors × [9,224,224] float32 ≈ 123 MB raw
    #   (vs original: 150 tensors × [9,288,288] ≈ 430 MB raw)
    # Enable mixed precision to halve that further.
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    total_epochs = max(1, ns.episodes // ns.episodes_per_epoch)
    history = []
    centroids = None
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(range(ns.episodes_per_epoch),
                    desc=f'[prototype-episodic] epoch {epoch+1}/{total_epochs}')
        for step in pbar:
            global_episode_idx = epoch * ns.episodes_per_epoch + step
            classes = choose_episode_classes(by_class, class_names, centroids,
                                             global_episode_idx, n_way=n_way)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            support, query, qys = [], [], []
            for cls in classes:
                items = random.choices(by_class[cls], k=n_support + n_query)
                support_samples = items[:n_support]
                query_samples   = items[n_support:]
                for s in support_samples:
                    support.append(_embed_pair(s, augment, ns.image_size, swap=False,
                                               light_preprocessor=light_preprocessor))
                    support.append(_embed_pair(s, augment, ns.image_size, swap=True,
                                               light_preprocessor=light_preprocessor))
                for s in query_samples:
                    query.append(_embed_pair(s, augment, ns.image_size, swap=False,
                                             light_preprocessor=light_preprocessor))
                    query.append(_embed_pair(s, augment, ns.image_size, swap=True,
                                             light_preprocessor=light_preprocessor))
                    qys.extend([class_to_idx[cls], class_to_idx[cls]])

            sx = torch.stack(support).to(device)
            qx = torch.stack(query).to(device)
            qy = torch.tensor(qys, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, semb = model(sx)
                _, qemb = model(qx)
                support_proto = []
                offset = 0
                per_cls = n_support * 2   # both swaps
                for _ in classes:
                    proto = semb[offset:offset + per_cls].mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                    support_proto.append(proto)
                    offset += per_cls
                protos = torch.stack(support_proto, dim=0)
                dists  = torch.cdist(F.normalize(qemb, dim=-1), protos)
                loss   = F.cross_entropy(-dists, qy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f'{epoch_loss / max(1, pbar.n):.4f}')

        history.append({'epoch': epoch + 1, 'loss': epoch_loss / ns.episodes_per_epoch})
        if (epoch + 1) % max(1, 1000 // ns.episodes_per_epoch) == 0:
            centroids = precompute_centroids(model, samples, class_names, device,
                                             ns.image_size,
                                             light_preprocessor=light_preprocessor)

    torch.save({'model_state': model.state_dict(), 'class_names': class_names, 'args': vars(ns), 'history': history}, outdir / 'prototype_best.ckpt')
    library = build_library(model, samples, class_names, device, ns.image_size,
                            light_preprocessor=light_preprocessor)
    (outdir / 'prototype_library.json').write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding='utf-8')
    (outdir / 'labels.json').write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding='utf-8')
    (outdir / 'metrics.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
