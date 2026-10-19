#!/usr/bin/env python
"""Pretrained vision-transformer cores, frozen or fine-tuned, in place of CiRF-ViT.

    python scripts/pretrained_cores.py --dataset mice --config dino_ft --fraction 0.2 --fold 3
    python scripts/pretrained_cores.py --dataset both --all          # the whole grid, resumable

Does a pretrained ViT, fine-tuned on the retina with little data, do what
CiRF-ViT does with its locality prior? Two kinds of
pretraining, each with the core frozen (only the readout is trained) or fully
fine-tuned:

    sup_*   ViT-S/16, supervised on ImageNet-21k then ImageNet-1k
            (timm `vit_small_patch16_224.augreg_in21k_ft_in1k`, 21.7M parameters)
    dino_*  ViT-S/16, self-supervised DINO on ImageNet-1k
            (timm `vit_small_patch16_224.dino`)
    v1t_*   the V1T core of Li et al. (2023), trained on mouse V1 responses to
            natural images (Sensorium 2022; weights huggingface.co/bryanlimy/V1T)

Everything else is the protocol of the ablations: the same K-fold partition and
training subsets as scripts/fraction_study_plain_vit.py (same `get_dataloaders`,
seed 0), Poisson NLL, Adam, early stopping on the held-out fold (patience and
maximum epochs of the dataset's config) with best-weight restore, and the same
factorised readout as CiRF-ViT (softmax spatial weights over the core's output
tokens, per-channel feature weights, ELU + 1), here over the whole token grid,
i.e. without a crop.

Input: the 108 x 108 stimulus, z-scored as stored. ImageNet ViTs get it resized
to 224 x 224 and repeated on three channels (196 tokens). V1T was trained on
36 x 64 images with 8 x 8 patches at stride 1; the stimulus is resized to
36 x 36 (the height of the V1T input) and the central 29 x 29 window of the
learned 29 x 57 position embedding is used (841 tokens); its behaviour MLP gets
a zero behaviour vector (no behaviour in the retina).

Learning rates (not tuned): readout 1e-3 as CiRF-ViT; core 1e-5 for the ImageNet
ViTs and 1e-4 for V1T when fine-tuned. bf16 autocast on the GPU.

Output: results/<dataset>/fraction_study/<config>/frac_XXX/fold_K/seed_0/
metrics.json (EV, adjusted R^2, CCnorm per neuron, test predictions,
parameters, epochs, time) and model.pt (trained weights only, fp16: readout,
plus the core when fine-tuned). A run whose metrics.json exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import ROOT, add_dataset_arg, cfg, datasets_from
from cirf import data as dat
from cirf.metrics import (adjusted_r2_per_neuron, cc_norm_per_neuron,
                          explained_variance_per_neuron)
from models.transformer import get_cfg
from models.transformer.losses import poisson_loss

WEIGHTS = ROOT / 'pretrained_weights'
os.environ.setdefault('HF_HOME', str(WEIGHTS / 'hf'))

TIMM = {'sup': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'dino': 'vit_small_patch16_224.dino'}
CORE_LR = {'sup': 1e-5, 'dino': 1e-5, 'v1t': 1e-4}
CONFIGS = [f'{c}_{m}' for c in ('sup', 'dino', 'v1t') for m in ('frozen', 'ft')]
FRACTIONS = (1.0, 0.5, 0.2, 0.1)


# -----------------------------------------------------------------------------
# cores: [B, 1, P, P] -> [B, T, D]
# -----------------------------------------------------------------------------

class TimmCore(nn.Module):
    def __init__(self, name, n_blocks=None):
        super().__init__()
        import timm
        self.vit = timm.create_model(name, pretrained=True, num_classes=0)
        if n_blocks:                       # keep only the first n blocks (then the final norm)
            self.vit.blocks = self.vit.blocks[:n_blocks]
        self.n_prefix = getattr(self.vit, 'num_prefix_tokens', 1)
        self.dim = self.vit.num_features
        self.n_tokens = (224 // 16) ** 2

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vit.forward_features(x.expand(-1, 3, -1, -1))
        return x[:, self.n_prefix:]


class V1TCore(nn.Module):
    SIZE = 36

    def __init__(self, run='v1t_sensorium'):
        super().__init__()
        import yaml
        from types import SimpleNamespace
        from models.pretrained.v1t_vit import Image2Patches, Transformer
        a = SimpleNamespace(**yaml.safe_load(open(WEIGHTS / 'v1t' / run / 'args.yaml')))
        c, h, w = a.input_shape[0], 36, 64            # images were downsampled 4x
        self.patch = Image2Patches((c, h, w), a.patch_mode, a.patch_size, a.patch_stride,
                                   a.emb_dim, a.p_dropout)
        self.transformer = Transformer(self.patch.output_shape, a.emb_dim, a.num_blocks,
                                       a.num_heads, a.mlp_dim, a.t_dropout, a.behavior_mode,
                                       list(a.output_shapes), a.use_lsa, a.drop_path,
                                       not a.disable_bias, False)
        sd = torch.load(WEIGHTS / 'v1t' / run / 'ckpt' / 'model_state.pt',
                        map_location='cpu', weights_only=False)['model']
        own = {k[len('core.patch_embedding.'):]: v for k, v in sd.items()
               if k.startswith('core.patch_embedding.')}
        self.patch.load_state_dict(own)
        own = {k[len('core.transformer.'):]: v for k, v in sd.items()
               if k.startswith('core.transformer.')}
        self.transformer.load_state_dict(own)
        self.gh = h - a.patch_size + 1                   # 29
        self.gw = w - a.patch_size + 1                   # 57
        self.g = self.SIZE - a.patch_size + 1            # 29 on the 36 x 36 input
        self.n_behaviour = 5 if a.behavior_mode in (3, 4) else 3
        self.dim = a.emb_dim
        self.n_tokens = self.g ** 2

    def forward(self, x):
        x = F.interpolate(x, size=(self.SIZE, self.SIZE), mode='bilinear', align_corners=False)
        p = self.patch.projection(x)                                    # [B, g*g, D]
        pos = self.patch.pos_embedding
        grid = pos[1:].reshape(self.gh, self.gw, -1)
        c0 = (self.gw - self.g) // 2
        grid = grid[:, c0:c0 + self.g].reshape(self.g * self.g, -1)
        cls = self.patch.cls_token.expand(x.shape[0], -1, -1) + pos[:1]
        z = self.patch.dropout(torch.cat([cls, p + grid], dim=1))
        beh = torch.zeros(x.shape[0], self.n_behaviour, device=x.device, dtype=z.dtype)
        z = self.transformer(z, mouse_id='S1', behaviors=beh)
        return z[:, 1:]


# -----------------------------------------------------------------------------
# model: core + the factorised readout of CiRF-ViT over all tokens
# -----------------------------------------------------------------------------

class PretrainedModel(nn.Module):
    def __init__(self, core, n_neurons):
        super().__init__()
        self.core = core
        self.u = nn.Parameter(torch.zeros(n_neurons, core.n_tokens))
        self.v = nn.Parameter(torch.randn(n_neurons, core.dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.frozen = False

    def freeze_core(self):
        self.frozen = True
        for p in self.core.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.core.eval()
        return self

    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            if self.frozen:
                with torch.no_grad():
                    z = self.core(x)
            else:
                z = self.core(x)
        z = z.float()                                                  # [B, T, D]
        spatial = torch.einsum('nt,btd->bnd', F.softmax(self.u, dim=-1), z)
        return F.elu((spatial * self.v).sum(-1) + self.bias) + 1.0


def build(config, n_neurons):
    """`<kind>[<n>]_<mode>`: kind sup / dino / v1t, n = blocks kept (ImageNet ViTs
    only; all 12 when absent), mode frozen / ft."""
    kind, mode = config.split('_')
    n_blocks = int(kind[len(kind.rstrip('0123456789')):] or 0) or None
    kind = kind.rstrip('0123456789')
    core = V1TCore() if kind == 'v1t' else TimmCore(TIMM[kind], n_blocks)
    model = PretrainedModel(core, n_neurons)
    if mode == 'frozen':
        model.freeze_core()
    return model, kind, mode


# -----------------------------------------------------------------------------
# training: the loop of models.transformer.train.fit, with two learning rates
# -----------------------------------------------------------------------------

def run(ds, config, fraction, fold, seed, device, n_folds):
    out = cfg.ensure(cfg.fraction_root(config, ds) / f'frac_{round(fraction * 100):03d}'
                     / f'fold_{fold}' / f'seed_{seed}')
    if (out / 'metrics.json').exists():
        print(f'[keep] {ds.key} {config} frac {fraction:.0%} fold {fold}', flush=True)
        return
    conf = get_cfg(ds)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, vl, _ = dat.get_dataloaders(ds, batch_size=conf['BATCH_SIZE'], data_fraction=fraction,
                                    random_seed=seed, fold=fold, n_folds=n_folds)
    model, kind, mode = build(config, dat.n_cells(ds))
    kind = kind.rstrip('0123456789')
    model.to(device)
    head = [model.u, model.v, model.bias]
    groups = [{'params': head, 'lr': conf['LEARNING_RATE']}]
    if mode == 'ft':
        groups.append({'params': list(model.core.parameters()), 'lr': CORE_LR[kind]})
    opt = torch.optim.Adam(groups, weight_decay=conf['WEIGHT_DECAY'])
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    best, patience, best_state, hist = float('inf'), 0, None, {'train_loss': [], 'val_loss': []}
    for epoch in range(1, int(conf['MAX_EPOCHS']) + 1):
        model.train()
        tl = 0.0
        for b in tr:
            x, y = b['image'].to(device), b['response'].to(device)
            opt.zero_grad(set_to_none=True)
            loss = poisson_loss(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for b in vl:
                vl_loss += poisson_loss(model(b['image'].to(device)), b['response'].to(device)).item()
        vl_loss /= len(vl)
        hist['train_loss'].append(tl / len(tr))
        hist['val_loss'].append(vl_loss)
        if vl_loss < best:
            best, patience = vl_loss, 0
            sd = model.state_dict()
            best_state = {k: sd[k].detach().to('cpu', copy=True) for k in trainable}
            hist['best_epoch'] = epoch
        else:
            patience += 1
            if patience >= conf['EARLY_STOP']:
                break
    model.load_state_dict(best_state, strict=False)
    model.eval()

    imgs = torch.tensor(dat.images('test', ds)[:, None], dtype=torch.float32)
    with torch.no_grad():
        y = torch.cat([model(imgs[i:i + 32].to(device)).float().cpu()
                       for i in range(0, len(imgs), 32)]).numpy()
    rep = dat.test_repeats(ds)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
               out / 'model.pt')
    (out / 'history.json').write_text(json.dumps(hist))
    m = {'dataset': ds.key, 'config': config, 'pretrained': kind, 'mode': mode,
         'fraction': fraction, 'fold': fold, 'seed': seed, 'n_train': len(tr.dataset),
         'epochs': epoch, 'best_epoch': hist.get('best_epoch'), 'val_loss': best,
         'n_params': n_par, 'n_trainable': n_train,
         'lr_readout': conf['LEARNING_RATE'], 'lr_core': CORE_LR[kind] if mode == 'ft' else None,
         'seconds': round(time.time() - t0, 1),
         'ev': explained_variance_per_neuron(y, rep).tolist(),
         'adj_r2': adjusted_r2_per_neuron(y, rep).tolist(),
         'cc_norm': cc_norm_per_neuron(y, rep).tolist(),
         'y_pred': y.tolist()}
    (out / 'metrics.json').write_text(json.dumps(m))
    print(f'{ds.key} {config} frac {fraction:.0%} fold {fold}: CCnorm {np.mean(m["cc_norm"]):.4f} '
          f'EV {np.mean(m["ev"]):.4f} ep {epoch} ({m["seconds"]:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(ap)
    ap.add_argument('--config')
    ap.add_argument('--fraction', type=float)
    ap.add_argument('--fold', type=int)
    ap.add_argument('--all', action='store_true', help='every config x fraction x fold')
    ap.add_argument('--configs', nargs='*', default=CONFIGS)
    ap.add_argument('--fractions', nargs='*', type=float, default=list(FRACTIONS))
    ap.add_argument('--seed', type=int, default=cfg.TRAIN_SEED)
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        if args.all:
            for frac in args.fractions:
                for config in args.configs:
                    for fold in range(args.n_folds):
                        run(ds, config, frac, fold, args.seed, args.device, args.n_folds)
        else:
            run(ds, args.config, args.fraction, args.fold, args.seed, args.device, args.n_folds)


if __name__ == '__main__':
    main()
