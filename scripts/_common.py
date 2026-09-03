"""Shared command-line plumbing for the scripts in this directory.

Importing it puts the repository root on `sys.path`, so the scripts run from a
clone without installing anything, and gives every script the same `--dataset`
/ `--data` / `--device` handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cirf import config as cfg           # noqa: E402
from cirf import data as dat             # noqa: E402


def add_dataset_arg(parser, default='mice', allow_both=True, help_extra=''):
    """`--dataset` (a registered name) and `--data` (any .npz)."""
    choices = 'mice|mouse|salamander' + ('|both' if allow_both else '')
    parser.add_argument('--dataset', default=default,
                        help=f'registered dataset: {choices} (default: {default}). '
                             f'{help_extra}')
    parser.add_argument('--data', default=None, metavar='NPZ',
                        help='path to your own dataset .npz (see '
                             'docs/DATA_FORMAT.md); overrides --dataset')
    return parser


def add_device_arg(parser, default=None):
    parser.add_argument('--device', default=default,
                        help='cuda or cpu (default: cuda when available)')
    return parser


def datasets_from(args):
    """The datasets a run should cover, as a list of `cfg.Dataset`."""
    if getattr(args, 'data', None):
        return [cfg.register_path(args.data)]
    return cfg.expand(args.dataset)


def device_from(args, default=None):
    import torch
    name = getattr(args, 'device', None) or default
    if name is None:
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def banner(text, char='='):
    print(f'\n{char * 72}\n{text}\n{char * 72}', flush=True)


def require_lsta(ds):
    """True when a dataset supports the LSTA benchmark, with a clear message."""
    if dat.meta(ds)['has_lsta'] and dat.meta(ds)['has_ellipses']:
        return True
    print(f'  [skip] {cfg.resolve(ds).label}: no experimental LSTA in the '
          f'dataset file, so the LSTA benchmark does not apply')
    return False
