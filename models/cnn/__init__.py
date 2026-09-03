"""CNN baseline: shared convolutional core + per-neuron factorised readout."""

from .config import DEFAULT_CFG, TUNED, get_cfg
from .model import RetinalModel
from .losses import poisson_loss, total_loss
from .train import build_model, train

__all__ = ['DEFAULT_CFG', 'TUNED', 'get_cfg', 'RetinalModel', 'poisson_loss',
           'total_loss', 'build_model', 'train']
