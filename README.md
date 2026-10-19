# CiRF-ViT — a vision transformer that is told where to look

Code for *Biologically grounded locality priors close the data gap for vision
transformers in neural prediction*.

A vision transformer normally needs far more data than a small
electrophysiology experiment provides: tens of neurons and a few thousand
stimulus–response pairs, two orders of magnitude below the scale at which these
architectures are usually trained. **CiRF-ViT** closes that gap with one
architectural change. Each neuron's receptive-field centre — already measured in
any standard recording — is handed to the model as a *differentiable circular
crop*: the neuron only ever attends to the tokens inside a circle around its own
receptive field, so the cost of attention scales with the receptive field rather
than with the image, and the locality prior comes from a measurement instead of
from data. One transformer is shared across the whole population; each neuron's
identity lives in its crop and in a small factorised readout.

This repository is **not only a reproduction package**. The models, the metrics
and the benchmarks work on any recording in the format of
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) — nothing is hard-wired to a neuron
count, an image size or a species. Reproducing the paper is one particular
invocation, described at the end.

The data and the trained checkpoints are **not** part of this repository (see
[Data](#data)).

---

## Install

```bash
conda env create -f environment.yml    # everything, GP baseline included
conda activate cirf
```

or, with an environment you already have:

```bash
pip install -r requirements.txt        # or: pip install -e .
```

PyTorch is the only heavy dependency. `gpytorch` is needed **only** for the
Gaussian-process baseline; everything else runs without it. A GPU is optional
but makes training minutes instead of hours.

## Quick start on your own recording

```bash
# 0. check the file: shapes, reliability, which benchmarks it supports
python scripts/check_dataset.py --data my_recording.npz

# 1. fit the model (and the baselines you want to compare against),
#    by 5-fold cross-validation: one checkpoint per fold
python scripts/train.py --model transformer --data my_recording.npz
python scripts/train.py --model cnn         --data my_recording.npz
python scripts/train.py --model LN          --data my_recording.npz

# 2. score them: explained variance, adjusted R², LSTA agreement, per fold
#    and averaged over the folds, plus the cross-validated Poisson NLL
python scripts/evaluate.py --data my_recording.npz

# 3. how much data does each architecture need?
python scripts/fraction_study.py --data my_recording.npz --seeds 0 1 2

# 4. figures
python scripts/make_extra_figures.py --data my_recording.npz
```

Everything produced lands under `results/<dataset>/`, keyed by the file's name.

Or from Python:

```python
from cirf import config, data, evaluate
from models.transformer import train, build_model, get_cfg

ds = config.register_path('my_recording.npz', label='my retina')
print(data.describe(ds))

for k in range(config.N_FOLDS):
    train(ds, fold=k)                          # fit CiRF-ViT, one fold at a time
evaluate.evaluate_model('transformer', ds=ds)  # score every fold and aggregate

model = build_model(get_cfg(ds), ds)          # or build one and train it yourself
```

If your recording has no tuned per-neuron radii (nobody's does, at first), the
model uses one radius for every neuron — set it with `--radius`, in stimulus
pixels, roughly the radius of a typical receptive field. That is the single
hyper-parameter worth trying two or three values of.

## What is in here

```
cirf/                the library: data, metrics, LSTA, benchmarks, plotting
  config.py          paths, dataset registry, palette
  data.py            dataset loading, crops, dataloaders, build_npz
  metrics.py         explained variance and adjusted R², per neuron
  lsta.py            model input gradients and the LSTA comparison
  lsta_filter.py     the band limit that comparison is restricted to
  checkpoints.py     rebuilding trained models from disk
  evaluate.py        eval_data.npz, the data-efficiency benchmark
  plots.py           figure styles and recurring panels

models/              one package per architecture, each with config/build/train
  transformer/       CiRF-ViT: tokeniser, circular crop, RoPE 2D, readout
    variants.py      the ablations and the ViT-P focal-bias comparison
    extra_variants.py  further ablations: convolutional core, crop centres,
                     radius, sampling
  pretrained/        the V1T core (Li et al., 2023), for the pretrained-core comparison
  cnn/               shared convolutional core + per-neuron readout
  gp/                sparse variational GP per neuron (Goldin et al., 2023)
  ln/                one linear–nonlinear cascade per neuron

scripts/             command-line entry points, all with --help
docs/DATA_FORMAT.md    the dataset schema
docs/PAPER_DATASETS.md what is in the paper's two recordings, and where from
```

Outputs are written under `results/` (override with `CIRF_RESULTS_DIR`), and
datasets are looked up in `data/` (override with `CIRF_DATA_DIR`):

```
results/<dataset>/checkpoints/<model>/fold_K/       model.pt / cell_XX.pt + config + history
results/<dataset>/checkpoints/<model>/fold_K/eval/  that fold's eval_data.npz
results/<dataset>/checkpoints/<model>/eval/         eval_data.npz aggregated over the folds
results/<dataset>/fraction_study/<model>/           frac_XXX/fold_K/seed_S/
results/<dataset>/experiments/                      ablation.json, history_nocrop.json
results/<dataset>/experiments/ablation/<variant>/fold_K/seed_S/   each ablation run's weights
results/<dataset>/experiments/crop_vs_nocrop/fold_K/              the no-crop run's weights
results/<dataset>/cache/                            derived arrays reused by the figures
results/figures/{paper,extra}/                      figures
```

## Cross-validation

Every model is fitted by classic **K-fold cross-validation** (K = 5,
`cirf.config.N_FOLDS`, or `--n-folds`). A recording carries one pool of fitting
stimuli and no train/validation split; the pool is partitioned once, from a
seeded permutation with a fixed seed, into K folds of equal size, and the same
partition is used by every model, every seed and every script. For fold *k* the
model is fitted on the other K−1 folds, with fold *k* held out: it is what early
stopping watches, and what the model's out-of-fold predictions are computed
on. Fold models are written to `checkpoints/<model>/fold_K/`.

`scripts/evaluate.py` scores each fold model on the repeated test stimuli
(explained variance, adjusted R², LSTA agreement — one `eval_data.npz` per
fold) and aggregates them into `checkpoints/<model>/eval/eval_data.npz`, which
is what every figure reads. There, each per-neuron metric is the **mean over
the folds** (`explained_variance`), every fold is kept alongside
(`explained_variance_folds`, shape `(K, N)`), `y_pred` is the mean prediction
of the K fold models (with `_ensemble` metrics of that prediction), and the
out-of-fold predictions of the K folds are stitched back into a prediction of
**every** pool stimulus by the one model that never saw it (`oof_pred`), from
which the cross-validated Poisson negative log-likelihood and single-trial
correlation per neuron (`cv_poisson_nll`, `cv_corr`) are computed. The
summary table prints the mean over folds with the standard deviation across
folds in parentheses.

The data-efficiency sweep and the ablation follow the same rule: a run is one
fold and one training seed (`--folds`, `--seeds`; by default every fold once),
the held-out fold does not depend on the fraction of data, and the training
folds are what gets subsampled. The sweep's run at 100 % of the data with the
canonical seed (`cirf.config.TRAIN_SEED`) is the same training as the fold's
canonical checkpoint, so whichever of `train.py` and `fraction_study.py` runs
first writes it to both locations and the other skips it: nothing is trained
twice, and both scripts resume where they stopped (`--force` retrains). The Python API mirrors this: every `train()`
takes `fold=` and `n_folds=`, `cirf.data.kfold_split` is the single place the
partition is defined, and `cirf.evaluate.evaluate_fold` / `aggregate_folds`
are the two halves of the evaluation.

## Models

| Model | What it assumes | Where the locality comes from |
|---|---|---|
| **CiRF-ViT** | one RF centre per neuron | a measured, per-neuron circular crop in feature space; centres are refined by training, radii are fixed |
| **CNN** | one RF centre per neuron | convolution (population-level), plus a distance-weighted penalty keeping each readout mask near its centre |
| **GP** | one RF centre per neuron | a learnable Gaussian envelope over pixels and a smoothness parameter, both priors of the kernel |
| **LN** | one RF centre per neuron | the model is fitted on a window around the centre; a single linear filter, so no context dependence by construction |

All four are trained with a Poisson negative log-likelihood, and all four are
given the same receptive-field information, so the comparison is about what each
architecture does with it.

## Benchmarks

**Prediction.** Explained variance and adjusted R², per neuron, against the
repeated test responses. Both normalise by the neuron's split-half reliability,
so 1.0 means roughly "as good as the neuron repeats itself", not "perfect". The
estimator (Goldin et al., PNAS 2023, Eq. 5) is not calibrated to exactly 1: an
oracle predicting the mean of the repeats scores 1.05 on the salamander and 1.08
on the mouse, and up to 1.4 on a single neuron. That is why the panels are drawn
to 1.12, and why a value slightly above 1 is not a red flag.

**Data efficiency.** The same models trained on 10 % … 100 % of the stimuli. This
is the claim that motivates the architecture, so it is a benchmark and not a
footnote: it shows where a transformer stops being viable and whether the crop
moves that point.

**LSTA — a zero-shot functional test.** The local spike-triggered average
measures a neuron's first-order sensitivity *around a particular image*, which is
exactly what the gradient of a model's predicted rate with respect to its input
is. No model is trained on it. It probes something explained variance cannot:
whether a model reproduces the context-dependent polarity inversion of the
retina, which a linear model cannot produce at all.

The comparison is band-limited first. Above some spatial frequency the measured
map is noise; correlating there cannot raise the covariance but does inflate the
model's variance in the denominator, penalising a model for having content where
there is nothing to agree with — and penalising the roughest model most. The
cutoff is derived from the experiment alone (`scripts/lsta_cutoff.py`), never
from the resulting correlations. Two further scripts quantify what the cut
removes (`scripts/band_power.py`) and whether the removed structure is
training-run noise or systematic (`scripts/gradient_reproducibility.py`).

## Scripts

| Script | Does |
|---|---|
| `check_dataset.py` | validate a dataset file and report what it supports |
| `train.py` | train one model or all four, K-fold (`--folds`, `--n-folds`) |
| `evaluate.py` | score every fold into `fold_K/eval/eval_data.npz` and aggregate |
| `fraction_study.py` | the data-efficiency sweep (the expensive one), one run per fold |
| `ablation.py` | remove one component at a time, one run per fold; also the ViT-P focal-bias variant and the two controls that separate the crop's effect on the attention from its effect on the readout (global attention with the circle in the readout only; the CNN with the circle on its readout); `--variants` also takes the further ablations of `extra_variants.py` |
| `ablation_posthoc.py` | LSTA agreement and trained crop centres of every fold of an ablation variant |
| `fraction_study_plain_vit.py` | the data-efficiency sweep of the plain ViT (CiRF-ViT without the crop) |
| `score_fractions.py` | test metrics of every data-efficiency run that has none yet |
| `pretrained_cores.py` | pretrained ViT cores (supervised ImageNet, DINO, V1T), frozen or fine-tuned, at every data fraction |
| `crop_vs_nocrop.py` | training dynamics with and without the crop, on one fold (`--fold`) |
| `lsta_cutoff.py` | derive the band limit from the experimental maps |
| `band_power.py` | how much gradient power the band limit removes, per model |
| `gradient_reproducibility.py` | is the fine structure systematic or run-to-run noise? |
| `make_paper_figures.py` | the paper's figures |
| `make_extra_figures.py` | supplementary figures, including per-neuron LSTA panels |
| `make_lsta_examples.py` | example LSTA maps for chosen neurons and stimuli |

Every script takes `--dataset` (a registered name) or `--data` (any `.npz`), and
`--help` explains what it does and what it needs.

## Data

The paper uses two multi-electrode-array datasets of retinal ganglion cells
collected by Goldin et al.
([2022](https://doi.org/10.1038/s41467-022-33242-8),
[2023](https://doi.org/10.1073/pnas.2301150120)): a mouse preparation of 41 cells
and two salamander preparations pooling 49 cells. They are downloadable from
Zenodo, DOI `10.5281/zenodo.22112881`. Place
`Mice_dataset.npz` and `Salamander_dataset.npz` in `data/`, or point
`CIRF_DATA_DIR` at wherever they live.

What is in those two files and what the numbers in them mean is in
[`docs/PAPER_DATASETS.md`](docs/PAPER_DATASETS.md).

Trained checkpoints are not distributed either — the pipeline below regenerates
them.

## Reproducing the paper

```bash
export CIRF_DATA_DIR=/path/to/datasets      # if not ./data

for DS in mice salamander; do
  python scripts/train.py           --model all --dataset $DS
  python scripts/evaluate.py                    --dataset $DS
  python scripts/fraction_study.py  --model all --dataset $DS
  python scripts/ablation.py                    --dataset $DS
  python scripts/crop_vs_nocrop.py              --dataset $DS
done

python scripts/make_paper_figures.py --dataset both
python scripts/lsta_cutoff.py        --dataset both      # appendix: the band limit
python scripts/band_power.py         --dataset both      # appendix: what it removes
```

Figure by figure:

| Paper figure | Script | Needs |
|---|---|---|
| 3a explained variance | `make_paper_figures.py --figures metrics_ev` | fraction study + LN |
| 3b per-neuron scatter | `--figures scatter_ev` | fraction study |
| 3c data efficiency | `--figures fraction_ev` | fraction study |
| 4c example LSTA maps | `make_lsta_examples.py` | evaluations with gradients |
| 4d LSTA correlation | `--figures metrics_lsta_corr` | evaluations with gradients |
| 5 ablation | `--figures ablation` | `ablation.py` + the CiRF-ViT and CNN folds |
| 6 focal-bias comparison | `--figures focal_comparison` | `ablation.py` + fraction study |
| 7 training dynamics | `--figures crop_vs_nocrop` | `crop_vs_nocrop.py` |
| Table 4 band power | `band_power.py` | fraction study at 100 % |
| Appendix A.2 cutoff | `lsta_cutoff.py` | dataset only |

Figures 1 and 2 are schematics of the stimulus protocol and of the
architectures, drawn by hand; the response statistics in figure 1c–d come from
`make_extra_figures.py --figures response_stats`. The manuscript's LSTA figure
is those two panels composed with species icons in a page layout — the panels
themselves are what the scripts produce.

Three things to know before comparing numbers:

* **Every run is shown.** The sweep trains one run per fold at every point
  (five with the default K) and the curves average all of them, with the band
  their spread. The canonical numbers are likewise the mean over the K fold
  models. No run or fold is selected, ranked or dropped anywhere in this
  repository.
* **The GP is stochastic in a way the others are not.** Its eigenbasis is rebuilt
  when a checkpoint is loaded, and the reconstruction is not guaranteed to be
  sign- and order-stable across PyTorch versions. Retraining reproduces the
  published means; re-loading very old checkpoints under a different PyTorch may
  not.
* **Exact bit-level reproduction is not expected** across GPUs, PyTorch versions
  or thread counts. The reported effects are much larger than that spread.

## License

MIT (see [`LICENSE`](LICENSE)). The recordings are not covered by it: they
belong to Goldin et al. and are cited below.

## Citation

```bibtex
@article{farina2026cirfvit,
  title  = {Biologically grounded locality priors close the data gap for vision
            transformers in neural prediction},
  author = {Farina, Matteo and Zamberlan, Pietro and Onken, Arno and
            Ferrari, Ulisse},
  year   = {2026},
  journal = {bioRxiv},
  doi    = {10.64898/2026.09.03.749141},
  note   = {Preprint}
}
```

The Gaussian-process baseline follows Goldin, Virgili and Chalk, *Scalable
Gaussian process inference of neural responses to natural images*, PNAS 2023,
and the datasets come from Goldin et al., *Context-dependent selectivity to
natural images in the retina*, Nature Communications 2022.
