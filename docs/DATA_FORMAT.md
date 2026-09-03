# Dataset format

Everything the code needs about one recording lives in a single `.npz`. Nothing
else is read from disk: no per-cell files, no side-car receptive-field file.

Put your file anywhere and pass `--data path/to/file.npz` to any script, or drop
it in `data/` and refer to it by name. Check it first:

```bash
python scripts/check_dataset.py --data my_recording.npz
```

## Required arrays

| Array | Shape | Meaning |
|---|---|---|
| `images_pool` | `(n_pool, P, P, 1)` | every stimulus a model may be fitted on |
| `responses_pool` | `(n_pool, N)` | spike count per stimulus, per neuron |
| `images_test` | `(n_test, P, P, 1)` | held-out stimuli |
| `responses_test` | `(n_rep, n_test, N)` | **repeated** presentations of the test stimuli |
| `centers_x` | `(N,)` | receptive-field centre, column coordinate, in image pixels |
| `centers_y` | `(N,)` | receptive-field centre, row coordinate |

**There is no train/validation split in the file.** A recording carries one
pool, and every model is fitted by classic K-fold cross-validation over it: at
training time `cirf.data.kfold_split` cuts the pool, from a seeded permutation
with a fixed seed (`cirf.config.N_FOLDS`, `KFOLD_SEED`), into K folds of equal
size; fold k is held out (it is what early stopping watches and what the
out-of-fold score is computed on) while the model is fitted on the other K-1.
Two consequences worth knowing: every model sees exactly the same folds, and
two runs of the same fold that differ only in their training seed differ only
in initialisation, dropout and batch order — never in which images they saw.

`P` is the stimulus size in pixels (square; 108 in the paper), `N` the number of
neurons. Both are read from the file — no constant in the code fixes either.

Two properties of the test set matter:

* **repeats are mandatory** (`n_rep >= 2`). Every metric divides by the neuron's
  own split-half reliability, so a model is scored against what the neuron can
  reproduce rather than against a single noisy measurement. With one repeat
  there is no ceiling to normalise by.
* the responses are raw counts in whatever window you chose; only the ratio
  between neurons and stimuli matters to the models. `cirf.config.STIM_DURATION_S`
  is used solely to print firing rates in Hz.

Receptive-field centres are the one piece of prior information CiRF-ViT needs.
In the paper they come from a spike-triggered-average calibration block, but any
estimate works — a fitted Gaussian, a peak-response location, or the centre of
an LN filter. What the model needs is a starting point; the centre is a
parameter and training refines it.

## Optional arrays

| Array | Shape | Unlocks |
|---|---|---|
| `lsta` | `(N, n_img, S, S)` | the LSTA benchmark. `S` is the resolution the maps were measured at; it does **not** have to equal `P` — the code resamples |
| `lsta_base_images` | `(n_img, P, P)` | the stimuli the LSTA was measured around. **Required whenever `lsta` is present**: the maps are a property of the stimulus they were measured on, so it travels with them |
| `ellipses` | `(N, 2, n_points)` | receptive-field outlines, `[:, 0]` = x, `[:, 1]` = y. Required by the LSTA benchmark, which scores inside each outline |
| `ellipse_px` | scalar | only if the outlines are **not** in image pixels: the frame they are in, so the code can rescale them by `P / ellipse_px`. The paper's two files do not need it. `validate` checks that the outlines land on `centers_x`/`centers_y`, so a wrong frame is caught rather than silently applied |

No model is initialised from a spike-triggered average: every readout starts
flat, so what a model knows about a neuron's location comes from its receptive
field centre and from nothing else.

That is the whole schema. A file holds what was measured; names for the
neurons and the stimuli, and which preparation each neuron came from, are not
part of it — for the paper's two recordings they are in `cirf.data`
(`CELL_INDICES`, `EXP_ID`, `STIMULUS_IDS`). Anything extra in a file is
ignored rather than rejected.

## Building a file

```python
from cirf.data import build_npz

build_npz(
    'data/my_recording.npz',
    images_pool, responses_pool,         # (n, P, P) or (n, P, P, 1), (n, N)
    images_test, responses_test,         # (n_rep, n_test, N)
    centers_x, centers_y,                # (N,) each, in image pixels
    lsta=lsta, lsta_base_images=base,    # optional, together
    ellipses=ellipses,                   # optional
)
```

`build_npz` normalises the shapes, writes the file and validates it, raising a
specific error when something does not line up.

## What each analysis needs

| Analysis | Needs |
|---|---|
| training, prediction metrics | the required arrays |
| data-efficiency sweep | the required arrays |
| LSTA benchmark, LSTA figures | `lsta` + `lsta_base_images` + `ellipses` |
| gradient reproducibility, band power | the above + a fraction-study sweep at fraction 1.0 |

Scripts skip what a file cannot support, with a message, instead of failing.

## Preprocessing in the paper's datasets

For reference, the two recordings shipped with the paper were prepared like
this: natural images from the van Hateren database, normalised to a fixed mean
luminance and RMS contrast, presented for 300 ms each, separated by 300 ms of
uniform grey; the response is the spike count in a 30–350 ms window after image
onset. Receptive fields come from a separate binary white-noise block, and the
centre is the location of the peak absolute value of the spike-triggered
average — the only thing that average is used for. None of this is assumed by the code — it is what the numbers in
`Mice_dataset.npz` and `Salamander_dataset.npz` mean.
