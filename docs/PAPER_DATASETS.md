# The paper's two datasets

The paper reports on two multi-electrode-array recordings of retinal ganglion
cells, both by Goldin et al.:

* **mouse**, 41 cells, one preparation.
* **salamander**, 49 cells, two preparations pooled (26 + 23).

Both were recorded for
[Goldin et al., Nature Communications 2022](https://doi.org/10.1038/s41467-022-33242-8);
the mouse recording was also used in
[Goldin et al., PNAS 2023](https://doi.org/10.1073/pnas.2301150120).

They are packaged for this code as `Mice_dataset.npz` and
`Salamander_dataset.npz`, in the schema of
[`DATA_FORMAT.md`](DATA_FORMAT.md), and are downloadable from Zenodo:

> **DOI: `10.5281/zenodo.22112881`**

Put both files in `data/`, or point `CIRF_DATA_DIR` at wherever they live, then

```bash
python scripts/check_dataset.py --dataset both
```

## What the numbers in them mean

Stimuli are natural images from the van Hateren database, normalised to a fixed
mean luminance and RMS contrast, presented for 300 ms each and separated by
300 ms of uniform grey. A response is the spike count in the 30–350 ms window
after image onset. Receptive-field centres come from a separate binary
white-noise block: a centre is the location of the peak absolute value of the
neuron's spike-triggered average. The RF outlines are in the same image pixels
as the centres.

Each file holds one pool of 3160 fitting stimuli and 30 held-out test stimuli,
the latter repeated 30 times for the mouse and 20 for the salamander. There is
no train/validation split in the file: the folds are cut in the code, from a
seeded permutation, so every model is fitted on the same stimuli. No model is
initialised from a spike-triggered average.

The experimental LSTA is stored at the resolution it was measured at, 72 px for
the mouse and 56 px for the salamander, together with the stimuli it was
measured around. `cirf.lsta` resamples it when it needs to, and
`cirf.lsta_filter` estimates the band limit at that native resolution so the
interpolation's own low-pass does not fold into the answer.

## Two details worth knowing

The salamander file excludes one neuron of its fifty, cluster 894 of
preparation 101: its split-half reliability is 0.44, against 0.63 for the next
lowest and 0.97 for the population median. Every metric normalises by that
reliability, so there was almost nothing there to normalise. It is not a
low-rate cell (2.99 Hz against a population median of 4.23 Hz), it is a noisy
one.

What the recordings called the neurons and the stimuli is not in the files; it
is in `cirf.data` (`CELL_INDICES`, `EXP_ID`, `STIMULUS_IDS`), so a neuron can
still be traced back to the preparation it came from.
