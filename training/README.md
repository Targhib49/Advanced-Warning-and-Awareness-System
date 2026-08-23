# T1 — eye-state model training

The learned replacement for the EAR threshold (plan v3 §3, tier T1).

Everything here runs on a laptop. This is not a cloud-scale problem: ~24k
parameters over 32×32 grayscale crops, ~85k images. The bottleneck in the
schedule is **dataset access**, not compute.

## Run it now, before the data arrives

```bash
python -m training.train_eyestate --smoke
python -m training.export
```

`--smoke` trains on synthetic eyes. It is not a benchmark — it exists so the
whole pipeline (training, checkpointing, metrics, ONNX export, int8
quantization, latency bench) is proven working in week 1 while the real
download is still in progress.

## The real run

1. Get the **MRL Eye Dataset** — ~85k infrared eye crops, 37 subjects,
   annotated in the filename. It is openly available and mirrored on Kaggle,
   so check there before downloading it twice. Unlike NTHU-DDD, UTA-RLDD and
   DROZY it needs no application, which is exactly why Phase 1 uses it.
2. Unpack to `./data/mrlEyes`, or point `AWAS_MRL_DIR` at it.
3. **Verify the filename convention** against what actually landed on disk —
   `mrl.py` documents what it expects. The parser raises rather than guessing,
   because a silent label flip would corrupt every number downstream and would
   not surface as an error anywhere.

```bash
python -m training.train_eyestate
python -m training.export --calib-dir ./data/mrlEyes
```

## Running on Kaggle or Colab unchanged

Paths come from the environment, so the same script runs in both places:

```bash
AWAS_MRL_DIR=/kaggle/input/mrl-eye-dataset AWAS_OUT_DIR=/kaggle/working \
  python -m training.train_eyestate
```

Use a notebook as the **reproducibility deliverable** for the competition, not
as the compute environment — judges can execute it, which is worth real marks.
Watch the session limits: a multi-hour job that dies partway with no
checkpointing costs more than it saved.

## Two things the code enforces

**Subject-wise splitting.** MRL has ~37 people with hundreds of near-identical
frames each. A random per-image split puts the same eyes in train and test and
reports an accuracy that has nothing to do with performance on a driver the
model has never met — which is the only number that matters for a device that
meets a new driver every time it is installed. `split_by_subject` is the only
split available here, and a test asserts no subject crosses a boundary.

**Baselines run first.** Majority class and logistic regression on raw pixels
are evaluated before the CNN, and the script prints a verdict. If the CNN does
not clear logistic regression by half a point, ship the baseline — deep
learning that does not beat something simpler is a liability on an embedded
target, not an achievement.

There is deliberately **no EAR baseline here**: EAR needs facial landmarks and
MRL ships pre-cropped eyes with no face around them. That comparison belongs
downstream on full-face video, and saying so beats fabricating a comparison
this dataset cannot support.

## Results — full MRL run

84,898 images, 37 subjects, split **by subject**: 24 train / 6 val / 7 test.
Trained 30 epochs on an M1 Pro (MPS), ~4.7 s per epoch.

| | test accuracy |
|---|---|
| **EyeStateNet (23.5k params)** | **0.9859** |
| logistic regression on pixels | 0.8682 |
| majority class | 0.5469 |

The CNN clears the logistic-regression baseline by 11.8 points, so it earns its
place on the device. That is the thing to report — not the headline accuracy,
which a leaky per-image split would have inflated past 0.99 and made meaningless.

**By slice**, which is where the useful information is:

| slice | n | accuracy |
|---|---|---|
| no glasses | 10,925 | 0.9902 |
| **glasses** | **1,321** | **0.9500** |
| bad light | 8,511 | 0.9855 |
| good light | 3,735 | 0.9866 |
| RealSense | 10,508 | 0.9872 |
| IDS | 1,738 | 0.9776 |

Two things worth carrying into the writeup:

- **Glasses cost 4 points.** This is the documented weakness of eye-state
  detection and now it is quantified rather than asserted. It is also an
  argument for the vehicle channel in plan v3 §4.3 — something that does not
  care about lenses.
- **Bad light costs nothing** (98.55 vs 98.66). Encouraging for night
  operation, which is the half of the use case that matters most. Note the
  dataset is infrared throughout, so this measures IR robustness, not
  performance in a dark cabin with a visible-light camera.

Caveat: the Aptina sensor has only 2,764 images across the whole dataset and
none of its subjects landed in the test split, so it is unmeasured here.

## Export

`export.py` produces fp32 ONNX and a statically quantized int8 ONNX, verifying
each stage against the one before it. Static rather than dynamic quantization
because this is a convolutional model — dynamic only helps MatMul/Gemm and
would leave the convolutions, which are all of the compute, untouched.

Calibration data must resemble what the model sees in a cabin. Pass
`--calib-dir` with real crops once you have them; the synthetic fallback exists
only to keep the pipeline runnable.

Measured on an M1 Pro, single-threaded, batch 2 (both eyes at once):

| | size | p50 | test accuracy |
|---|---|---|---|
| fp32 | 24.8 KB | 0.185 ms | 0.9859 |
| int8 | 49.1 KB | 0.108 ms | 0.9868 |

Quantization costs no accuracy at all here — the +0.0009 is noise, not a gain.
`export.py` measures this on the held-out subjects, not on random inputs: an
argmax-agreement check against noise proves the graph translated faithfully and
says nothing about whether real eyes still classify correctly.

**Read that size column again: int8 is bigger than fp32.** At 23.5k parameters
the QDQ format's quantize/dequantize nodes cost more than the weight savings
return. Quantization buys about 0.08 ms and costs a calibration dependency plus
an accuracy risk — so unless the Pi measurement says otherwise, **ship fp32**.
It is 25 KB and it runs in well under a millisecond.

Keep the int8 path anyway: it is nearly free to maintain, and it becomes worth
using the moment the model grows (the T2 temporal net, or a wider T1).

A Pi 5 is several times slower, so treat these as a floor to measure against —
not a prediction. Re-run `python -m training.export --bench 500` on the Pi
itself before writing any number into the latency budget.
