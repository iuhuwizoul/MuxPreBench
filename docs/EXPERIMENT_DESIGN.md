# Experiment design

## Question

How do background subtraction, retrospective illumination correction, and
sub-pixel cycle registration affect both image fidelity and biological signal
preservation in multiplexed fluorescence images?

## Scope

This prototype benchmarks three of the four preprocessing axes in the TUM
DI-LAB project description. Resolution enhancement by compressed sensing is
deliberately out of scope: a credible comparison would require measured point
spread functions, suitable high-resolution references, and specialized
methods. The package API leaves room for that extension without pretending
that generic sharpening is equivalent to Sparse-SIM.

## Data-generating process

Each synthetic field contains 20 cells with a binary cell type and a continuous
cell-stage variable. Four spatial protein patterns are rendered: DNA, membrane,
actin, and ERK-like signal. Every iterative acquisition cycle contains a repeated
DNA anchor plus one marker. Known artifacts are then applied:

1. cycle-level translation shared by anchor and marker;
2. smooth multiplicative illumination gradients and vignetting;
3. spatially varying autofluorescence-like background;
4. cycle-dependent bleaching, Poisson shot noise, and Gaussian read noise.

Biology is held fixed across artifact-severity conditions, so robustness
comparisons are paired rather than confounded by different fields of view.

## Methods and ablations

- Background: grayscale morphological opening.
- Illumination: retrospective Gaussian flat-field estimation.
- Registration: upsampled Fourier phase cross-correlation on the repeated DNA
  anchor, followed by the same correction on the paired marker.
- Full method plus single-component baselines and leave-one-component-out
  ablations.

These are intentionally classical, inspectable baselines. They are fast enough
to run on a laptop and establish the benchmark contract against which wavelet,
mixture-model, learned background, deformable registration, or sparse
deconvolution methods could later be compared.

## Technical endpoints

- SSIM and foreground Pearson correlation to clean ground truth;
- normalized foreground RMSE;
- residual background-to-signal ratio;
- correction-vector error in pixels;
- wall-clock runtime per field of view.

All aggregate endpoints use field-level bootstrap confidence intervals.

## Downstream endpoints

Splits are made by field of view, never by pixel or cell, to prevent spatial
leakage. Identical splits are reused across preprocessing methods.

1. **Held-out protein localization:** ERK is predicted pixel-wise from the
   remaining markers using local-context features and gradient boosting.
2. **Cell type classification:** marker summary features feed a class-balanced
   logistic regression; balanced accuracy and macro-F1 are reported.
3. **Cell-stage prediction:** the same features feed cross-validated ridge
   regression; R² and MAE are reported.

## Limitations and next experiment

Synthetic ground truth makes causal evaluation possible but does not establish
performance on real 4i data. The first real-data extension should keep the
pipeline and split contract, replace the generator with an OME-TIFF/AnnData
adapter, use repeated-anchor consistency and expert annotations as technical
references, and validate conclusions on a second biological context.
