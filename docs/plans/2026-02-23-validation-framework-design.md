# Embedding Validation Framework Design

**Date:** 2026-02-23
**Status:** Approved

## Problem

The current model comparison (`06_compare_models.py`) evaluates embedding models on only 56 hand-labeled disambiguation entries. It never validates whether models can recover known tags and edges. Hyperparameters (calibration percentile, specificity, margin, must-bridge) were chosen by intuition, not empirically.

Additionally, the current entity representation (average or k-means centroids) collapses the shape of an entity's semantic space into 1-3 points, losing information about variance and multi-modal structure.

## Goals

1. Build a rigorous validation framework using the 7,813 explicitly-tagged entries as ground truth
2. Compare 4 entity representation methods (k-means baseline, GMM diagonal, GMM full covariance, KDE)
3. Compare across 4 embedding models (Azure OpenAI, Cohere, Gemini, Voyage)
4. Find empirically optimal hyperparameters for the frontend sliders
5. Measure both tag-level and edge-level recovery

## Design

### Validation Structure: Stratified 5-Fold Cross-Validation

Split the 7,813 tagged entries into 5 stratified folds (scikit-learn `StratifiedKFold` adapted for multi-label). Each fold rotation:

- **3 folds**: Training (build entity representations)
- **1 fold**: Validation (tune hyperparameters)
- **1 fold**: Test (final held-out evaluation, touched once)

Entity inclusion by entry count:
- **3+ entries**: Full train/test evaluation (stratified k-fold)
- **2 entries**: Leave-one-out evaluation (train on 1, test on 1)
- **1 entry**: False-positive-only evaluation (can the model avoid tagging unrelated entries?)

### Representation Methods

**Method 1: K-means prototypes (baseline)**
- Current system: average tagged entries into 1-3 centroids via k-means
- Score: max cosine similarity across centroids

**Method 2: GMM with diagonal covariance**
- Fit `sklearn.mixture.GaussianMixture` with diagonal covariance
- Number of components: selected by BIC (1-5 candidates)
- Score: log-likelihood under the mixture
- Compact enough for static `scores.json`

**Method 3: GMM with full covariance (PCA-reduced)**
- PCA-reduce all embeddings to ~100 dimensions first
- Fit GMM with full covariance matrices on reduced embeddings
- Score: log-likelihood
- Requires larger `scores.json` or API backend

**Method 4: KDE (gold standard)**
- No model fitting; store all tagged entry embeddings per entity
- Score: kernel density estimation with Gaussian kernel
- Bandwidth: tuned per entity via cross-validation
- Requires API backend for query-time scoring

### Embedding Models

All 4 models with cached embeddings:
- Azure OpenAI `text-embedding-3-large` (3072-dim)
- Azure Cohere `embed-v4` (1024-dim)
- Google Gemini `embedding-001` (3072-dim)
- Voyage `voyage-4` (1024-dim)

### Metrics

**Tag-level (per entity, aggregated across folds):**
- Precision, Recall, F1
- Mean Average Precision (MAP) -- ranking quality
- Optimal threshold per entity

**Edge-level (graph-level):**
- Edge recall @ weight >= 2
- Edge precision @ weight >= 2
- Edge F1
- Weight correlation (Spearman rank)
- Rescued nodes
- Novel edge count (reported separately, not in precision calc)

### Hyperparameter Tuning

Grid search on validation fold:

| Parameter | Search range |
|-----------|-------------|
| Calibration percentile | 10, 15, 20, 25, 30, 35, 40, 45, 50 |
| Min specificity | 0, 1, 2, 3, 4, 5, 6, 7, 8 |
| Confidence margin | 0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20 |
| Must-bridge | true, false |
| Max GMM components | 1, 2, 3, 4, 5 |
| PCA dimensions (Method 3) | 50, 100, 200 |

Optimize for F1 (primary) and MAP (secondary).

## Implementation

### New file: `notebooks/08_validate_models.py`

```
Inputs:
  - data/embeddings_cache/*.npy (4 models)
  - data/embeddings_cache/entry_ids.json
  - data/tag_classifications.json
  - ../../words-of-brandon/wob_entries.json

Outputs:
  - data/validation_results.json (full results)
  - data/validation_summary.txt (comparison table)
```

CLI:
```
python 08_validate_models.py
python 08_validate_models.py --models azure_openai,voyage
python 08_validate_models.py --methods kmeans,gmm_diag
python 08_validate_models.py --folds 3
```

### Dependencies

No new dependencies. Uses scikit-learn (already installed):
- `sklearn.mixture.GaussianMixture`
- `sklearn.neighbors.KernelDensity`
- `sklearn.decomposition.PCA`
- `sklearn.model_selection.StratifiedKFold`

### Performance

Estimated ~10 minutes on M4 Pro CPU for the full 4x4 matrix (4 models x 4 methods x 5 folds x hyperparameter grid). GPU acceleration via MLX is a future optimization if needed.

### Changes to Existing Code

None. This is a standalone evaluation script. If a new method wins, `07_export_scores.py` would be updated in a separate step.

## Expected Output

A comparison table:

```
Model + Method          | Tag F1 | Tag MAP | Edge F1 | Edge Recall | Best Calibration | Best Specificity | Best Margin
Azure OpenAI + k-means  |  0.XX  |  0.XX   |  0.XX   |  0.XX       | p30              | 1.5              | 0.05
Azure OpenAI + GMM-diag |  0.XX  |  0.XX   |  0.XX   |  0.XX       | p20              | 2.0              | 0.03
Voyage + GMM-full       |  0.XX  |  0.XX   |  0.XX   |  0.XX       | p15              | 1.0              | 0.08
...
```

## Decision Points After Validation

1. If GMM-diagonal beats k-means: update `07_export_scores.py` to use GMM. Same deployment model.
2. If GMM-full or KDE beats GMM-diagonal significantly: evaluate whether the accuracy gain justifies adding an API backend.
3. If Voyage + better representation beats Azure OpenAI + k-means: switch default model.
4. Update frontend slider defaults to empirically optimal values.
