# Alpha Selection via Nested Cross-Validation

## Overview

The probe training now supports **automatic alpha selection** via nested cross-validation to properly select the regularization strength without overfitting.

## How It Works

**Nested CV Architecture:**
```
Outer CV (already exists):
  - Selects best (layer, position) combination
  - 5-fold cross-validation on training data
  
Inner CV (NEW):
  - For each (layer, position), selects best alpha
  - Tests each alpha in grid via 5-fold CV
  - Chooses alpha with highest mean CV score
```

## Usage

### Option 1: Grid Search (RECOMMENDED)

Enable in `run.sh`:
```bash
ALPHA_GRID="0,0.001,0.01,0.1,1,10,100,1000"
```

Or via command line:
```bash
python3 -m scripts.train_probe \
    --train_activations data/train.pt \
    --test_activations data/test.pt \
    --output_dir results/ \
    --alpha_grid "0,0.001,0.01,0.1,1,10,100,1000" \
    --k_fold \
    --n_folds 5
```

### Option 2: Fixed Alpha

Disable grid search in `run.sh`:
```bash
ALPHA_GRID=""  # Empty = disabled
ALPHA=1.0      # Use this fixed value
```

Or via command line:
```bash
python3 -m scripts.train_probe \
    --train_activations data/train.pt \
    --test_activations data/test.pt \
    --output_dir results/ \
    --alpha 1.0 \
    --k_fold \
    --n_folds 5
```

## Recommended Alpha Grid

For **Ridge Regression** (continuous targets):
```
[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
```

For **Logistic Regression** (binary targets):
```
[0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Avoid 0 (no regularization can fail)
```

## Output

### performance.json
Now includes:
```json
{
  "alpha_grid": [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
  "selected_alphas": {
    "pos_-2_layer_0": 0.1,
    "pos_-2_layer_1": 1.0,
    ...
  },
  "alpha_cv_scores": {
    "pos_-2_layer_0": {
      "0": 0.45,
      "0.001": 0.52,
      "0.01": 0.58,
      "0.1": 0.62,  // <- best
      "1": 0.59,
      ...
    }
  }
}
```

### best_probe_predictions.json
Now includes:
```json
{
  "best_position": -2,
  "best_layer": 15,
  "selected_alpha": 0.1,
  "alpha_cv_scores": {
    "0": 0.45,
    "0.001": 0.52,
    ...
  },
  ...
}
```

## Interpretation

**Alpha selection shows:**
- Which regularization strength works best for each layer/position
- How sensitive performance is to alpha (via `alpha_cv_scores`)
- Whether you need strong regularization (high alpha) or weak (low alpha)

**Common patterns:**
- **Early layers**: Often benefit from stronger regularization (higher alpha)
- **Later layers**: May need less regularization if features are more refined
- **Alpha=0**: No regularization - only use if you have tons of data
- **Alpha>100**: Very strong regularization - suggests overfitting without it

## Computational Cost

**With grid search (8 alphas, 5-fold CV, 30 layers, 5 positions):**
- ~8× slower than fixed alpha
- But still fast! (sklearn is efficient for linear models)
- Extract activations once, experiment with many alpha grids

**Recommendation:**
- Use grid search for your final experiments
- Use fixed alpha for quick prototyping/debugging

## Why This Matters for Mechanistic Interpretability

From a mech interp perspective:
1. **Avoids arbitrary choices**: Alpha is selected by data, not researcher intuition
2. **Reproducible**: Others can verify your alpha selection via CV scores
3. **Transparent**: All alpha scores saved, not just the winner
4. **Principled**: Nested CV prevents leakage between selection and evaluation

This is the **proper way** to do model selection in science!
