# Evaluation & Visualization - Quick Start

## 30-Second Setup

```python
# 1. Copy this into your notebook after imports
from evaluation_visualization import *

# 2. Load your models
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

# 3. Run evaluations
plot_roc_curves(models_dict, X_test, y_test)
plot_precision_recall_curves(models_dict, X_test, y_test)
plot_confusion_matrices(models_dict, X_test, y_test)
plot_model_comparison(models_dict, X_test, y_test)
```

## One-Liner Examples

### ROC Curves
```python
plot_roc_curves({'Model': model}, X_test, y_test)
```

### Precision-Recall Curves
```python
plot_precision_recall_curves({'Model': model}, X_test, y_test)
```

### Confusion Matrix
```python
plot_confusion_matrices({'Model': model}, X_test, y_test)
```

### Feature Importance
```python
importance = compute_feature_importance_gradients(model, X_test, y_test)
plot_feature_importance(importance)
```

### Activation Maps
```python
plot_activation_maps(model, X_test, 'conv2d_1', n_samples=3)
```

### Cross-Validation
```python
cv = perform_cross_validation(model, X_train, y_train, cv_folds=5)
plot_cross_validation_results({'Model': cv})
```

### Prediction Examples
```python
examples = get_prediction_examples(model, X_test, y_test)
plot_prediction_examples(examples)
```

### Prediction Analysis
```python
print_prediction_analysis(model, X_test, y_test)
```

### Model Comparison
```python
plot_model_comparison(models_dict, X_test, y_test)
```

### Full Report
```python
generate_evaluation_report(models_dict, X_test, y_test)
```

## What to Look For

### ROC Curves
- ✓ Curve should be in upper-left corner
- ✓ AUC > 0.9 is excellent
- ✓ AUC > 0.8 is good
- ✗ AUC < 0.7 indicates poor model

### Precision-Recall Curves
- ✓ Curve should be in upper-right corner
- ✓ AP > 0.9 is excellent
- ✓ AP > 0.8 is good
- ✗ AP < 0.7 indicates poor model

### Confusion Matrices
- ✓ Diagonal should be high (correct predictions)
- ✗ Off-diagonal indicates misclassifications
- ✗ Look for patterns in errors

### Feature Importance
- ✓ Shows which features matter
- ✓ Can help with feature selection
- ✗ All features equally important = model may be underfitting

### Activation Maps
- ✓ Different patterns for different filters
- ✓ Early layers: edges, textures
- ✓ Later layers: complex patterns
- ✗ All blank = layer not learning

### Cross-Validation
- ✓ Low variance = stable model
- ✓ High mean = good performance
- ✗ High variance = overfitting
- ✗ Low mean = underfitting

### Prediction Examples
- ✓ Incorrect predictions should be hard cases
- ✗ Easy cases being misclassified = model issue
- ✓ Confidence high for correct, low for incorrect

### Model Comparison
- ✓ Select model with highest accuracy
- ✓ Consider precision/recall trade-off
- ✗ Large gaps between models = check for issues

## Common Issues & Fixes

### "No module named 'evaluation_visualization'"
```python
import sys
sys.path.append('./v2_model_development_enhancements')
from evaluation_visualization import *
```

### "Layer not found"
```python
# Check available layers:
model.summary()
# Use correct layer name
```

### "Out of memory"
```python
# Reduce samples:
compute_feature_importance_gradients(model, X_test[:100], y_test[:100])
```

### "Cross-validation too slow"
```python
# Use fewer folds:
perform_cross_validation(model, X_train, y_train, cv_folds=3)
```

## Integration Checklist

- [ ] Copy `evaluation_visualization.py` to notebook directory
- [ ] Import all functions
- [ ] Create `models_dict` with your models
- [ ] Run ROC and PR curves
- [ ] Run confusion matrices
- [ ] Run model comparison
- [ ] Run feature importance
- [ ] Run activation maps
- [ ] Run prediction examples
- [ ] Run full report

## Next Steps

1. **Identify best model** using model comparison
2. **Analyze errors** using prediction examples
3. **Understand model** using activation maps
4. **Validate robustness** using cross-validation
5. **Deploy best model** with confidence

## Files Included

- `evaluation_visualization.py` - Main module (copy to notebook)
- `EVALUATION_README.md` - Full documentation
- `EVALUATION_QUICK_START.md` - This file
- `integration_guide.md` - Integration instructions
