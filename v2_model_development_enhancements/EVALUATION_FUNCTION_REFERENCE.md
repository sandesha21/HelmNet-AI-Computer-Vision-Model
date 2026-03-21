# Function Reference - Quick Lookup

## All 14 Functions

### 1. plot_roc_curves()
```python
plot_roc_curves(models_dict, X_test, y_test, figsize=(15, 5))
```
**Purpose**: Plot ROC curves for multiple models  
**Returns**: matplotlib figure  
**Time**: 1-2 seconds  
**Example**:
```python
fig = plot_roc_curves(models_dict, X_test, y_test)
plt.show()
```

### 2. plot_precision_recall_curves()
```python
plot_precision_recall_curves(models_dict, X_test, y_test, figsize=(15, 5))
```
**Purpose**: Plot Precision-Recall curves (better for imbalanced data)  
**Returns**: matplotlib figure  
**Time**: 1-2 seconds  
**Example**:
```python
fig = plot_precision_recall_curves(models_dict, X_test, y_test)
plt.show()
```

### 3. compute_feature_importance_gradients()
```python
compute_feature_importance_gradients(model, X_test, y_test, n_samples=100)
```
**Purpose**: Compute gradient-based feature importance  
**Returns**: numpy array of importance scores  
**Time**: 5-10 seconds  
**Example**:
```python
importance = compute_feature_importance_gradients(model, X_test, y_test)
```

### 4. plot_feature_importance()
```python
plot_feature_importance(importance_scores, figsize=(10, 6))
```
**Purpose**: Visualize feature importance scores  
**Returns**: matplotlib figure  
**Time**: <1 second  
**Example**:
```python
fig = plot_feature_importance(importance)
plt.show()
```

### 5. plot_activation_maps()
```python
plot_activation_maps(model, X_test, layer_name, n_samples=3, figsize=(15, 4))
```
**Purpose**: Visualize activation maps from intermediate layers  
**Returns**: matplotlib figure  
**Time**: 1-2 seconds  
**Example**:
```python
fig = plot_activation_maps(model, X_test, 'conv2d_1', n_samples=3)
plt.show()
```

### 6. perform_cross_validation()
```python
perform_cross_validation(model, X_train, y_train, cv_folds=5, metrics=['accuracy'])
```
**Purpose**: Perform k-fold cross-validation  
**Returns**: dict with cross-validation scores  
**Time**: 5-10 minutes  
**Example**:
```python
cv_results = perform_cross_validation(model, X_train, y_train, cv_folds=5)
```

### 7. plot_cross_validation_results()
```python
plot_cross_validation_results(cv_results_dict, figsize=(12, 5))
```
**Purpose**: Plot cross-validation results  
**Returns**: matplotlib figure  
**Time**: <1 second  
**Example**:
```python
fig = plot_cross_validation_results({'Model': cv_results})
plt.show()
```

### 8. print_cross_validation_summary()
```python
print_cross_validation_summary(cv_results_dict)
```
**Purpose**: Print cross-validation summary statistics  
**Returns**: None (prints to console)  
**Time**: <1 second  
**Example**:
```python
print_cross_validation_summary({'Model': cv_results})
```

### 9. get_prediction_examples()
```python
get_prediction_examples(model, X_test, y_test, n_correct=3, n_incorrect=3)
```
**Purpose**: Get examples of correct and incorrect predictions  
**Returns**: dict with examples  
**Time**: <1 second  
**Example**:
```python
examples = get_prediction_examples(model, X_test, y_test)
```

### 10. plot_prediction_examples()
```python
plot_prediction_examples(examples, figsize=(14, 6))
```
**Purpose**: Plot correct and incorrect prediction examples  
**Returns**: matplotlib figure  
**Time**: <1 second  
**Example**:
```python
fig = plot_prediction_examples(examples)
plt.show()
```

### 11. print_prediction_analysis()
```python
print_prediction_analysis(model, X_test, y_test, class_names=None)
```
**Purpose**: Print detailed prediction analysis  
**Returns**: None (prints to console)  
**Time**: 1-2 seconds  
**Example**:
```python
print_prediction_analysis(model, X_test, y_test)
```

### 12. plot_confusion_matrices()
```python
plot_confusion_matrices(models_dict, X_test, y_test, figsize=(15, 5))
```
**Purpose**: Plot confusion matrices for multiple models  
**Returns**: matplotlib figure  
**Time**: 1-2 seconds  
**Example**:
```python
fig = plot_confusion_matrices(models_dict, X_test, y_test)
plt.show()
```

### 13. plot_model_comparison()
```python
plot_model_comparison(models_dict, X_test, y_test, figsize=(12, 6))
```
**Purpose**: Compare multiple models with various metrics  
**Returns**: matplotlib figure  
**Time**: 2-3 seconds  
**Example**:
```python
fig = plot_model_comparison(models_dict, X_test, y_test)
plt.show()
```

### 14. generate_evaluation_report()
```python
generate_evaluation_report(models_dict, X_test, y_test, X_train=None, y_train=None, 
                          class_names=None, cv_folds=5)
```
**Purpose**: Generate comprehensive evaluation report  
**Returns**: None (prints to console)  
**Time**: 2-3 seconds  
**Example**:
```python
generate_evaluation_report(models_dict, X_test, y_test)
```

## Function Categories

### ROC & PR Curves (2 functions)
- `plot_roc_curves()` - ROC curves
- `plot_precision_recall_curves()` - PR curves

### Feature Analysis (3 functions)
- `compute_feature_importance_gradients()` - Compute importance
- `plot_feature_importance()` - Visualize importance
- `plot_activation_maps()` - Visualize activations

### Cross-Validation (3 functions)
- `perform_cross_validation()` - Run CV
- `plot_cross_validation_results()` - Visualize CV
- `print_cross_validation_summary()` - Print CV stats

### Prediction Analysis (3 functions)
- `get_prediction_examples()` - Get examples
- `plot_prediction_examples()` - Visualize examples
- `print_prediction_analysis()` - Print analysis

### Model Comparison (2 functions)
- `plot_confusion_matrices()` - Confusion matrices
- `plot_model_comparison()` - Compare metrics

### Comprehensive Report (1 function)
- `generate_evaluation_report()` - Full report

## Common Workflows

### Quick Model Comparison
```python
plot_roc_curves(models_dict, X_test, y_test)
plot_precision_recall_curves(models_dict, X_test, y_test)
plot_confusion_matrices(models_dict, X_test, y_test)
plot_model_comparison(models_dict, X_test, y_test)
```

### Detailed Analysis
```python
importance = compute_feature_importance_gradients(model, X_test, y_test)
plot_feature_importance(importance)
plot_activation_maps(model, X_test, 'conv2d_1')
```

### Error Analysis
```python
examples = get_prediction_examples(model, X_test, y_test)
plot_prediction_examples(examples)
print_prediction_analysis(model, X_test, y_test)
```

### Robustness Validation
```python
cv = perform_cross_validation(model, X_train, y_train)
plot_cross_validation_results({'Model': cv})
print_cross_validation_summary({'Model': cv})
```

### Full Evaluation
```python
generate_evaluation_report(models_dict, X_test, y_test)
```

## Parameter Guide

### models_dict
Dictionary of model names to models:
```python
models_dict = {
    'Model 1': model_1,
    'Model 2': model_2,
    'Model 3': model_3,
    'Model 4': model_4
}
```

### X_test, y_test
Test features and labels:
```python
X_test  # Shape: (n_samples, height, width, channels)
y_test  # Shape: (n_samples,)
```

### X_train, y_train
Training features and labels (for cross-validation):
```python
X_train  # Shape: (n_samples, height, width, channels)
y_train  # Shape: (n_samples,)
```

### layer_name
Name of layer to visualize (check with model.summary()):
```python
layer_name = 'conv2d_1'  # or 'conv2d_2', 'dense_1', etc.
```

### figsize
Figure size (width, height):
```python
figsize = (12, 6)  # Default varies by function
```

### n_samples
Number of samples to use:
```python
n_samples = 100  # For feature importance
n_samples = 3    # For activation maps
```

### cv_folds
Number of cross-validation folds:
```python
cv_folds = 5  # Default
```

### class_names
List of class names (optional):
```python
class_names = ['Class 0', 'Class 1']
```

## Return Values

### Figures
Functions that return figures:
```python
fig = plot_roc_curves(...)
fig = plot_precision_recall_curves(...)
fig = plot_feature_importance(...)
fig = plot_activation_maps(...)
fig = plot_cross_validation_results(...)
fig = plot_prediction_examples(...)
fig = plot_confusion_matrices(...)
fig = plot_model_comparison(...)
```

### Arrays
Functions that return arrays:
```python
importance = compute_feature_importance_gradients(...)
```

### Dicts
Functions that return dicts:
```python
cv_results = perform_cross_validation(...)
examples = get_prediction_examples(...)
```

### None
Functions that print to console:
```python
print_cross_validation_summary(...)
print_prediction_analysis(...)
generate_evaluation_report(...)
```

## Performance Summary

| Function | Time | Memory | Notes |
|----------|------|--------|-------|
| plot_roc_curves | 1-2s | Low | All models |
| plot_precision_recall_curves | 1-2s | Low | All models |
| compute_feature_importance_gradients | 5-10s | Medium | Depends on n_samples |
| plot_feature_importance | <1s | Low | Fast |
| plot_activation_maps | 1-2s | Medium | Single model |
| perform_cross_validation | 5-10m | High | Slow, 5 folds |
| plot_cross_validation_results | <1s | Low | Fast |
| print_cross_validation_summary | <1s | Low | Fast |
| get_prediction_examples | <1s | Low | Very fast |
| plot_prediction_examples | <1s | Low | Very fast |
| print_prediction_analysis | 1-2s | Low | Fast |
| plot_confusion_matrices | 1-2s | Low | All models |
| plot_model_comparison | 2-3s | Low | All models |
| generate_evaluation_report | 2-3s | Low | All models |

## Error Handling

### ImportError
```python
import sys
sys.path.append('./v2_model_development_enhancements')
from evaluation_visualization import *
```

### Layer not found
```python
model.summary()  # Check available layers
plot_activation_maps(model, X_test, 'correct_layer_name')
```

### Out of memory
```python
# Reduce samples
compute_feature_importance_gradients(model, X_test[:100], y_test[:100])
```

### Cross-validation too slow
```python
# Use fewer folds
perform_cross_validation(model, X_train, y_train, cv_folds=3)
```

## Tips & Tricks

1. **Save figures**:
   ```python
   fig.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
   ```

2. **Adjust figure size**:
   ```python
   plot_roc_curves(models_dict, X_test, y_test, figsize=(20, 5))
   ```

3. **Reduce computation time**:
   ```python
   # Use fewer samples
   importance = compute_feature_importance_gradients(model, X_test[:100], y_test[:100])
   ```

4. **Check layer names**:
   ```python
   model.summary()
   ```

5. **Batch processing**:
   ```python
   for model_name, model in models_dict.items():
       print(f"\nAnalyzing {model_name}...")
       print_prediction_analysis(model, X_test, y_test)
   ```
