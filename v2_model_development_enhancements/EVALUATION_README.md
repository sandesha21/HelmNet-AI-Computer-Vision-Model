# Evaluation & Visualization Module

Comprehensive evaluation and visualization tools for HelmNet models.

## Features

### 1. ROC Curves (`plot_roc_curves`)
- Plots ROC curves for multiple models
- Shows AUC scores
- Compares model discrimination ability
- Perfect for binary and multiclass classification

**Usage:**
```python
fig = plot_roc_curves(models_dict, X_test, y_test)
plt.show()
```

### 2. Precision-Recall Curves (`plot_precision_recall_curves`)
- Better for imbalanced datasets
- Shows precision vs recall trade-off
- Displays Average Precision (AP) scores
- Includes baseline for comparison

**Usage:**
```python
fig = plot_precision_recall_curves(models_dict, X_test, y_test)
plt.show()
```

### 3. Confusion Matrices (`plot_confusion_matrices`)
- Visualizes prediction distribution
- Shows True Positives, False Positives, etc.
- Heatmap format for easy interpretation
- Helps identify confused classes

**Usage:**
```python
fig = plot_confusion_matrices(models_dict, X_test, y_test)
plt.show()
```

### 4. Feature Importance (`compute_feature_importance_gradients`, `plot_feature_importance`)
- Gradient-based importance computation
- Shows which input features matter most
- Top 20 features displayed
- Useful for model interpretability

**Usage:**
```python
importance = compute_feature_importance_gradients(model, X_test, y_test)
fig = plot_feature_importance(importance)
plt.show()
```

### 5. Activation Maps (`plot_activation_maps`)
- Visualizes intermediate layer activations
- Shows what filters learn
- Helps understand model internals
- Useful for debugging

**Usage:**
```python
fig = plot_activation_maps(model, X_test, layer_name='conv2d_1', n_samples=3)
plt.show()
```

### 6. Cross-Validation (`perform_cross_validation`, `plot_cross_validation_results`)
- K-fold cross-validation
- More robust than single split
- Shows variance in performance
- Detects overfitting

**Usage:**
```python
cv_results = perform_cross_validation(model, X_train, y_train, cv_folds=5)
fig = plot_cross_validation_results({'Model': cv_results})
plt.show()
```

### 7. Prediction Examples (`get_prediction_examples`, `plot_prediction_examples`)
- Shows correct and incorrect predictions
- Displays confidence scores
- Side-by-side comparison
- Useful for error analysis

**Usage:**
```python
examples = get_prediction_examples(model, X_test, y_test, n_correct=3, n_incorrect=3)
fig = plot_prediction_examples(examples)
plt.show()
```

### 8. Prediction Analysis (`print_prediction_analysis`)
- Detailed prediction statistics
- Per-class metrics
- Confidence analysis
- Classification report

**Usage:**
```python
print_prediction_analysis(model, X_test, y_test)
```

### 9. Model Comparison (`plot_model_comparison`)
- Compares multiple models
- Shows Accuracy, Precision, Recall, F1-Score
- Bar chart visualization
- Easy model selection

**Usage:**
```python
fig = plot_model_comparison(models_dict, X_test, y_test)
plt.show()
```

### 10. Comprehensive Report (`generate_evaluation_report`)
- Full evaluation for all models
- Test set performance
- Confusion matrices
- Per-class metrics
- ROC-AUC scores

**Usage:**
```python
generate_evaluation_report(models_dict, X_test, y_test)
```

## Complete Example

```python
from evaluation_visualization import *

# Load models
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

# 1. ROC and PR curves
plot_roc_curves(models_dict, X_test, y_test)
plot_precision_recall_curves(models_dict, X_test, y_test)

# 2. Confusion matrices and comparison
plot_confusion_matrices(models_dict, X_test, y_test)
plot_model_comparison(models_dict, X_test, y_test)

# 3. Feature importance and activation maps
importance = compute_feature_importance_gradients(best_model_4, X_test, y_test)
plot_feature_importance(importance)
plot_activation_maps(best_model_4, X_test, 'conv2d_1')

# 4. Cross-validation
cv_results = perform_cross_validation(best_model_4, X_train, y_train)
plot_cross_validation_results({'Model 4': cv_results})

# 5. Prediction examples
examples = get_prediction_examples(best_model_4, X_test, y_test)
plot_prediction_examples(examples)
print_prediction_analysis(best_model_4, X_test, y_test)

# 6. Full report
generate_evaluation_report(models_dict, X_test, y_test)
```

## Function Reference

### ROC Curves
```python
plot_roc_curves(models_dict, X_test, y_test, figsize=(15, 5))
```
- **models_dict**: Dict of {name: model}
- **X_test**: Test features
- **y_test**: Test labels
- **figsize**: Figure size (default: (15, 5))
- **Returns**: matplotlib figure

### Precision-Recall Curves
```python
plot_precision_recall_curves(models_dict, X_test, y_test, figsize=(15, 5))
```
- **models_dict**: Dict of {name: model}
- **X_test**: Test features
- **y_test**: Test labels
- **figsize**: Figure size (default: (15, 5))
- **Returns**: matplotlib figure

### Feature Importance
```python
importance = compute_feature_importance_gradients(model, X_test, y_test, n_samples=100)
fig = plot_feature_importance(importance, figsize=(10, 6))
```
- **model**: Trained model
- **X_test**: Test features
- **y_test**: Test labels
- **n_samples**: Number of samples to use (default: 100)
- **Returns**: importance array and figure

### Activation Maps
```python
fig = plot_activation_maps(model, X_test, layer_name, n_samples=3, figsize=(15, 4))
```
- **model**: Trained model
- **X_test**: Test features
- **layer_name**: Name of layer to visualize
- **n_samples**: Number of samples (default: 3)
- **figsize**: Figure size (default: (15, 4))
- **Returns**: matplotlib figure

### Cross-Validation
```python
cv_results = perform_cross_validation(model, X_train, y_train, cv_folds=5, metrics=['accuracy'])
fig = plot_cross_validation_results(cv_results_dict, figsize=(12, 5))
```
- **model**: Compiled model
- **X_train**: Training features
- **y_train**: Training labels
- **cv_folds**: Number of folds (default: 5)
- **metrics**: List of metrics (default: ['accuracy'])
- **Returns**: results dict and figure

### Prediction Examples
```python
examples = get_prediction_examples(model, X_test, y_test, n_correct=3, n_incorrect=3)
fig = plot_prediction_examples(examples, figsize=(14, 6))
```
- **model**: Trained model
- **X_test**: Test features
- **y_test**: Test labels
- **n_correct**: Number of correct examples (default: 3)
- **n_incorrect**: Number of incorrect examples (default: 3)
- **Returns**: examples dict and figure

### Prediction Analysis
```python
print_prediction_analysis(model, X_test, y_test, class_names=None)
```
- **model**: Trained model
- **X_test**: Test features
- **y_test**: Test labels
- **class_names**: List of class names (optional)
- **Returns**: Prints analysis to console

### Model Comparison
```python
fig = plot_model_comparison(models_dict, X_test, y_test, figsize=(12, 6))
```
- **models_dict**: Dict of {name: model}
- **X_test**: Test features
- **y_test**: Test labels
- **figsize**: Figure size (default: (12, 6))
- **Returns**: matplotlib figure

### Confusion Matrices
```python
fig = plot_confusion_matrices(models_dict, X_test, y_test, figsize=(15, 5))
```
- **models_dict**: Dict of {name: model}
- **X_test**: Test features
- **y_test**: Test labels
- **figsize**: Figure size (default: (15, 5))
- **Returns**: matplotlib figure

### Comprehensive Report
```python
generate_evaluation_report(models_dict, X_test, y_test, X_train=None, y_train=None, 
                          class_names=None, cv_folds=5)
```
- **models_dict**: Dict of {name: model}
- **X_test**: Test features
- **y_test**: Test labels
- **X_train**: Training features (optional, for CV)
- **y_train**: Training labels (optional, for CV)
- **class_names**: List of class names (optional)
- **cv_folds**: Number of CV folds (default: 5)
- **Returns**: Prints report to console

## Tips & Best Practices

1. **ROC vs PR Curves**: Use PR curves for imbalanced datasets, ROC for balanced
2. **Feature Importance**: Useful for understanding model decisions
3. **Activation Maps**: Check layer names with `model.summary()`
4. **Cross-Validation**: Takes time but more robust than single split
5. **Prediction Examples**: Great for error analysis and debugging
6. **Model Comparison**: Use to select best model for deployment

## Performance Notes

- **ROC/PR Curves**: Fast, ~1-2 seconds per model
- **Feature Importance**: Depends on n_samples, ~5-10 seconds
- **Activation Maps**: Fast, ~1-2 seconds
- **Cross-Validation**: Slow, ~5-10 minutes for 5 folds
- **Prediction Examples**: Fast, <1 second
- **Model Comparison**: Fast, ~2-3 seconds

## Requirements

- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow/keras

All included in standard ML environments.
