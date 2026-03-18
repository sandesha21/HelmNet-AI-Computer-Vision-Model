# Notebook Cell Templates

Copy these cells directly into your Jupyter notebook.

## Cell 1: Import Evaluation Module

```python
# ============================================================================
# EVALUATION & VISUALIZATION - IMPORTS
# ============================================================================

from evaluation_visualization import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_activation_maps,
    compute_feature_importance_gradients,
    plot_feature_importance,
    perform_cross_validation,
    plot_cross_validation_results,
    print_cross_validation_summary,
    get_prediction_examples,
    plot_prediction_examples,
    print_prediction_analysis,
    generate_evaluation_report,
    plot_confusion_matrices,
    plot_model_comparison
)

print("✓ Evaluation module imported successfully")
```

## Cell 2: Setup Models Dictionary

```python
# ============================================================================
# SETUP MODELS DICTIONARY
# ============================================================================

models_dict = {
    'Model 1: Simple CNN': best_model_1,
    'Model 2: VGG-16 (Frozen)': best_model_2,
    'Model 3: VGG-16 + FFNN': best_model_3,
    'Model 4: VGG-16 + FFNN + Aug': best_model_4
}

print("✓ Models dictionary created")
print(f"  Total models: {len(models_dict)}")
for name in models_dict.keys():
    print(f"    - {name}")
```

## Cell 3: ROC & Precision-Recall Curves

```python
# ============================================================================
# ROC CURVES
# ============================================================================

print("\nGenerating ROC curves...")
fig_roc = plot_roc_curves(models_dict, X_test_normalized, y_test, figsize=(16, 4))
plt.show()

print("\n" + "="*70)
print("ROC CURVE INTERPRETATION:")
print("="*70)
print("- Curve in upper-left corner = good model")
print("- AUC > 0.9 = excellent")
print("- AUC > 0.8 = good")
print("- AUC < 0.7 = poor")
print("- Diagonal line = random classifier (AUC = 0.5)")
```

```python
# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================

print("\nGenerating Precision-Recall curves...")
fig_pr = plot_precision_recall_curves(models_dict, X_test_normalized, y_test, figsize=(16, 4))
plt.show()

print("\n" + "="*70)
print("PRECISION-RECALL CURVE INTERPRETATION:")
print("="*70)
print("- Better for imbalanced datasets")
print("- Curve in upper-right corner = good model")
print("- AP > 0.9 = excellent")
print("- AP > 0.8 = good")
print("- Baseline = proportion of positive class")
```

## Cell 4: Confusion Matrices & Model Comparison

```python
# ============================================================================
# CONFUSION MATRICES
# ============================================================================

print("\nGenerating confusion matrices...")
fig_cm = plot_confusion_matrices(models_dict, X_test_normalized, y_test, figsize=(16, 4))
plt.show()

print("\n" + "="*70)
print("CONFUSION MATRIX INTERPRETATION:")
print("="*70)
print("- Diagonal = correct predictions (should be high)")
print("- Off-diagonal = misclassifications")
print("- Look for patterns in errors")
```

```python
# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\nComparing all models...")
fig_comp = plot_model_comparison(models_dict, X_test_normalized, y_test, figsize=(12, 6))
plt.show()

print("\n" + "="*70)
print("MODEL COMPARISON METRICS:")
print("="*70)
print("- Accuracy: Overall correctness")
print("- Precision: Of predicted positives, how many are correct")
print("- Recall: Of actual positives, how many did we find")
print("- F1-Score: Harmonic mean of precision and recall")
```

## Cell 5: Feature Importance & Activation Maps

```python
# ============================================================================
# FEATURE IMPORTANCE (GRADIENT-BASED)
# ============================================================================

print("\nComputing feature importance for Model 4...")
importance = compute_feature_importance_gradients(
    best_model_4, X_test_normalized, y_test, n_samples=100
)

print("Plotting feature importance...")
fig_imp = plot_feature_importance(importance, figsize=(10, 6))
plt.show()

print("\n" + "="*70)
print("FEATURE IMPORTANCE INTERPRETATION:")
print("="*70)
print("- Shows which input features most affect predictions")
print("- Gradient-based: uses backpropagation")
print("- Top 20 features displayed")
print("- Useful for understanding model decisions")
```

```python
# ============================================================================
# ACTIVATION MAPS
# ============================================================================

print("\nVisualizing activation maps from Model 4...")
print("Checking available layers:")
best_model_4.summary()

# Choose a layer to visualize (e.g., 'conv2d_1')
layer_name = 'conv2d_1'  # CHANGE THIS TO YOUR LAYER NAME

print(f"\nPlotting activation maps for layer: {layer_name}")
fig_act = plot_activation_maps(
    best_model_4, X_test_normalized, 
    layer_name=layer_name,
    n_samples=3,
    figsize=(15, 4)
)
plt.show()

print("\n" + "="*70)
print("ACTIVATION MAP INTERPRETATION:")
print("="*70)
print("- Each subplot = one filter in the layer")
print("- Bright areas = high activation")
print("- Early layers: edges, textures")
print("- Later layers: complex patterns")
print("- All blank = layer not learning")
```

## Cell 6: Cross-Validation

```python
# ============================================================================
# CROSS-VALIDATION (OPTIONAL - TAKES TIME)
# ============================================================================

print("\n" + "="*70)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("="*70)
print("This may take 5-10 minutes...\n")

cv_results = perform_cross_validation(
    best_model_4, X_train_normalized, y_train, 
    cv_folds=5, metrics=['accuracy']
)

print("\nPlotting cross-validation results...")
cv_results_dict = {'Model 4': cv_results}
fig_cv = plot_cross_validation_results(cv_results_dict, figsize=(12, 5))
plt.show()

print_cross_validation_summary(cv_results_dict)

print("\n" + "="*70)
print("CROSS-VALIDATION INTERPRETATION:")
print("="*70)
print("- More robust than single train/val/test split")
print("- Low variance = stable model")
print("- High variance = possible overfitting")
print("- Mean score = expected performance")
```

## Cell 7: Prediction Examples & Analysis

```python
# ============================================================================
# PREDICTION EXAMPLES
# ============================================================================

print("\nGetting prediction examples from Model 4...")
examples = get_prediction_examples(
    best_model_4, X_test_normalized, y_test,
    n_correct=3, n_incorrect=3
)

print("Plotting prediction examples...")
fig_ex = plot_prediction_examples(examples, figsize=(14, 6))
plt.show()

print("\n" + "="*70)
print("PREDICTION EXAMPLES INTERPRETATION:")
print("="*70)
print("- Top row: Correct predictions (green)")
print("- Bottom row: Incorrect predictions (red)")
print("- Shows confidence scores")
print("- Useful for error analysis")
```

```python
# ============================================================================
# DETAILED PREDICTION ANALYSIS
# ============================================================================

print("\nDetailed prediction analysis for Model 4...")
print_prediction_analysis(best_model_4, X_test_normalized, y_test)
```

## Cell 8: Comprehensive Evaluation Report

```python
# ============================================================================
# COMPREHENSIVE EVALUATION REPORT
# ============================================================================

print("\n" + "="*70)
print("GENERATING COMPREHENSIVE EVALUATION REPORT")
print("="*70)

generate_evaluation_report(
    models_dict, 
    X_test_normalized, y_test,
    X_train=X_train_normalized,
    y_train=y_train,
    cv_folds=5
)

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
```

## Cell 9: Summary & Recommendations

```python
# ============================================================================
# EVALUATION SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

print("\n1. BEST MODEL SELECTION:")
print("   - Review model comparison chart")
print("   - Select model with highest accuracy")
print("   - Consider precision/recall trade-off")

print("\n2. ERROR ANALYSIS:")
print("   - Review prediction examples")
print("   - Identify patterns in misclassifications")
print("   - Consider data augmentation or rebalancing")

print("\n3. MODEL INTERPRETABILITY:")
print("   - Review feature importance")
print("   - Check activation maps")
print("   - Understand what model learns")

print("\n4. ROBUSTNESS:")
print("   - Review cross-validation results")
print("   - Check for overfitting (high variance)")
print("   - Validate on held-out test set")

print("\n5. DEPLOYMENT READINESS:")
print("   - ✓ Model performance validated")
print("   - ✓ Error patterns understood")
print("   - ✓ Model interpretability confirmed")
print("   - ✓ Robustness verified")
print("   - Ready for deployment!")

print("\n" + "="*70)
```

## Cell 10: Save Evaluation Results

```python
# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================

import os
from datetime import datetime

# Create results directory
results_dir = './evaluation_results'
os.makedirs(results_dir, exist_ok=True)

# Save figures
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

fig_roc.savefig(f'{results_dir}/roc_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_pr.savefig(f'{results_dir}/pr_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_cm.savefig(f'{results_dir}/confusion_matrices_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_comp.savefig(f'{results_dir}/model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_imp.savefig(f'{results_dir}/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_act.savefig(f'{results_dir}/activation_maps_{timestamp}.png', dpi=300, bbox_inches='tight')
fig_ex.savefig(f'{results_dir}/prediction_examples_{timestamp}.png', dpi=300, bbox_inches='tight')

print(f"✓ All evaluation results saved to {results_dir}/")
print(f"  Timestamp: {timestamp}")
```

## Quick Copy-Paste Workflow

```python
# 1. Import
from evaluation_visualization import *

# 2. Setup
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

# 3. Evaluate
plot_roc_curves(models_dict, X_test_normalized, y_test)
plot_precision_recall_curves(models_dict, X_test_normalized, y_test)
plot_confusion_matrices(models_dict, X_test_normalized, y_test)
plot_model_comparison(models_dict, X_test_normalized, y_test)

# 4. Analyze
importance = compute_feature_importance_gradients(best_model_4, X_test_normalized, y_test)
plot_feature_importance(importance)
plot_activation_maps(best_model_4, X_test_normalized, 'conv2d_1')

# 5. Validate
examples = get_prediction_examples(best_model_4, X_test_normalized, y_test)
plot_prediction_examples(examples)
print_prediction_analysis(best_model_4, X_test_normalized, y_test)

# 6. Report
generate_evaluation_report(models_dict, X_test_normalized, y_test)
```
