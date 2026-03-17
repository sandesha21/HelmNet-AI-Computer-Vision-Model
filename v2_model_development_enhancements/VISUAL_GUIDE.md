# Visual Guide - Evaluation & Visualization

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION WORKFLOW                      │
└─────────────────────────────────────────────────────────────┘

1. IMPORT & SETUP
   ├─ from evaluation_visualization import *
   └─ models_dict = {Model 1, Model 2, Model 3, Model 4}

2. MODEL COMPARISON (Quick Overview)
   ├─ plot_roc_curves()
   ├─ plot_precision_recall_curves()
   ├─ plot_confusion_matrices()
   └─ plot_model_comparison()

3. DETAILED ANALYSIS (Best Model)
   ├─ compute_feature_importance_gradients()
   ├─ plot_feature_importance()
   └─ plot_activation_maps()

4. ROBUSTNESS VALIDATION
   ├─ perform_cross_validation()
   ├─ plot_cross_validation_results()
   └─ print_cross_validation_summary()

5. ERROR ANALYSIS
   ├─ get_prediction_examples()
   ├─ plot_prediction_examples()
   └─ print_prediction_analysis()

6. FINAL REPORT
   └─ generate_evaluation_report()
```

## Function Call Hierarchy

```
evaluation_visualization.py
│
├─ ROC & PR Curves
│  ├─ plot_roc_curves()
│  └─ plot_precision_recall_curves()
│
├─ Confusion Matrices
│  └─ plot_confusion_matrices()
│
├─ Feature Analysis
│  ├─ compute_feature_importance_gradients()
│  ├─ plot_feature_importance()
│  └─ plot_activation_maps()
│
├─ Cross-Validation
│  ├─ perform_cross_validation()
│  ├─ plot_cross_validation_results()
│  └─ print_cross_validation_summary()
│
├─ Prediction Analysis
│  ├─ get_prediction_examples()
│  ├─ plot_prediction_examples()
│  └─ print_prediction_analysis()
│
├─ Model Comparison
│  └─ plot_model_comparison()
│
└─ Comprehensive Report
   └─ generate_evaluation_report()
```

## Visualization Types

### 1. ROC Curves
```
Sensitivity (TPR)
    1.0 ┌─────────────────────┐
        │    ╱╱╱╱╱╱╱╱╱╱╱╱╱  │ ← Good Model (AUC=0.95)
        │   ╱╱╱╱╱╱╱╱╱╱╱╱╱   │
        │  ╱╱╱╱╱╱╱╱╱╱╱╱╱    │
        │ ╱╱╱╱╱╱╱╱╱╱╱╱╱     │
        │╱╱╱╱╱╱╱╱╱╱╱╱╱      │
        │─────────────────── │ ← Random (AUC=0.5)
        │                    │
    0.0 └────────────────────┘
        0.0              1.0
        False Positive Rate (FPR)
```

### 2. Precision-Recall Curves
```
Precision
    1.0 ┌─────────────────────┐
        │╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │ ← Good Model (AP=0.92)
        │ ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │
        │  ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │
        │   ╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │
        │    ╲╲╲╲╲╲╲╲╲╲╲╲╲ │
        │─────────────────── │ ← Baseline
        │                    │
    0.0 └────────────────────┘
        0.0              1.0
        Recall
```

### 3. Confusion Matrix
```
                Predicted
              Positive  Negative
Actual  ┌──────────────────────┐
Pos     │  TP (True Pos)  FN   │
        │  (Correct)      (Miss)
Neg     │  FP (False Pos) TN   │
        │  (False Alarm)  (Correct)
        └──────────────────────┘
```

### 4. Feature Importance
```
Feature Importance (Top 20)

Feature 15  ████████████████████ 0.85
Feature 8   ██████████████████   0.78
Feature 22  ████████████████     0.72
Feature 3   ██████████████       0.65
Feature 19  ████████████         0.58
...
```

### 5. Activation Maps
```
Layer: conv2d_1 (8 filters shown)

Filter 0    Filter 1    Filter 2    Filter 3
[Image]     [Image]     [Image]     [Image]

Filter 4    Filter 5    Filter 6    Filter 7
[Image]     [Image]     [Image]     [Image]
```

### 6. Cross-Validation Results
```
Accuracy Distribution        Mean ± Std

  1.0 ┌─────────┐
      │    ┌─┐  │
  0.9 │    │ │  │  Model 1: 0.950 ± 0.015
      │    │ │  │  Model 2: 0.965 ± 0.012
  0.8 │    │ │  │  Model 3: 0.975 ± 0.010
      │    │ │  │  Model 4: 0.985 ± 0.008
      └────┴─┴──┘
```

### 7. Prediction Examples
```
CORRECT PREDICTIONS (Green)
┌──────────────┬──────────────┬──────────────┐
│ True: 1      │ True: 0      │ True: 1      │
│ Pred: 1      │ Pred: 0      │ Pred: 1      │
│ Conf: 0.98   │ Conf: 0.95   │ Conf: 0.97   │
└──────────────┴──────────────┴──────────────┘

INCORRECT PREDICTIONS (Red)
┌──────────────┬──────────────┬──────────────┐
│ True: 1      │ True: 0      │ True: 1      │
│ Pred: 0      │ Pred: 1      │ Pred: 0      │
│ Conf: 0.52   │ Conf: 0.48   │ Conf: 0.61   │
└──────────────┴──────────────┴──────────────┘
```

### 8. Model Comparison
```
Performance Metrics

1.0 ├─────────────────────────────────────┤
    │ ▓ Accuracy  ░ Precision  ▒ Recall  ▓ F1
0.9 ├─────────────────────────────────────┤
    │ ▓▓░░▒▒▓▓  ▓▓░░▒▒▓▓  ▓▓░░▒▒▓▓  ▓▓░░▒▒▓▓
0.8 ├─────────────────────────────────────┤
    │ M1  M2  M3  M4
```

## Decision Tree

```
START: Evaluate Models
│
├─ Want to compare all models?
│  ├─ YES → plot_roc_curves()
│  │        plot_precision_recall_curves()
│  │        plot_model_comparison()
│  └─ NO → Skip to next
│
├─ Want to understand errors?
│  ├─ YES → get_prediction_examples()
│  │        plot_prediction_examples()
│  │        print_prediction_analysis()
│  └─ NO → Skip to next
│
├─ Want to understand model internals?
│  ├─ YES → compute_feature_importance_gradients()
│  │        plot_feature_importance()
│  │        plot_activation_maps()
│  └─ NO → Skip to next
│
├─ Want to validate robustness?
│  ├─ YES → perform_cross_validation()
│  │        plot_cross_validation_results()
│  └─ NO → Skip to next
│
└─ Generate final report
   └─ generate_evaluation_report()
```

## Interpretation Guide

### ROC Curve
- **Upper-left corner** = Good model
- **AUC > 0.9** = Excellent
- **AUC > 0.8** = Good
- **AUC < 0.7** = Poor
- **Diagonal line** = Random classifier

### Precision-Recall Curve
- **Upper-right corner** = Good model
- **AP > 0.9** = Excellent
- **AP > 0.8** = Good
- **AP < 0.7** = Poor
- **Baseline** = Proportion of positive class

### Confusion Matrix
- **Diagonal high** = Good predictions
- **Off-diagonal high** = Misclassifications
- **Look for patterns** = Systematic errors

### Feature Importance
- **High values** = Important features
- **Low values** = Unimportant features
- **All equal** = Possible underfitting

### Activation Maps
- **Varied patterns** = Learning different features
- **All blank** = Layer not learning
- **Early layers** = Edges, textures
- **Later layers** = Complex patterns

### Cross-Validation
- **Low variance** = Stable model
- **High mean** = Good performance
- **High variance** = Overfitting
- **Low mean** = Underfitting

### Prediction Examples
- **Incorrect = hard cases** = Good model
- **Incorrect = easy cases** = Model issue
- **High confidence correct** = Good
- **High confidence incorrect** = Problem

## Quick Reference

| Need | Function | Time |
|------|----------|------|
| Compare models | `plot_model_comparison()` | 2-3s |
| ROC curves | `plot_roc_curves()` | 1-2s |
| PR curves | `plot_precision_recall_curves()` | 1-2s |
| Confusion matrices | `plot_confusion_matrices()` | 1-2s |
| Feature importance | `compute_feature_importance_gradients()` | 5-10s |
| Activation maps | `plot_activation_maps()` | 1-2s |
| Cross-validation | `perform_cross_validation()` | 5-10m |
| Prediction examples | `get_prediction_examples()` | <1s |
| Full analysis | `generate_evaluation_report()` | 2-3s |
