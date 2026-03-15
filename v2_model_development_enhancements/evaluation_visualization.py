"""
Evaluation & Visualization Module for HelmNet Models
Provides comprehensive evaluation metrics and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ROC CURVES
# ============================================================================

def plot_roc_curves(models_dict, X_test, y_test, figsize=(15, 5)):
    """
    Plot ROC curves for multiple models
    
    Args:
        models_dict: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred_proba = model.predict(X_test, verbose=0)
        
        # Handle binary vs multiclass
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = y_pred_proba.flatten()
        else:
            y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[idx].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate', fontsize=10)
        axes[idx].set_ylabel('True Positive Rate', fontsize=10)
        axes[idx].set_title(f'{name}\nROC Curve', fontsize=11, fontweight='bold')
        axes[idx].legend(loc="lower right", fontsize=9)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 2. PRECISION-RECALL CURVES
# ============================================================================

def plot_precision_recall_curves(models_dict, X_test, y_test, figsize=(15, 5)):
    """
    Plot Precision-Recall curves (better for imbalanced datasets)
    
    Args:
        models_dict: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred_proba = model.predict(X_test, verbose=0)
        
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = y_pred_proba.flatten()
        else:
            y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        axes[idx].plot(recall, precision, color='green', lw=2,
                      label=f'PR curve (AP = {avg_precision:.3f})')
        axes[idx].axhline(y=np.mean(y_test), color='red', linestyle='--', lw=2,
                         label=f'Baseline = {np.mean(y_test):.3f}')
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('Recall', fontsize=10)
        axes[idx].set_ylabel('Precision', fontsize=10)
        axes[idx].set_title(f'{name}\nPrecision-Recall Curve', fontsize=11, fontweight='bold')
        axes[idx].legend(loc="best", fontsize=9)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 3. FEATURE IMPORTANCE & ACTIVATION MAPS
# ============================================================================

def plot_activation_maps(model, X_test, layer_name, n_samples=3, figsize=(15, 4)):
    """
    Visualize activation maps from intermediate layers
    
    Args:
        model: Trained model
        X_test: Test images
        layer_name: Name of layer to visualize
        n_samples: Number of samples to show
        figsize: Figure size
    """
    # Create model that outputs intermediate layer
    intermediate_layer_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    intermediate_output = intermediate_layer_model.predict(X_test[:n_samples], verbose=0)
    
    n_features = min(8, intermediate_output.shape[-1])
    fig, axes = plt.subplots(n_samples, n_features, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            ax = axes[sample_idx, feature_idx]
            activation = intermediate_output[sample_idx, :, :, feature_idx]
            ax.imshow(activation, cmap='viridis')
            ax.axis('off')
            if sample_idx == 0:
                ax.set_title(f'Filter {feature_idx}', fontsize=9)
    
    fig.suptitle(f'Activation Maps - Layer: {layer_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def compute_feature_importance_gradients(model, X_test, y_test, n_samples=100):
    """
    Compute feature importance using gradient-based method
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        n_samples: Number of samples to use
    
    Returns:
        importance_scores: Average absolute gradients
    """
    X_test_tensor = tf.convert_to_tensor(X_test[:n_samples], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_test_tensor)
        predictions = model(X_test_tensor)
    
    gradients = tape.gradient(predictions, X_test_tensor)
    importance = tf.reduce_mean(tf.abs(gradients), axis=0)
    
    return importance.numpy()

def plot_feature_importance(importance_scores, figsize=(10, 6)):
    """
    Plot feature importance scores
    
    Args:
        importance_scores: Array of importance scores
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten if needed
    if len(importance_scores.shape) > 1:
        importance_scores = np.mean(importance_scores, axis=tuple(range(1, len(importance_scores.shape))))
    
    indices = np.argsort(importance_scores)[-20:]
    values = importance_scores[indices]
    
    ax.barh(range(len(values)), values, color='steelblue')
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Top 20 Feature Importance (Gradient-based)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 4. CROSS-VALIDATION RESULTS
# ============================================================================

def perform_cross_validation(model, X_train, y_train, cv_folds=5, metrics=['accuracy']):
    """
    Perform k-fold cross-validation
    
    Args:
        model: Compiled model
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of folds
        metrics: List of metrics to evaluate
    
    Returns:
        cv_results: Dict with cross-validation scores
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {metric: [] for metric in metrics}
    
    fold_num = 1
    for train_idx, val_idx in skf.split(X_train, y_train):
        print(f"  Fold {fold_num}/{cv_folds}...", end=" ")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train model on fold
        model.fit(X_fold_train, y_fold_train, epochs=10, batch_size=32,
                 validation_data=(X_fold_val, y_fold_val), verbose=0)
        
        # Evaluate on fold
        y_pred = model.predict(X_fold_val, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        
        for metric in metrics:
            if metric == 'accuracy':
                score = np.mean(y_pred_class == y_fold_val)
                cv_results[metric].append(score)
        
        print(f"Accuracy: {score:.4f}")
        fold_num += 1
    
    return cv_results

def plot_cross_validation_results(cv_results_dict, figsize=(12, 5)):
    """
    Plot cross-validation results for multiple models
    
    Args:
        cv_results_dict: Dict of {model_name: cv_results}
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    data_for_box = [scores['accuracy'] for scores in cv_results_dict.values()]
    bp = axes[0].boxplot(data_for_box, labels=cv_results_dict.keys(), patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Cross-Validation Accuracy Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Mean and std
    means = [np.mean(scores['accuracy']) for scores in cv_results_dict.values()]
    stds = [np.std(scores['accuracy']) for scores in cv_results_dict.values()]
    x_pos = np.arange(len(cv_results_dict))
    
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(cv_results_dict.keys())
    axes[1].set_ylabel('Mean Accuracy', fontsize=11)
    axes[1].set_title('Cross-Validation Mean ± Std', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0.8, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def print_cross_validation_summary(cv_results_dict):
    """Print cross-validation summary statistics"""
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    for model_name, cv_results in cv_results_dict.items():
        scores = cv_results['accuracy']
        print(f"\n{model_name}:")
        print(f"  Mean Accuracy:  {np.mean(scores):.4f}")
        print(f"  Std Dev:        {np.std(scores):.4f}")
        print(f"  Min:            {np.min(scores):.4f}")
        print(f"  Max:            {np.max(scores):.4f}")
        print(f"  Fold Scores:    {[f'{s:.4f}' for s in scores]}")

# ============================================================================
# 5. PREDICTION EXAMPLES & EXPLANATIONS
# ============================================================================

def get_prediction_examples(model, X_test, y_test, n_correct=3, n_incorrect=3):
    """
    Get examples of correct and incorrect predictions
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        n_correct: Number of correct predictions to show
        n_incorrect: Number of incorrect predictions to show
    
    Returns:
        examples: Dict with correct and incorrect examples
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
    y_pred_proba = np.max(y_pred, axis=1) if y_pred.shape[1] > 1 else y_pred.flatten()
    
    correct_mask = y_pred_class == y_test
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    examples = {
        'correct': [],
        'incorrect': []
    }
    
    # Get correct examples
    for idx in correct_indices[:n_correct]:
        examples['correct'].append({
            'index': idx,
            'true_label': y_test[idx],
            'pred_label': y_pred_class[idx],
            'confidence': y_pred_proba[idx],
            'image': X_test[idx]
        })
    
    # Get incorrect examples
    for idx in incorrect_indices[:n_incorrect]:
        examples['incorrect'].append({
            'index': idx,
            'true_label': y_test[idx],
            'pred_label': y_pred_class[idx],
            'confidence': y_pred_proba[idx],
            'image': X_test[idx]
        })
    
    return examples

def plot_prediction_examples(examples, figsize=(14, 6)):
    """
    Plot correct and incorrect prediction examples
    
    Args:
        examples: Dict from get_prediction_examples
        figsize: Figure size
    """
    n_correct = len(examples['correct'])
    n_incorrect = len(examples['incorrect'])
    total = n_correct + n_incorrect
    
    fig, axes = plt.subplots(2, max(n_correct, n_incorrect), figsize=figsize)
    
    # Plot correct predictions
    for i, ex in enumerate(examples['correct']):
        ax = axes[0, i]
        ax.imshow(ex['image'], cmap='gray')
        ax.set_title(f"✓ Correct\nTrue: {ex['true_label']}, Pred: {ex['pred_label']}\nConf: {ex['confidence']:.3f}",
                    fontsize=10, color='green', fontweight='bold')
        ax.axis('off')
    
    # Hide unused correct slots
    for i in range(n_correct, max(n_correct, n_incorrect)):
        axes[0, i].axis('off')
    
    # Plot incorrect predictions
    for i, ex in enumerate(examples['incorrect']):
        ax = axes[1, i]
        ax.imshow(ex['image'], cmap='gray')
        ax.set_title(f"✗ Incorrect\nTrue: {ex['true_label']}, Pred: {ex['pred_label']}\nConf: {ex['confidence']:.3f}",
                    fontsize=10, color='red', fontweight='bold')
        ax.axis('off')
    
    # Hide unused incorrect slots
    for i in range(n_incorrect, max(n_correct, n_incorrect)):
        axes[1, i].axis('off')
    
    fig.suptitle('Prediction Examples: Correct vs Incorrect', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

def print_prediction_analysis(model, X_test, y_test, class_names=None):
    """
    Print detailed prediction analysis
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
    y_pred_proba = np.max(y_pred, axis=1) if y_pred.shape[1] > 1 else y_pred.flatten()
    
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)
    
    # Overall accuracy
    accuracy = np.mean(y_pred_class == y_test)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean Confidence:     {np.mean(y_pred_proba):.4f}")
    print(f"  Median Confidence:   {np.median(y_pred_proba):.4f}")
    print(f"  Min Confidence:      {np.min(y_pred_proba):.4f}")
    print(f"  Max Confidence:      {np.max(y_pred_proba):.4f}")
    
    # Correct vs incorrect confidence
    correct_mask = y_pred_class == y_test
    print(f"\nCorrect Predictions:")
    print(f"  Count:               {np.sum(correct_mask)}")
    print(f"  Mean Confidence:     {np.mean(y_pred_proba[correct_mask]):.4f}")
    
    print(f"\nIncorrect Predictions:")
    print(f"  Count:               {np.sum(~correct_mask)}")
    print(f"  Mean Confidence:     {np.mean(y_pred_proba[~correct_mask]):.4f}")
    
    # Per-class analysis
    print(f"\nPer-Class Analysis:")
    unique_classes = np.unique(y_test)
    for cls in unique_classes:
        class_mask = y_test == cls
        class_accuracy = np.mean(y_pred_class[class_mask] == y_test[class_mask])
        class_name = class_names[cls] if class_names else f"Class {cls}"
        print(f"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_class, 
                               target_names=class_names if class_names else None))

# ============================================================================
# 6. COMPREHENSIVE EVALUATION REPORT
# ============================================================================

def generate_evaluation_report(models_dict, X_test, y_test, X_train=None, y_train=None, 
                              class_names=None, cv_folds=5):
    """
    Generate comprehensive evaluation report for all models
    
    Args:
        models_dict: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        X_train: Training features (for cross-validation)
        y_train: Training labels (for cross-validation)
        class_names: List of class names
        cv_folds: Number of CV folds
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*70)
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")
        
        # Test set evaluation
        y_pred = model.predict(X_test, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        
        test_accuracy = np.mean(y_pred_class == y_test)
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_class)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(classification_report(y_test, y_pred_class, 
                                   target_names=class_names if class_names else None))
        
        # ROC-AUC if binary
        if y_pred.shape[1] == 2 or y_pred.shape[1] == 1:
            y_pred_proba = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred.flatten()
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC Score: {roc_auc:.4f}")

def plot_confusion_matrices(models_dict, X_test, y_test, figsize=(15, 5)):
    """
    Plot confusion matrices for multiple models
    
    Args:
        models_dict: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(y_test, y_pred_class)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   cbar=False, square=True)
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(models_dict, X_test, y_test, figsize=(12, 6)):
    """
    Compare multiple models with various metrics
    
    Args:
        models_dict: Dict of {model_name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    """
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    model_names = []
    
    for name, model in models_dict.items():
        model_names.append(name)
        y_pred = model.predict(X_test, verbose=0)
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics['Accuracy'].append(np.mean(y_pred_class == y_test))
        metrics['Precision'].append(precision_score(y_test, y_pred_class, average='weighted', zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_pred_class, average='weighted', zero_division=0))
        metrics['F1-Score'].append(f1_score(y_test, y_pred_class, average='weighted', zero_division=0))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i*width, values, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison - Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (metric_name, values) in enumerate(metrics.items()):
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    return fig
