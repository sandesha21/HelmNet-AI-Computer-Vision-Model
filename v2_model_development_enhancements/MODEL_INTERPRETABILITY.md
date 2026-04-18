# Model Interpretability

## Overview
Understanding why the model makes specific predictions is crucial for trust, debugging, and regulatory compliance.

## Interpretability Techniques

### 1. Feature Importance Analysis

#### Permutation Feature Importance
```python
from sklearn.inspection import permutation_importance

def calculate_permutation_importance(model, X_test, y_test, n_repeats=10):
    """Calculate permutation-based feature importance"""
    
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df
```

**Interpretation**: Shows how much model performance decreases when a feature is randomly shuffled. Higher values indicate more important features.

#### SHAP (SHapley Additive exPlanations) Values
```python
import shap

def explain_with_shap(model, X_train, X_test):
    """Generate SHAP explanations"""
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    # Dependence plot for top feature
    top_feature = X_test.columns[0]
    shap.dependence_plot(top_feature, shap_values, X_test)
    
    return shap_values, explainer
```

**Interpretation**: SHAP values show each feature's contribution to pushing the prediction from the base value to the actual prediction.

#### Gradient-Based Importance (for Neural Networks)
```python
def calculate_gradient_importance(model, X_test):
    """Calculate gradient-based feature importance for neural networks"""
    
    X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
    
    gradients = tape.gradient(predictions, X_tensor)
    importance = tf.reduce_mean(tf.abs(gradients), axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': importance.numpy()
    }).sort_values('importance', ascending=False)
    
    return importance_df
```

### 2. Local Interpretability (Individual Predictions)

#### LIME (Local Interpretable Model-agnostic Explanations)
```python
import lime
import lime.tabular

def explain_prediction_lime(model, X_train, X_test, instance_idx):
    """Explain individual prediction using LIME"""
    
    explainer = lime.tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Class_0', 'Class_1'],
        mode='classification'
    )
    
    explanation = explainer.explain_instance(
        X_test.iloc[instance_idx].values,
        model.predict_proba,
        num_features=10
    )
    
    return explanation
```

**Use Case**: Understand why a specific prediction was made for an individual instance.

#### Counterfactual Explanations
```python
def generate_counterfactual(model, instance, target_class, feature_ranges):
    """Generate counterfactual explanation"""
    
    # Find minimal changes needed to change prediction
    current_pred = model.predict([instance])[0]
    
    if current_pred == target_class:
        return "Already predicts target class"
    
    # Iteratively modify features to reach target
    modified_instance = instance.copy()
    changes = {}
    
    for feature_idx in range(len(instance)):
        min_val, max_val = feature_ranges[feature_idx]
        
        # Try different values
        for new_val in np.linspace(min_val, max_val, 20):
            modified_instance[feature_idx] = new_val
            if model.predict([modified_instance])[0] == target_class:
                changes[feature_idx] = (instance[feature_idx], new_val)
                break
    
    return {
        'original_instance': instance,
        'counterfactual': modified_instance,
        'changes': changes
    }
```

### 3. Global Interpretability

#### Partial Dependence Plots
```python
from sklearn.inspection import partial_dependence

def plot_partial_dependence(model, X_test, features):
    """Plot partial dependence for features"""
    
    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))
    
    for idx, feature in enumerate(features):
        pd_result = partial_dependence(model, X_test, [feature])
        
        axes[idx].plot(pd_result['grid_values'][0], pd_result['average'][0])
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Partial Dependence')
        axes[idx].set_title(f'PDP: {feature}')
    
    plt.tight_layout()
    return fig
```

**Interpretation**: Shows the marginal effect of a feature on predictions, averaged over other features.

#### Accumulated Local Effects (ALE)
```python
def plot_ale(model, X_test, feature_idx, num_bins=50):
    """Plot Accumulated Local Effects"""
    
    feature_values = X_test.iloc[:, feature_idx].values
    sorted_indices = np.argsort(feature_values)
    
    bin_size = len(X_test) // num_bins
    ale_values = []
    bin_centers = []
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        
        bin_data = X_test.iloc[sorted_indices[start_idx:end_idx]]
        bin_center = feature_values[sorted_indices[start_idx:end_idx]].mean()
        
        predictions = model.predict(bin_data)
        ale_values.append(predictions.mean())
        bin_centers.append(bin_center)
    
    plt.plot(bin_centers, ale_values)
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('ALE')
    return plt.gcf()
```

### 4. Model-Specific Interpretability

#### Decision Trees & Random Forests
```python
from sklearn.tree import plot_tree

def visualize_decision_tree(tree_model, feature_names, class_names):
    """Visualize decision tree structure"""
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        ax=ax
    )
    return fig
```

#### Neural Network Attention Weights
```python
def extract_attention_weights(model, X_test):
    """Extract attention weights from attention layers"""
    
    # Get intermediate layer outputs
    attention_layer_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('attention').output
    )
    
    attention_weights = attention_layer_model.predict(X_test)
    
    return attention_weights
```

## Interpretability Report

```python
def generate_interpretability_report(model, X_train, X_test, y_test):
    """Generate comprehensive interpretability report"""
    
    report = {
        'feature_importance': calculate_permutation_importance(model, X_test, y_test),
        'shap_values': explain_with_shap(model, X_train, X_test),
        'top_features': get_top_features(model, X_test, n=10),
        'feature_interactions': detect_feature_interactions(model, X_test),
        'model_complexity': calculate_model_complexity(model),
        'interpretability_score': calculate_interpretability_score(model)
    }
    
    return report
```

## Interpretability Metrics

| Metric | Definition | Calculation |
|--------|-----------|-------------|
| Feature Importance | Contribution of feature to predictions | Permutation, SHAP, or gradient-based |
| Model Complexity | Difficulty of understanding model | Number of parameters, tree depth |
| Fidelity | How well explanation matches model | Correlation between explanation and prediction |
| Stability | Consistency of explanations | Variance across similar instances |
| Sparsity | Number of features needed for explanation | Average features per explanation |

## Best Practices

### For Stakeholders
1. **Use SHAP values** for global and local explanations
2. **Create feature importance plots** for top 10-15 features
3. **Generate counterfactual examples** for edge cases
4. **Document model assumptions** clearly

### For Debugging
1. **Analyze misclassified examples** with LIME
2. **Check for feature leakage** using permutation importance
3. **Validate feature engineering** with partial dependence plots
4. **Detect data issues** through unexpected feature importance

### For Compliance
1. **Maintain audit trail** of model decisions
2. **Document interpretability methods** used
3. **Provide explanations** for high-stakes decisions
4. **Regular interpretability audits** for model drift

## Interpretability Limitations

- **Explanations ≠ Causation**: Interpretability shows correlation, not causation
- **Local vs Global**: Local explanations may not reflect global model behavior
- **Approximation Error**: Some methods approximate model behavior
- **Computational Cost**: Some techniques are computationally expensive
- **Feature Interactions**: Complex interactions may be difficult to explain
