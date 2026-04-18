# Failure Analysis

## Overview
Systematic analysis of when, why, and how the model fails to make correct predictions.

## Types of Failures

### 1. False Positives (Type I Errors)
**Definition**: Model predicts positive when actual is negative

```python
def analyze_false_positives(y_true, y_pred, X_test, feature_names):
    """Analyze false positive predictions"""
    
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_indices = np.where(fp_mask)[0]
    
    fp_analysis = {
        'count': len(fp_indices),
        'rate': len(fp_indices) / len(y_true[y_true == 0]),
        'examples': X_test.iloc[fp_indices],
        'feature_stats': X_test.iloc[fp_indices].describe()
    }
    
    return fp_analysis
```

**Impact**: Depends on use case
- Medical diagnosis: Unnecessary treatment
- Fraud detection: False alarms
- Spam detection: Legitimate emails marked as spam

### 2. False Negatives (Type II Errors)
**Definition**: Model predicts negative when actual is positive

```python
def analyze_false_negatives(y_true, y_pred, X_test, feature_names):
    """Analyze false negative predictions"""
    
    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_indices = np.where(fn_mask)[0]
    
    fn_analysis = {
        'count': len(fn_indices),
        'rate': len(fn_indices) / len(y_true[y_true == 1]),
        'examples': X_test.iloc[fn_indices],
        'feature_stats': X_test.iloc[fn_indices].describe()
    }
    
    return fn_analysis
```

**Impact**: Depends on use case
- Medical diagnosis: Missed disease
- Fraud detection: Undetected fraud
- Spam detection: Spam reaches inbox

## Failure Pattern Detection

### 1. Systematic Failures by Feature Values
```python
def detect_failure_patterns(y_true, y_pred, X_test):
    """Detect systematic failure patterns"""
    
    failures = y_true != y_pred
    failure_data = X_test[failures]
    
    patterns = {}
    
    for col in X_test.columns:
        # Bin continuous features
        if X_test[col].dtype in ['float64', 'int64']:
            bins = pd.qcut(X_test[col], q=5, duplicates='drop')
            failure_rate_by_bin = failure_data[col].groupby(bins).apply(
                lambda x: (y_true[failure_data.index] != y_pred[failure_data.index]).sum() / len(x)
            )
        else:
            # Categorical features
            failure_rate_by_bin = failure_data[col].value_counts() / X_test[col].value_counts()
        
        patterns[col] = failure_rate_by_bin
    
    return patterns
```

### 2. Confidence-Based Failures
```python
def analyze_confidence_failures(y_true, y_pred_proba, y_pred):
    """Analyze failures by prediction confidence"""
    
    max_proba = np.max(y_pred_proba, axis=1)
    failures = y_true != y_pred
    
    # Bin by confidence
    confidence_bins = pd.cut(max_proba, bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    failure_analysis = pd.DataFrame({
        'confidence': max_proba,
        'is_failure': failures,
        'confidence_bin': confidence_bins
    }).groupby('confidence_bin').agg({
        'is_failure': ['sum', 'count', 'mean']
    })
    
    return failure_analysis
```

### 3. Boundary Failures
```python
def detect_boundary_failures(model, X_test, y_test, feature_pairs):
    """Detect failures near decision boundaries"""
    
    boundary_failures = {}
    
    for feat1, feat2 in feature_pairs:
        # Create mesh grid
        x_min, x_max = X_test[feat1].min() - 1, X_test[feat1].max() + 1
        y_min, y_max = X_test[feat2].min() - 1, X_test[feat2].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points).reshape(xx.shape)
        
        # Find test points near boundaries
        test_pred = model.predict(X_test[[feat1, feat2]])
        near_boundary = np.abs(Z[np.digitize(X_test[feat1], xx[0]) - 1,
                                  np.digitize(X_test[feat2], yy[:, 0]) - 1] - 0.5) < 0.1
        
        boundary_failures[f'{feat1}_vs_{feat2}'] = {
            'near_boundary_count': near_boundary.sum(),
            'failure_rate': (y_test[near_boundary] != test_pred[near_boundary]).mean()
        }
    
    return boundary_failures
```

## Root Cause Analysis

### 1. Data-Related Failures
```python
def analyze_data_related_failures(y_true, y_pred, X_test):
    """Identify data quality issues causing failures"""
    
    failures = y_true != y_pred
    failed_samples = X_test[failures]
    
    issues = {
        'missing_values': failed_samples.isna().sum(),
        'outliers': detect_outliers(failed_samples),
        'class_imbalance': y_true.value_counts(),
        'feature_distribution': failed_samples.describe()
    }
    
    return issues
```

### 2. Model-Related Failures
```python
def analyze_model_related_failures(model, X_test, y_test):
    """Identify model architecture issues"""
    
    # Check for overfitting
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    overfitting_gap = train_acc - test_acc
    
    # Check for underfitting
    underfitting = test_acc < 0.7
    
    # Check for class imbalance handling
    per_class_accuracy = {}
    for class_label in np.unique(y_test):
        mask = y_test == class_label
        per_class_accuracy[class_label] = accuracy_score(
            y_test[mask], test_pred[mask]
        )
    
    return {
        'overfitting_gap': overfitting_gap,
        'underfitting': underfitting,
        'per_class_accuracy': per_class_accuracy
    }
```

### 3. Feature-Related Failures
```python
def analyze_feature_related_failures(model, X_test, y_test):
    """Identify feature engineering issues"""
    
    # Feature importance for failures
    failures = y_test != model.predict(X_test)
    
    # Compare feature distributions for failures vs successes
    feature_analysis = {}
    
    for col in X_test.columns:
        failed_dist = X_test[failures][col].describe()
        success_dist = X_test[~failures][col].describe()
        
        feature_analysis[col] = {
            'failed_mean': failed_dist['mean'],
            'success_mean': success_dist['mean'],
            'mean_difference': failed_dist['mean'] - success_dist['mean']
        }
    
    return feature_analysis
```

## Failure Modes & Effects Analysis (FMEA)

```python
def create_fmea_table(failure_modes):
    """Create FMEA table for model failures"""
    
    fmea_data = []
    
    for mode in failure_modes:
        severity = mode.get('severity', 5)  # 1-10 scale
        occurrence = mode.get('occurrence', 5)  # 1-10 scale
        detection = mode.get('detection', 5)  # 1-10 scale
        
        rpn = severity * occurrence * detection  # Risk Priority Number
        
        fmea_data.append({
            'Failure Mode': mode['name'],
            'Potential Cause': mode['cause'],
            'Potential Effect': mode['effect'],
            'Severity': severity,
            'Occurrence': occurrence,
            'Detection': detection,
            'RPN': rpn,
            'Mitigation': mode['mitigation']
        })
    
    fmea_df = pd.DataFrame(fmea_data).sort_values('RPN', ascending=False)
    return fmea_df
```

## Failure Monitoring & Alerting

```python
def setup_failure_monitoring(model, X_test, y_test, thresholds):
    """Setup monitoring for model failures"""
    
    monitoring_config = {
        'false_positive_rate': {
            'threshold': thresholds.get('fp_rate', 0.05),
            'current': calculate_fp_rate(y_test, model.predict(X_test)),
            'alert': False
        },
        'false_negative_rate': {
            'threshold': thresholds.get('fn_rate', 0.05),
            'current': calculate_fn_rate(y_test, model.predict(X_test)),
            'alert': False
        },
        'per_class_accuracy': {
            'threshold': thresholds.get('min_accuracy', 0.80),
            'current': calculate_per_class_accuracy(y_test, model.predict(X_test)),
            'alert': False
        }
    }
    
    # Check alerts
    for metric, config in monitoring_config.items():
        if isinstance(config['current'], dict):
            config['alert'] = any(v < config['threshold'] for v in config['current'].values())
        else:
            config['alert'] = config['current'] > config['threshold']
    
    return monitoring_config
```

## Failure Recovery Strategies

| Failure Type | Detection | Recovery Strategy |
|-------------|-----------|------------------|
| Systematic FP | High FP rate for specific feature values | Retrain with class weights, feature engineering |
| Systematic FN | High FN rate for specific feature values | Adjust decision threshold, collect more data |
| Boundary Failures | High error near decision boundaries | Ensemble methods, uncertainty quantification |
| Data Drift | Increasing error over time | Retrain on recent data, online learning |
| Outlier Failures | Errors on extreme values | Robust scaling, outlier detection |
| Class Imbalance | Poor minority class performance | Resampling, weighted loss, SMOTE |

## Comprehensive Failure Report

```python
def generate_failure_report(model, X_test, y_test):
    """Generate comprehensive failure analysis report"""
    
    report = {
        'summary': {
            'total_failures': (y_test != model.predict(X_test)).sum(),
            'failure_rate': (y_test != model.predict(X_test)).mean(),
            'accuracy': accuracy_score(y_test, model.predict(X_test))
        },
        'false_positives': analyze_false_positives(y_test, model.predict(X_test), X_test),
        'false_negatives': analyze_false_negatives(y_test, model.predict(X_test), X_test),
        'failure_patterns': detect_failure_patterns(y_test, model.predict(X_test), X_test),
        'root_causes': analyze_model_related_failures(model, X_test, y_test),
        'recommendations': generate_recommendations(model, X_test, y_test)
    }
    
    return report
```
