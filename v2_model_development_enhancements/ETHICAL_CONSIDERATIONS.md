# Ethical Considerations

## Overview
Comprehensive framework for addressing privacy, bias, fairness, and ethical concerns in model development and deployment.

## 1. Privacy Concerns

### Data Privacy
```python
def assess_privacy_risks(X_train, sensitive_features):
    """Assess privacy risks in training data"""
    
    privacy_assessment = {
        'sensitive_features': sensitive_features,
        'data_retention_policy': 'Define retention period',
        'anonymization_status': check_anonymization(X_train, sensitive_features),
        'pii_detection': detect_pii(X_train),
        'encryption_status': 'Verify encryption at rest and in transit'
    }
    
    return privacy_assessment

def detect_pii(df):
    """Detect Personally Identifiable Information"""
    
    pii_patterns = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    }
    
    detected_pii = {}
    for col in df.select_dtypes(include=['object']).columns:
        for pii_type, pattern in pii_patterns.items():
            matches = df[col].astype(str).str.contains(pattern, regex=True).sum()
            if matches > 0:
                detected_pii[f'{col}_{pii_type}'] = matches
    
    return detected_pii
```

### Model Privacy (Membership Inference)
```python
def assess_membership_inference_risk(model, X_train, X_test, y_train, y_test):
    """Assess risk of membership inference attacks"""
    
    # Train model on training data
    train_pred = model.predict(X_train)
    train_confidence = np.max(model.predict_proba(X_train), axis=1)
    
    # Test on test data
    test_pred = model.predict(X_test)
    test_confidence = np.max(model.predict_proba(X_test), axis=1)
    
    # Calculate separation
    train_correct = (train_pred == y_train).astype(int)
    test_correct = (test_pred == y_test).astype(int)
    
    # If model is much more confident on training data, membership inference is easier
    train_avg_confidence = train_confidence[train_correct == 1].mean()
    test_avg_confidence = test_confidence[test_correct == 1].mean()
    
    membership_risk = {
        'train_avg_confidence': train_avg_confidence,
        'test_avg_confidence': test_avg_confidence,
        'confidence_gap': train_avg_confidence - test_avg_confidence,
        'risk_level': 'HIGH' if train_avg_confidence - test_avg_confidence > 0.1 else 'LOW'
    }
    
    return membership_risk
```

### Data Minimization
```python
def assess_data_minimization(X_train, feature_importance):
    """Assess if only necessary features are used"""
    
    # Identify low-importance features
    low_importance_threshold = 0.01
    low_importance_features = feature_importance[
        feature_importance['importance'] < low_importance_threshold
    ]['feature'].tolist()
    
    minimization_report = {
        'total_features': len(X_train.columns),
        'low_importance_features': low_importance_features,
        'recommendation': f'Consider removing {len(low_importance_features)} low-importance features',
        'data_reduction_potential': len(low_importance_features) / len(X_train.columns)
    }
    
    return minimization_report
```

## 2. Bias Detection & Mitigation

### Demographic Parity
```python
def check_demographic_parity(y_pred, sensitive_attr, threshold=0.8):
    """Check if positive prediction rate is similar across groups"""
    
    groups = np.unique(sensitive_attr)
    positive_rates = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        positive_rate = (y_pred[group_mask] == 1).mean()
        positive_rates[group] = positive_rate
    
    # Calculate disparate impact ratio
    min_rate = min(positive_rates.values())
    max_rate = max(positive_rates.values())
    disparate_impact = min_rate / max_rate if max_rate > 0 else 0
    
    return {
        'positive_rates': positive_rates,
        'disparate_impact_ratio': disparate_impact,
        'passes_threshold': disparate_impact >= threshold,
        'recommendation': 'Adjust decision threshold or retrain' if disparate_impact < threshold else 'OK'
    }
```

### Equalized Odds
```python
def check_equalized_odds(y_true, y_pred, sensitive_attr, threshold=0.8):
    """Check if TPR and FPR are similar across groups"""
    
    groups = np.unique(sensitive_attr)
    tpr_by_group = {}
    fpr_by_group = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        
        # True Positive Rate
        tp = ((y_pred[group_mask] == 1) & (y_true[group_mask] == 1)).sum()
        fn = ((y_pred[group_mask] == 0) & (y_true[group_mask] == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_by_group[group] = tpr
        
        # False Positive Rate
        fp = ((y_pred[group_mask] == 1) & (y_true[group_mask] == 0)).sum()
        tn = ((y_pred[group_mask] == 0) & (y_true[group_mask] == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_by_group[group] = fpr
    
    # Calculate variance
    tpr_variance = np.var(list(tpr_by_group.values()))
    fpr_variance = np.var(list(fpr_by_group.values()))
    
    return {
        'tpr_by_group': tpr_by_group,
        'fpr_by_group': fpr_by_group,
        'tpr_variance': tpr_variance,
        'fpr_variance': fpr_variance,
        'passes_threshold': tpr_variance < (1 - threshold) and fpr_variance < (1 - threshold)
    }
```

### Calibration Across Groups
```python
def check_calibration_fairness(y_true, y_pred_proba, sensitive_attr):
    """Check if model calibration is similar across demographic groups"""
    
    groups = np.unique(sensitive_attr)
    calibration_by_group = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true[group_mask],
            y_pred_proba[group_mask],
            n_bins=10
        )
        
        # Expected Calibration Error
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        calibration_by_group[group] = {
            'ece': ece,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
    
    return calibration_by_group
```

### Bias Mitigation Strategies
```python
def apply_bias_mitigation(X_train, y_train, sensitive_attr, method='reweighting'):
    """Apply bias mitigation techniques"""
    
    if method == 'reweighting':
        # Reweight samples to balance representation
        sample_weights = compute_sample_weights(sensitive_attr, y_train)
        return sample_weights
    
    elif method == 'threshold_optimization':
        # Optimize decision threshold per group
        thresholds = {}
        for group in np.unique(sensitive_attr):
            group_mask = sensitive_attr == group
            # Find threshold that maximizes fairness metric
            thresholds[group] = optimize_threshold(y_train[group_mask])
        return thresholds
    
    elif method == 'adversarial_debiasing':
        # Use adversarial training to remove sensitive attribute information
        # Implementation depends on model architecture
        pass
```

## 3. Fairness Assessment

### Fairness Metrics Dashboard
```python
def create_fairness_dashboard(y_true, y_pred, y_pred_proba, sensitive_attrs):
    """Create comprehensive fairness assessment dashboard"""
    
    fairness_report = {}
    
    for attr_name, attr_values in sensitive_attrs.items():
        fairness_report[attr_name] = {
            'demographic_parity': check_demographic_parity(y_pred, attr_values),
            'equalized_odds': check_equalized_odds(y_true, y_pred, attr_values),
            'calibration': check_calibration_fairness(y_true, y_pred_proba, attr_values),
            'accuracy_by_group': calculate_accuracy_by_group(y_true, y_pred, attr_values)
        }
    
    return fairness_report
```

### Accuracy by Demographic Group
```python
def calculate_accuracy_by_group(y_true, y_pred, sensitive_attr):
    """Calculate accuracy for each demographic group"""
    
    groups = np.unique(sensitive_attr)
    accuracy_by_group = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
        accuracy_by_group[group] = accuracy
    
    # Check for disparities
    min_accuracy = min(accuracy_by_group.values())
    max_accuracy = max(accuracy_by_group.values())
    accuracy_gap = max_accuracy - min_accuracy
    
    return {
        'accuracy_by_group': accuracy_by_group,
        'accuracy_gap': accuracy_gap,
        'recommendation': 'Investigate' if accuracy_gap > 0.1 else 'Acceptable'
    }
```

## 4. Transparency & Accountability

### Model Card
```python
def create_model_card(model_name, model_version, intended_use, limitations):
    """Create model card for transparency"""
    
    model_card = f"""
    # Model Card: {model_name}
    
    ## Model Details
    - Version: {model_version}
    - Type: {type(model).__name__}
    - Training Date: {datetime.now().isoformat()}
    
    ## Intended Use
    {intended_use}
    
    ## Limitations
    {limitations}
    
    ## Performance Metrics
    - Accuracy: [INSERT]
    - Precision: [INSERT]
    - Recall: [INSERT]
    - F1-Score: [INSERT]
    
    ## Fairness Assessment
    - Demographic Parity: [INSERT]
    - Equalized Odds: [INSERT]
    - Calibration: [INSERT]
    
    ## Ethical Considerations
    - Privacy Risks: [INSERT]
    - Bias Concerns: [INSERT]
    - Mitigation Strategies: [INSERT]
    
    ## Recommendations
    - Use Cases: [INSERT]
    - Avoid Using For: [INSERT]
    - Monitoring Requirements: [INSERT]
    """
    
    return model_card
```

### Audit Trail
```python
def create_audit_trail(model, X_test, y_test, predictions):
    """Create audit trail for model decisions"""
    
    audit_trail = {
        'timestamp': datetime.now().isoformat(),
        'model_version': getattr(model, 'version', 'unknown'),
        'test_set_size': len(X_test),
        'accuracy': accuracy_score(y_test, predictions),
        'predictions_made': len(predictions),
        'high_confidence_predictions': (np.max(model.predict_proba(X_test), axis=1) > 0.9).sum(),
        'low_confidence_predictions': (np.max(model.predict_proba(X_test), axis=1) < 0.6).sum()
    }
    
    return audit_trail
```

## 5. Ethical Decision Framework

```python
def ethical_review_checklist():
    """Ethical review checklist for model deployment"""
    
    checklist = {
        'Privacy': {
            'PII removed from training data': False,
            'Data retention policy defined': False,
            'Encryption implemented': False,
            'Access controls in place': False
        },
        'Bias & Fairness': {
            'Demographic parity checked': False,
            'Equalized odds verified': False,
            'Accuracy gaps investigated': False,
            'Mitigation strategies applied': False
        },
        'Transparency': {
            'Model card created': False,
            'Limitations documented': False,
            'Interpretability analysis done': False,
            'Stakeholders informed': False
        },
        'Accountability': {
            'Audit trail established': False,
            'Monitoring plan created': False,
            'Escalation procedures defined': False,
            'Responsible party assigned': False
        },
        'Safety': {
            'Failure modes identified': False,
            'Rollback plan prepared': False,
            'Human oversight defined': False,
            'Testing completed': False
        }
    }
    
    return checklist
```

## 6. Regulatory Compliance

### GDPR Compliance
- **Right to Explanation**: Provide explanations for automated decisions
- **Data Minimization**: Use only necessary data
- **Consent**: Obtain explicit consent for data usage
- **Right to Deletion**: Implement data deletion mechanisms

### AI Act Compliance (EU)
- **High-Risk Assessment**: Classify model risk level
- **Documentation**: Maintain detailed documentation
- **Monitoring**: Implement continuous monitoring
- **Human Oversight**: Ensure human review for high-risk decisions

### Fair Lending (US)
- **Disparate Impact**: Monitor for unintended discrimination
- **Adverse Action Notices**: Provide explanations for denials
- **Record Keeping**: Maintain audit trails
- **Regular Testing**: Conduct periodic fairness audits

## Ethical Considerations Summary

| Concern | Risk Level | Mitigation |
|---------|-----------|-----------|
| Privacy Breach | High | Encryption, anonymization, access controls |
| Demographic Bias | High | Fairness metrics, bias mitigation, monitoring |
| Model Interpretability | Medium | SHAP, LIME, feature importance analysis |
| Accountability Gap | Medium | Audit trails, model cards, documentation |
| Regulatory Non-Compliance | High | Legal review, compliance checklist |
| Unintended Consequences | Medium | Failure analysis, human oversight |

## Resources & References

- [Fairness Indicators](https://github.com/tensorflow/fairness-indicators)
- [AI Fairness 360](https://github.com/Trusted-AI/AIF360)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [GDPR Compliance Guide](https://gdpr-info.eu/)
- [Responsible AI Practices](https://ai.google/principles/)
