# Data Quality Assessment

## Overview
Comprehensive assessment of data quality issues, their impact on model performance, and mitigation strategies.

## Data Quality Dimensions

### 1. Completeness
**Definition**: Percentage of non-missing values in the dataset

```python
def assess_completeness(df):
    """Assess data completeness"""
    completeness = {}
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        completeness[col] = {
            'missing_count': df[col].isna().sum(),
            'missing_percentage': missing_pct,
            'status': 'OK' if missing_pct < 5 else 'WARNING' if missing_pct < 20 else 'CRITICAL'
        }
    return completeness
```

**Thresholds**:
- ✅ OK: < 5% missing
- ⚠️ WARNING: 5-20% missing
- ❌ CRITICAL: > 20% missing

**Mitigation**:
- Forward/backward fill for time series
- Mean/median imputation for numerical features
- Mode imputation for categorical features
- Remove rows if > 30% missing

### 2. Accuracy
**Definition**: Correctness of data values against source of truth

```python
def assess_accuracy(df, validation_rules):
    """Assess data accuracy against validation rules"""
    accuracy_report = {}
    
    for col, rule in validation_rules.items():
        valid_count = rule(df[col]).sum()
        accuracy = (valid_count / len(df)) * 100
        accuracy_report[col] = {
            'valid_records': valid_count,
            'accuracy_percentage': accuracy,
            'status': 'OK' if accuracy > 95 else 'WARNING' if accuracy > 80 else 'CRITICAL'
        }
    
    return accuracy_report
```

**Common Issues**:
- Typos in categorical values
- Out-of-range numerical values
- Incorrect data types
- Logical inconsistencies

### 3. Consistency
**Definition**: Uniformity of data format and values across the dataset

```python
def assess_consistency(df):
    """Assess data consistency"""
    consistency_issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        consistency_issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for inconsistent formatting
    for col in df.select_dtypes(include=['object']).columns:
        unique_formats = df[col].str.len().nunique()
        if unique_formats > 5:
            consistency_issues.append(f"Column {col} has inconsistent formatting")
    
    return consistency_issues
```

**Issues to Check**:
- Duplicate records
- Inconsistent date formats
- Mixed case in categorical values
- Inconsistent units (e.g., kg vs lbs)

### 4. Validity
**Definition**: Conformance to defined formats and domains

```python
def assess_validity(df, schema):
    """Assess data validity against schema"""
    validity_report = {}
    
    for col, col_schema in schema.items():
        if col not in df.columns:
            validity_report[col] = {'status': 'MISSING', 'valid_records': 0}
            continue
        
        # Check data type
        type_valid = df[col].dtype == col_schema['type']
        
        # Check value range
        if 'min' in col_schema and 'max' in col_schema:
            range_valid = (df[col] >= col_schema['min']).all() and \
                         (df[col] <= col_schema['max']).all()
        else:
            range_valid = True
        
        validity_report[col] = {
            'type_valid': type_valid,
            'range_valid': range_valid,
            'status': 'OK' if type_valid and range_valid else 'INVALID'
        }
    
    return validity_report
```

**Validation Rules**:
- Data type matches schema
- Values within acceptable range
- Categorical values in allowed set
- Required fields are present

### 5. Uniqueness
**Definition**: Absence of unintended duplicates

```python
def assess_uniqueness(df, key_columns):
    """Assess data uniqueness"""
    uniqueness_report = {}
    
    for key_col in key_columns:
        unique_count = df[key_col].nunique()
        total_count = len(df)
        uniqueness_pct = (unique_count / total_count) * 100
        
        duplicates = total_count - unique_count
        
        uniqueness_report[key_col] = {
            'unique_values': unique_count,
            'total_values': total_count,
            'duplicates': duplicates,
            'uniqueness_percentage': uniqueness_pct,
            'status': 'OK' if duplicates == 0 else 'WARNING'
        }
    
    return uniqueness_report
```

## Data Quality Issues & Impact

| Issue | Severity | Impact on Model | Detection Method | Mitigation |
|-------|----------|-----------------|------------------|-----------|
| Missing Values | High | Reduced training data, biased estimates | Null checks | Imputation, removal |
| Outliers | Medium | Skewed distributions, poor generalization | Statistical tests | Clipping, removal, robust scaling |
| Duplicates | High | Data leakage, inflated metrics | Duplicate detection | Remove duplicates |
| Inconsistent Format | Medium | Feature engineering errors | Pattern matching | Standardization |
| Incorrect Labels | Critical | Model learns wrong patterns | Manual review, consistency checks | Relabeling, removal |
| Class Imbalance | Medium | Biased predictions toward majority class | Class distribution analysis | Resampling, weighted loss |
| Feature Drift | High | Model performance degradation | Distribution comparison | Retraining, monitoring |

## Comprehensive Quality Assessment

```python
def comprehensive_data_quality_assessment(df, schema, validation_rules, key_columns):
    """Perform comprehensive data quality assessment"""
    
    assessment = {
        'completeness': assess_completeness(df),
        'accuracy': assess_accuracy(df, validation_rules),
        'consistency': assess_consistency(df),
        'validity': assess_validity(df, schema),
        'uniqueness': assess_uniqueness(df, key_columns),
        'timestamp': datetime.now().isoformat()
    }
    
    # Calculate overall quality score
    scores = []
    for dimension in ['completeness', 'accuracy', 'validity']:
        dimension_scores = [
            v['accuracy_percentage'] if 'accuracy_percentage' in v else 100
            for v in assessment[dimension].values()
        ]
        scores.extend(dimension_scores)
    
    overall_score = np.mean(scores) if scores else 0
    assessment['overall_quality_score'] = overall_score
    assessment['quality_status'] = 'EXCELLENT' if overall_score > 95 else \
                                   'GOOD' if overall_score > 85 else \
                                   'FAIR' if overall_score > 70 else 'POOR'
    
    return assessment
```

## Quality Monitoring Dashboard

### Key Metrics
- **Overall Quality Score**: Weighted average of all dimensions
- **Data Freshness**: Time since last data update
- **Anomaly Detection Rate**: Percentage of records flagged as anomalies
- **Schema Compliance**: Percentage of records matching schema
- **Duplicate Rate**: Percentage of duplicate records

### Alerting Rules
- Quality Score < 80%: Investigate data pipeline
- Missing Data > 10%: Review data collection process
- Duplicate Rate > 1%: Check for data ingestion issues
- Schema Violations > 5%: Update schema or fix data

## Data Quality Report Template

```python
def generate_quality_report(assessment):
    """Generate human-readable quality report"""
    
    report = f"""
    DATA QUALITY ASSESSMENT REPORT
    Generated: {assessment['timestamp']}
    
    OVERALL QUALITY SCORE: {assessment['overall_quality_score']:.2f}%
    STATUS: {assessment['quality_status']}
    
    COMPLETENESS:
    {json.dumps(assessment['completeness'], indent=2)}
    
    ACCURACY:
    {json.dumps(assessment['accuracy'], indent=2)}
    
    CONSISTENCY ISSUES:
    {json.dumps(assessment['consistency'], indent=2)}
    
    VALIDITY:
    {json.dumps(assessment['validity'], indent=2)}
    
    UNIQUENESS:
    {json.dumps(assessment['uniqueness'], indent=2)}
    
    RECOMMENDATIONS:
    - Review data collection process
    - Implement automated quality checks
    - Establish data governance policies
    - Schedule regular quality audits
    """
    
    return report
```
