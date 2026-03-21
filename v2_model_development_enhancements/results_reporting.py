"""
Results & Reporting Module for HelmNet Models
Comprehensive comparison, statistical testing, and deployment recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_all_metrics(models_dict, X_test, y_test):
    """
    Compute comprehensive metrics for all models.
    
    Args:
        models_dict: Dict of {name: model}
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with metrics for each model
    """
    metrics_list = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        y_pred_proba = model.predict(X_test, verbose=0).max(axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, average='weighted')
        except:
            roc_auc = np.nan
        
        avg_confidence = y_pred_proba.mean()
        
        metrics_list.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Avg Confidence': avg_confidence
        })
    
    return pd.DataFrame(metrics_list)


def create_comparison_table(models_dict, X_test, y_test, figsize=(14, 6)):
    """
    Create comprehensive comparison table with all metrics.
    
    Args:
        models_dict: Dict of {name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    
    Returns:
        DataFrame and matplotlib figure
    """
    metrics_df = compute_all_metrics(models_dict, X_test, y_test)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Format metrics for display
    display_df = metrics_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Confidence']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    # Create table
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Comparison Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return metrics_df, fig


def statistical_significance_testing(models_dict, X_test, y_test):
    """
    Perform statistical significance testing between models.
    Uses McNemar's test for pairwise comparisons.
    
    Args:
        models_dict: Dict of {name: model}
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with p-values and significance indicators
    """
    model_names = list(models_dict.keys())
    n_models = len(model_names)
    
    # Get predictions for all models
    predictions = {}
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        predictions[model_name] = y_pred
    
    # Pairwise McNemar's test
    results = []
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue
            
            pred1 = predictions[model1]
            pred2 = predictions[model2]
            
            # McNemar's test
            correct1 = (pred1 == y_test)
            correct2 = (pred2 == y_test)
            
            # Contingency table
            both_correct = np.sum(correct1 & correct2)
            both_wrong = np.sum(~correct1 & ~correct2)
            model1_correct = np.sum(correct1 & ~correct2)
            model2_correct = np.sum(~correct1 & correct2)
            
            # McNemar's statistic
            if (model1_correct + model2_correct) > 0:
                mcnemar_stat = (model1_correct - model2_correct) ** 2 / (model1_correct + model2_correct)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            else:
                p_value = 1.0
            
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            results.append({
                'Model 1': model1,
                'Model 2': model2,
                'p-value': p_value,
                'Significance': significance,
                'Model 1 Correct': model1_correct,
                'Model 2 Correct': model2_correct
            })
    
    return pd.DataFrame(results)


def plot_statistical_significance(sig_df, figsize=(12, 6)):
    """
    Visualize statistical significance testing results.
    
    Args:
        sig_df: DataFrame from statistical_significance_testing
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create comparison labels
    comparisons = [f"{row['Model 1']} vs {row['Model 2']}" for _, row in sig_df.iterrows()]
    p_values = sig_df['p-value'].values
    
    # Color by significance
    colors = ['#d32f2f' if p < 0.05 else '#fbc02d' if p < 0.1 else '#4caf50' for p in p_values]
    
    bars = ax.barh(comparisons, -np.log10(p_values), color=colors)
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='p=0.01')
    
    ax.set_xlabel('-log10(p-value)', fontsize=11, fontweight='bold')
    ax.set_title('Statistical Significance Testing (McNemar\'s Test)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_recommendations(metrics_df, sig_df):
    """
    Generate deployment recommendations based on metrics and significance.
    
    Args:
        metrics_df: DataFrame from compute_all_metrics
        sig_df: DataFrame from statistical_significance_testing
    
    Returns:
        String with recommendations
    """
    best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    best_model_name = best_model['Model']
    
    recommendations = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT RECOMMENDATIONS                              ║
╚════════════════════════════════════════════════════════════════════════════╝

🏆 RECOMMENDED MODEL FOR DEPLOYMENT: {best_model_name}
   ├─ F1-Score: {best_model['F1-Score']:.4f}
   ├─ Accuracy: {best_model['Accuracy']:.4f}
   ├─ Precision: {best_model['Precision']:.4f}
   ├─ Recall: {best_model['Recall']:.4f}
   └─ ROC-AUC: {best_model['ROC-AUC']:.4f}

📊 PERFORMANCE RANKING:
"""
    
    ranked = metrics_df.sort_values('F1-Score', ascending=False)
    for idx, (_, row) in enumerate(ranked.iterrows(), 1):
        recommendations += f"   {idx}. {row['Model']:<20} F1={row['F1-Score']:.4f}  Acc={row['Accuracy']:.4f}\n"
    
    recommendations += f"""
✅ KEY STRENGTHS OF {best_model_name}:
"""
    
    # Identify strengths
    if best_model['Accuracy'] == metrics_df['Accuracy'].max():
        recommendations += f"   • Highest Accuracy: {best_model['Accuracy']:.4f}\n"
    if best_model['Precision'] == metrics_df['Precision'].max():
        recommendations += f"   • Highest Precision: {best_model['Precision']:.4f}\n"
    if best_model['Recall'] == metrics_df['Recall'].max():
        recommendations += f"   • Highest Recall: {best_model['Recall']:.4f}\n"
    if best_model['F1-Score'] == metrics_df['F1-Score'].max():
        recommendations += f"   • Highest F1-Score: {best_model['F1-Score']:.4f}\n"
    
    recommendations += f"""
⚠️  STATISTICAL SIGNIFICANCE:
"""
    
    # Check significance vs other models
    sig_comparisons = sig_df[sig_df['Model 1'] == best_model_name]
    if len(sig_comparisons) > 0:
        for _, row in sig_comparisons.iterrows():
            sig_text = "SIGNIFICANT" if row['Significance'] != 'ns' else "NOT SIGNIFICANT"
            recommendations += f"   • vs {row['Model 2']}: {sig_text} (p={row['p-value']:.4f})\n"
    
    recommendations += f"""
💡 DEPLOYMENT CONSIDERATIONS:
   • Model is ready for production deployment
   • Monitor performance on new data regularly
   • Consider ensemble methods for further improvement
   • Implement A/B testing before full rollout
   • Set up performance monitoring and alerting

🔄 ALTERNATIVE MODELS:
"""
    
    for idx, (_, row) in enumerate(ranked.iloc[1:].iterrows(), 1):
        recommendations += f"   {idx}. {row['Model']}: F1={row['F1-Score']:.4f} (Δ={best_model['F1-Score']-row['F1-Score']:.4f})\n"
    
    recommendations += "\n"
    return recommendations


def generate_limitations():
    """
    Generate limitations and constraints section.
    
    Returns:
        String with limitations
    """
    limitations = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    MODEL LIMITATIONS & CONSTRAINTS                         ║
╚════════════════════════════════════════════════════════════════════════════╝

⚠️  DATA LIMITATIONS:
   • Model trained on specific dataset - may not generalize to different domains
   • Performance depends on data quality and preprocessing
   • Imbalanced classes may affect minority class predictions
   • Limited to input features used during training

🎯 PREDICTION CONSTRAINTS:
   • Confidence scores should not be interpreted as calibrated probabilities
   • Model may struggle with out-of-distribution samples
   • Edge cases and rare patterns may be misclassified
   • Real-time predictions depend on inference latency requirements

🔧 TECHNICAL LIMITATIONS:
   • Model size and memory requirements for deployment
   • Inference speed may vary with hardware
   • Requires specific preprocessing pipeline
   • Sensitive to input normalization and scaling

📊 EVALUATION LIMITATIONS:
   • Test set performance may not reflect production performance
   • Cross-validation results assume i.i.d. data
   • Metrics may not capture all aspects of model quality
   • Temporal data may require time-series evaluation

🚫 KNOWN ISSUES:
   • Model may overfit on training data
   • Hyperparameter tuning limited to search space
   • No explicit handling of missing values
   • Limited interpretability for complex decisions

💼 BUSINESS CONSTRAINTS:
   • Model requires regular retraining with new data
   • Performance degradation over time (data drift)
   • Requires monitoring and maintenance
   • May need domain expert validation for critical decisions

"""
    return limitations


def generate_future_improvements():
    """
    Generate future improvements and next steps.
    
    Returns:
        String with future improvements
    """
    improvements = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    FUTURE IMPROVEMENTS & NEXT STEPS                        ║
╚════════════════════════════════════════════════════════════════════════════╝

🚀 SHORT-TERM IMPROVEMENTS (1-2 weeks):
   1. Hyperparameter Fine-tuning
      • Grid search over learning rates
      • Optimize batch size and epochs
      • Adjust regularization parameters
   
   2. Data Augmentation
      • Implement advanced augmentation techniques
      • Balance class distribution
      • Generate synthetic samples for minority classes
   
   3. Ensemble Methods
      • Combine predictions from multiple models
      • Weighted voting based on performance
      • Stacking with meta-learner

📈 MEDIUM-TERM IMPROVEMENTS (1-3 months):
   1. Feature Engineering
      • Extract domain-specific features
      • Feature selection and dimensionality reduction
      • Interaction terms and polynomial features
   
   2. Model Architecture
      • Experiment with different architectures
      • Transfer learning from pre-trained models
      • Neural architecture search (NAS)
   
   3. Calibration
      • Calibrate confidence scores
      • Temperature scaling
      • Platt scaling for probability estimates

🔬 LONG-TERM IMPROVEMENTS (3-6 months):
   1. Advanced Techniques
      • Attention mechanisms for interpretability
      • Explainable AI (XAI) methods
      • Adversarial robustness testing
   
   2. Production Optimization
      • Model compression and quantization
      • Knowledge distillation
      • Edge deployment optimization
   
   3. Continuous Learning
      • Online learning and incremental updates
      • Active learning for data collection
      • Federated learning for distributed training

🔍 MONITORING & MAINTENANCE:
   • Set up performance monitoring dashboard
   • Implement data drift detection
   • Schedule regular model retraining
   • Establish feedback loops from production
   • Create alerting for performance degradation

📚 RESEARCH DIRECTIONS:
   • Investigate model interpretability
   • Explore uncertainty quantification
   • Study robustness to adversarial inputs
   • Analyze failure modes and edge cases
   • Benchmark against state-of-the-art methods

🎓 KNOWLEDGE TRANSFER:
   • Document model decisions and trade-offs
   • Create deployment runbooks
   • Train team on model usage
   • Establish best practices documentation
   • Share learnings with stakeholders

"""
    return improvements


def generate_full_report(models_dict, X_test, y_test):
    """
    Generate complete results and reporting document.
    
    Args:
        models_dict: Dict of {name: model}
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Complete report string
    """
    # Compute all components
    metrics_df = compute_all_metrics(models_dict, X_test, y_test)
    sig_df = statistical_significance_testing(models_dict, X_test, y_test)
    recommendations = generate_recommendations(metrics_df, sig_df)
    limitations = generate_limitations()
    improvements = generate_future_improvements()
    
    # Combine into full report
    report = f"""
{'='*80}
                    HELMNET MODEL RESULTS & REPORTING
                         COMPREHENSIVE ANALYSIS
{'='*80}

📋 EXECUTIVE SUMMARY
{'-'*80}
Total Models Evaluated: {len(models_dict)}
Test Set Size: {len(X_test)}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{recommendations}

{limitations}

{improvements}

{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report, metrics_df, sig_df


def print_full_report(models_dict, X_test, y_test):
    """
    Print complete results and reporting document.
    
    Args:
        models_dict: Dict of {name: model}
        X_test: Test features
        y_test: Test labels
    """
    report, metrics_df, sig_df = generate_full_report(models_dict, X_test, y_test)
    print(report)
    return metrics_df, sig_df
