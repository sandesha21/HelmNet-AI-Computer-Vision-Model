# Results & Reporting Notebook Cells

Copy-paste ready notebook cells for Results & Reporting.

## Cell 1: Import Module

```python
# Import Results & Reporting module
from results_reporting import *

print("✅ Results & Reporting module imported successfully")
```

## Cell 2: Create Models Dictionary

```python
# Create dictionary of all trained models
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

print(f"✅ Models dictionary created with {len(models_dict)} models")
for name in models_dict.keys():
    print(f"   • {name}")
```

## Cell 3: Compute All Metrics

```python
# Compute comprehensive metrics for all models
metrics_df = compute_all_metrics(models_dict, X_test, y_test)

print("✅ Metrics computed for all models\n")
print(metrics_df.to_string(index=False))
```

## Cell 4: Create Comparison Table

```python
# Create comprehensive comparison table
metrics_df, fig = create_comparison_table(models_dict, X_test, y_test, figsize=(14, 6))

print("✅ Comparison table created")
plt.show()

# Display as dataframe
print("\nMetrics Summary:")
print(metrics_df.to_string(index=False))
```

## Cell 5: Statistical Significance Testing

```python
# Perform statistical significance testing (McNemar's test)
sig_df = statistical_significance_testing(models_dict, X_test, y_test)

print("✅ Statistical significance testing completed\n")
print(sig_df.to_string(index=False))

# Summary
print("\n" + "="*60)
print("SIGNIFICANCE SUMMARY")
print("="*60)
for _, row in sig_df.iterrows():
    print(f"{row['Model 1']} vs {row['Model 2']}: {row['Significance']} (p={row['p-value']:.4f})")
```

## Cell 6: Visualize Statistical Significance

```python
# Visualize statistical significance testing results
fig = plot_statistical_significance(sig_df, figsize=(12, 6))

print("✅ Statistical significance visualization created")
plt.show()
```

## Cell 7: Generate Recommendations

```python
# Generate deployment recommendations
recommendations = generate_recommendations(metrics_df, sig_df)

print(recommendations)
```

## Cell 8: View Limitations

```python
# Display model limitations and constraints
limitations = generate_limitations()

print(limitations)
```

## Cell 9: View Future Improvements

```python
# Display future improvements and next steps
improvements = generate_future_improvements()

print(improvements)
```

## Cell 10: Full Report (All-in-One)

```python
# Generate complete results and reporting document
metrics_df, sig_df = print_full_report(models_dict, X_test, y_test)
```

## Cell 11: Export Results to CSV

```python
# Export metrics and significance results to CSV
metrics_df.to_csv('model_metrics.csv', index=False)
sig_df.to_csv('statistical_significance.csv', index=False)

print("✅ Results exported to CSV files")
print("   • model_metrics.csv")
print("   • statistical_significance.csv")
```

## Cell 12: Best Model Summary

```python
# Get best model information
best_model_idx = metrics_df['F1-Score'].idxmax()
best_model = metrics_df.iloc[best_model_idx]

print("="*60)
print("BEST MODEL FOR DEPLOYMENT")
print("="*60)
print(f"Model Name: {best_model['Model']}")
print(f"Accuracy:   {best_model['Accuracy']:.4f}")
print(f"Precision:  {best_model['Precision']:.4f}")
print(f"Recall:     {best_model['Recall']:.4f}")
print(f"F1-Score:   {best_model['F1-Score']:.4f}")
print(f"ROC-AUC:    {best_model['ROC-AUC']:.4f}")
print(f"Avg Conf:   {best_model['Avg Confidence']:.4f}")
print("="*60)
```

## Cell 13: Model Ranking

```python
# Display model ranking by F1-Score
print("="*60)
print("MODEL RANKING (by F1-Score)")
print("="*60)

ranked = metrics_df.sort_values('F1-Score', ascending=False)
for idx, (_, row) in enumerate(ranked.iterrows(), 1):
    print(f"{idx}. {row['Model']:<20} F1={row['F1-Score']:.4f}  Acc={row['Accuracy']:.4f}")

print("="*60)
```

## Cell 14: Pairwise Comparisons

```python
# Detailed pairwise comparisons
print("="*60)
print("PAIRWISE MODEL COMPARISONS")
print("="*60)

for _, row in sig_df.iterrows():
    m1 = row['Model 1']
    m2 = row['Model 2']
    
    m1_f1 = metrics_df[metrics_df['Model'] == m1]['F1-Score'].values[0]
    m2_f1 = metrics_df[metrics_df['Model'] == m2]['F1-Score'].values[0]
    
    winner = m1 if m1_f1 > m2_f1 else m2
    diff = abs(m1_f1 - m2_f1)
    
    print(f"\n{m1} vs {m2}")
    print(f"  F1 Difference: {diff:.4f}")
    print(f"  Winner: {winner}")
    print(f"  Significance: {row['Significance']} (p={row['p-value']:.4f})")
```

## Cell 15: Performance Gaps

```python
# Analyze performance gaps between models
print("="*60)
print("PERFORMANCE GAPS ANALYSIS")
print("="*60)

best_f1 = metrics_df['F1-Score'].max()
best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']

print(f"Best Model: {best_model_name} (F1={best_f1:.4f})\n")

for _, row in metrics_df.iterrows():
    if row['Model'] != best_model_name:
        gap = best_f1 - row['F1-Score']
        gap_pct = (gap / best_f1) * 100
        print(f"{row['Model']:<20} Gap: {gap:.4f} ({gap_pct:.2f}%)")
```

## Cell 16: Confidence Analysis

```python
# Analyze model confidence scores
print("="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)

for _, row in metrics_df.iterrows():
    print(f"{row['Model']:<20} Avg Confidence: {row['Avg Confidence']:.4f}")

avg_conf = metrics_df['Avg Confidence'].mean()
print(f"\nAverage Confidence: {avg_conf:.4f}")
```

## Cell 17: Complete Analysis Pipeline

```python
# Complete analysis pipeline - run all steps
print("Starting complete analysis pipeline...\n")

# Step 1: Metrics
print("Step 1: Computing metrics...")
metrics_df = compute_all_metrics(models_dict, X_test, y_test)
print(f"✅ Computed metrics for {len(models_dict)} models\n")

# Step 2: Comparison table
print("Step 2: Creating comparison table...")
metrics_df, fig = create_comparison_table(models_dict, X_test, y_test)
plt.show()
print("✅ Comparison table created\n")

# Step 3: Statistical testing
print("Step 3: Statistical significance testing...")
sig_df = statistical_significance_testing(models_dict, X_test, y_test)
print(f"✅ Tested {len(sig_df)} model pairs\n")

# Step 4: Significance visualization
print("Step 4: Visualizing significance...")
fig = plot_statistical_significance(sig_df)
plt.show()
print("✅ Significance visualization created\n")

# Step 5: Recommendations
print("Step 5: Generating recommendations...")
recommendations = generate_recommendations(metrics_df, sig_df)
print(recommendations)

# Step 6: Limitations
print("Step 6: Documenting limitations...")
limitations = generate_limitations()
print(limitations)

# Step 7: Future improvements
print("Step 7: Planning improvements...")
improvements = generate_future_improvements()
print(improvements)

print("✅ Complete analysis pipeline finished!")
```

## Cell 18: Save Full Report to File

```python
# Save complete report to text file
report, metrics_df, sig_df = generate_full_report(models_dict, X_test, y_test)

with open('helmnet_results_report.txt', 'w') as f:
    f.write(report)

print("✅ Full report saved to 'helmnet_results_report.txt'")
print(f"Report size: {len(report)} characters")
```

## Usage Instructions

1. **Copy cells in order** - Start with Cell 1 (Import)
2. **Run Cell 2** - Create models dictionary
3. **Run cells 3-9** - Individual analyses
4. **Or run Cell 10** - Full report (all-in-one)
5. **Run cells 11-18** - Additional analyses and exports

## Quick Start (Minimal)

If you just want the essentials:

```python
# Cell 1: Import
from results_reporting import *

# Cell 2: Models
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

# Cell 3: Full Report
metrics_df, sig_df = print_full_report(models_dict, X_test, y_test)
```

## Tips

- Run cells in order for best results
- Each cell is independent and can be run separately
- Modify figsize parameters for different display sizes
- Export results to CSV for further analysis
- Save report to file for documentation

## Next Steps

After running these cells:
1. Review the comparison table
2. Check statistical significance
3. Read deployment recommendations
4. Document limitations
5. Plan future improvements
6. Deploy recommended model
7. Set up monitoring
