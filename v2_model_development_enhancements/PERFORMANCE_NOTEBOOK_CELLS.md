# Performance Optimization - Notebook Cells

Copy and paste these cells into your Jupyter notebook.

## Cell 1: Setup and Imports

```python
# Import performance optimization module
from performance_optimization import (
    MemoryTracker,
    TrainingTimeTracker,
    InferenceBenchmark,
    PerformanceTrackingCallback,
    recommend_batch_size,
    get_available_memory,
    generate_performance_report,
    print_performance_summary,
    plot_memory_usage,
    plot_training_times,
    plot_inference_benchmark
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✓ Performance optimization module loaded")
```

## Cell 2: Check Available Hardware

```python
# Check available memory and hardware
memory_info = get_available_memory()

print("📊 AVAILABLE HARDWARE")
print(f"CPU Available: {memory_info['cpu_available_gb']:.2f} GB")
print(f"CPU Total: {memory_info['cpu_total_gb']:.2f} GB")
print(f"CPU Usage: {memory_info['cpu_percent_used']:.1f}%")
print(f"GPU Available: {memory_info['gpu_available']}")
if memory_info['gpu_available']:
    print(f"Number of GPUs: {memory_info['num_gpus']}")
```

## Cell 3: Find Optimal Batch Size

```python
# Recommend batch size for your model
# Replace with your actual input shape
input_shape = (224, 224, 3)  # Example: image shape

recommendations = recommend_batch_size(
    model=best_model_4,  # Use your trained model
    input_shape=input_shape,
    test_batch_sizes=[8, 16, 32, 64, 128, 256, 512]
)

print("🎯 BATCH SIZE RECOMMENDATIONS")
print(f"Tested batch sizes: {recommendations['tested_batch_sizes']}")
print(f"Successful batch sizes: {recommendations['successful_batch_sizes']}")
print(f"Failed batch sizes: {recommendations['failed_batch_sizes']}")
print(f"\n✓ Recommended batch size: {recommendations['recommended_batch_size']}")
```

## Cell 4: Setup Performance Tracking

```python
# Initialize trackers
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()

# Create callback for automatic tracking
performance_callback = PerformanceTrackingCallback(
    memory_tracker=memory_tracker,
    time_tracker=time_tracker
)

print("✓ Performance trackers initialized")
print("✓ Ready to train with automatic tracking")
```

## Cell 5: Train with Performance Tracking

```python
# Train model with performance tracking
# This example assumes you have X_train, y_train, X_val, y_val

history = best_model_4.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,  # Use recommended batch size
    callbacks=[performance_callback],
    verbose=1
)

print("\n✓ Training completed with performance tracking")
```

## Cell 6: View Training Time Summary

```python
# Get training time summary
time_summary = time_tracker.get_summary()

print("⏱️  TRAINING TIME SUMMARY")
print(f"Total training time: {time_summary.get('total_training_time_minutes', 0):.2f} minutes")
print(f"Average per epoch: {time_summary.get('avg_epoch_time_seconds', 0):.2f} seconds")
print(f"Min epoch time: {time_summary.get('min_epoch_time_seconds', 0):.2f} seconds")
print(f"Max epoch time: {time_summary.get('max_epoch_time_seconds', 0):.2f} seconds")
print(f"Average per batch: {time_summary.get('avg_batch_time_ms', 0):.2f} ms")

# Get detailed timing data
epoch_df, batch_df = time_tracker.get_timings_dataframe()
print("\nEpoch timings:")
print(epoch_df.head(10))
```

## Cell 7: View Memory Usage Summary

```python
# Get memory usage summary
memory_summary = memory_tracker.get_peak_memory()

print("💾 MEMORY USAGE SUMMARY")
print(f"Peak CPU RSS: {memory_summary.get('peak_cpu_rss_mb', 0):.2f} MB")
print(f"Peak CPU VMS: {memory_summary.get('peak_cpu_vms_mb', 0):.2f} MB")
print(f"Average CPU RSS: {memory_summary.get('avg_cpu_rss_mb', 0):.2f} MB")
print(f"Average CPU VMS: {memory_summary.get('avg_cpu_vms_mb', 0):.2f} MB")
print(f"Peak CPU usage: {memory_summary.get('peak_cpu_percent', 0):.2f}%")
print(f"Average CPU usage: {memory_summary.get('avg_cpu_percent', 0):.2f}%")

# Get detailed memory data
memory_df = memory_tracker.get_memory_dataframe()
print("\nMemory log (first 10 entries):")
print(memory_df.head(10))
```

## Cell 8: Plot Memory Usage

```python
# Visualize memory usage during training
plot_memory_usage(memory_tracker, figsize=(14, 5))
```

## Cell 9: Plot Training Times

```python
# Visualize training times
plot_training_times(time_tracker, figsize=(14, 5))
```

## Cell 10: Benchmark Single Sample Inference

```python
# Create benchmark object
benchmark = InferenceBenchmark(best_model_4)

# Benchmark single sample inference
single_result = benchmark.benchmark_single_sample(X_test, num_runs=100)

print("⚡ SINGLE SAMPLE INFERENCE")
print(f"Mean time: {single_result['mean_time_ms']:.2f} ms")
print(f"Std dev: {single_result['std_time_ms']:.2f} ms")
print(f"Min time: {single_result['min_time_ms']:.2f} ms")
print(f"Max time: {single_result['max_time_ms']:.2f} ms")
print(f"Median time: {single_result['median_time_ms']:.2f} ms")
print(f"Throughput: {single_result['throughput_samples_per_sec']:.1f} samples/sec")
```

## Cell 11: Benchmark Batch Inference

```python
# Benchmark batch inference at different sizes
batch_results = benchmark.benchmark_batch(
    X_test,
    batch_sizes=[1, 8, 16, 32, 64, 128],
    num_runs=10
)

print("⚡ BATCH INFERENCE BENCHMARKS")
for result in batch_results:
    print(f"\nBatch size {result['batch_size']}:")
    print(f"  Mean time: {result['mean_time_ms']:.2f} ms (±{result['std_time_ms']:.2f} ms)")
    print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
```

## Cell 12: Plot Inference Benchmarks

```python
# Visualize inference performance
plot_inference_benchmark(benchmark, figsize=(14, 5))
```

## Cell 13: Get Benchmark Results as DataFrame

```python
# Get all benchmark results as DataFrame
benchmark_df = benchmark.get_results_dataframe()

print("📊 BENCHMARK RESULTS")
print(benchmark_df)

# Summary statistics
print("\nSummary by batch size:")
batch_summary = benchmark_df[benchmark_df['type'] == 'batch'].groupby('batch_size').agg({
    'mean_time_ms': 'mean',
    'throughput_samples_per_sec': 'mean'
})
print(batch_summary)
```

## Cell 14: Compare Multiple Models

```python
# Compare performance across all models
models_dict = {
    'Model 1': best_model_1,
    'Model 2': best_model_2,
    'Model 3': best_model_3,
    'Model 4': best_model_4
}

comparison_results = []

for model_name, model in models_dict.items():
    bench = InferenceBenchmark(model)
    bench.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])
    
    results_df = bench.get_results_dataframe()
    batch_df = results_df[results_df['type'] == 'batch']
    
    if not batch_df.empty:
        avg_throughput = batch_df['throughput_samples_per_sec'].mean()
        avg_latency = batch_df['mean_time_ms'].mean()
        
        comparison_results.append({
            'Model': model_name,
            'Avg Latency (ms)': avg_latency,
            'Avg Throughput (samples/sec)': avg_throughput,
            'Params': model.count_params()
        })

comparison_df = pd.DataFrame(comparison_results)
print("📊 MODEL PERFORMANCE COMPARISON")
print(comparison_df)
```

## Cell 15: Generate Full Performance Report

```python
# Generate comprehensive performance report
report = generate_performance_report(
    model=best_model_4,
    X_test=X_test,
    memory_tracker=memory_tracker,
    time_tracker=time_tracker,
    benchmark=benchmark,
    model_name="Best Model (Model 4)"
)

# Print formatted summary
print_performance_summary(report)
```

## Cell 16: Export Report to JSON

```python
import json
from datetime import datetime

# Convert report to JSON-serializable format
report_json = {
    'model_name': report['model_name'],
    'timestamp': report['timestamp'],
    'model_info': report['model_info'],
    'memory': report['memory'],
    'training_time': report['training_time'],
    'inference': report['inference']
}

# Save to file
filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(report_json, f, indent=2)

print(f"✓ Report saved to {filename}")
```

## Cell 17: Create Performance Summary Table

```python
# Create summary table for all models
summary_data = []

for model_name, model in models_dict.items():
    # Get model info
    total_params = model.count_params()
    
    # Benchmark inference
    bench = InferenceBenchmark(model)
    bench.benchmark_batch(X_test, batch_sizes=[32])
    
    results_df = bench.get_results_dataframe()
    batch_df = results_df[results_df['type'] == 'batch']
    
    if not batch_df.empty:
        latency = batch_df.iloc[0]['mean_time_ms']
        throughput = batch_df.iloc[0]['throughput_samples_per_sec']
    else:
        latency = 0
        throughput = 0
    
    summary_data.append({
        'Model': model_name,
        'Parameters': f"{total_params:,}",
        'Latency (ms)': f"{latency:.2f}",
        'Throughput (samples/sec)': f"{throughput:.1f}"
    })

summary_table = pd.DataFrame(summary_data)
print("📊 PERFORMANCE SUMMARY TABLE")
print(summary_table.to_string(index=False))
```

## Cell 18: Recommendations and Insights

```python
# Generate performance insights
print("🎯 PERFORMANCE INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Memory insights
memory_summary = memory_tracker.get_peak_memory()
peak_memory = memory_summary.get('peak_cpu_rss_mb', 0)
print(f"\n💾 Memory:")
if peak_memory > 4000:
    print(f"  ⚠️  High memory usage ({peak_memory:.0f} MB)")
    print(f"     Consider: Reduce batch size, use gradient checkpointing")
else:
    print(f"  ✓ Memory usage is reasonable ({peak_memory:.0f} MB)")

# Training time insights
time_summary = time_tracker.get_summary()
total_time = time_summary.get('total_training_time_minutes', 0)
print(f"\n⏱️  Training Time:")
print(f"  Total: {total_time:.2f} minutes")
if total_time > 60:
    print(f"  ⚠️  Long training time")
    print(f"     Consider: Use learning rate scheduling, early stopping")
else:
    print(f"  ✓ Training time is acceptable")

# Inference insights
batch_df = benchmark.get_results_dataframe()
batch_df = batch_df[batch_df['type'] == 'batch']
if not batch_df.empty:
    best_throughput = batch_df['throughput_samples_per_sec'].max()
    print(f"\n⚡ Inference:")
    print(f"  Best throughput: {best_throughput:.1f} samples/sec")
    if best_throughput > 100:
        print(f"  ✓ Good inference speed for deployment")
    else:
        print(f"  ⚠️  Consider model optimization for production")

print("\n" + "="*60)
```

## Cell 19: Save All Visualizations

```python
# Save all visualizations to files
import os

# Create output directory
os.makedirs('performance_plots', exist_ok=True)

# Memory usage
fig = plt.figure(figsize=(14, 5))
plot_memory_usage(memory_tracker)
plt.savefig('performance_plots/memory_usage.png', dpi=150, bbox_inches='tight')
plt.close()

# Training times
fig = plt.figure(figsize=(14, 5))
plot_training_times(time_tracker)
plt.savefig('performance_plots/training_times.png', dpi=150, bbox_inches='tight')
plt.close()

# Inference benchmarks
fig = plt.figure(figsize=(14, 5))
plot_inference_benchmark(benchmark)
plt.savefig('performance_plots/inference_benchmark.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ All visualizations saved to 'performance_plots/' directory")
```

## Cell 20: Complete Performance Analysis Pipeline

```python
# Complete pipeline: Setup → Train → Benchmark → Report

print("🚀 COMPLETE PERFORMANCE ANALYSIS PIPELINE")
print("="*60)

# 1. Check hardware
print("\n1️⃣  Checking available hardware...")
memory_info = get_available_memory()
print(f"   CPU: {memory_info['cpu_available_gb']:.2f} GB available")
print(f"   GPU: {'Yes' if memory_info['gpu_available'] else 'No'}")

# 2. Find optimal batch size
print("\n2️⃣  Finding optimal batch size...")
rec = recommend_batch_size(best_model_4, input_shape=(224, 224, 3))
optimal_batch = rec['recommended_batch_size']
print(f"   Recommended batch size: {optimal_batch}")

# 3. Setup tracking
print("\n3️⃣  Setting up performance tracking...")
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)
print("   ✓ Trackers ready")

# 4. Benchmark inference
print("\n4️⃣  Benchmarking inference...")
benchmark = InferenceBenchmark(best_model_4)
benchmark.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])
print("   ✓ Inference benchmarked")

# 5. Generate report
print("\n5️⃣  Generating performance report...")
report = generate_performance_report(
    best_model_4, X_test, memory_tracker, time_tracker, benchmark, "Best Model"
)
print_performance_summary(report)

print("\n" + "="*60)
print("✓ Performance analysis complete!")
```

## Tips for Using These Cells

1. **Run in order** - Each cell builds on previous ones
2. **Modify as needed** - Adjust batch sizes, epochs, etc. for your data
3. **Save outputs** - Use Cell 19 to save visualizations
4. **Compare models** - Use Cell 14 to compare all models
5. **Generate reports** - Use Cell 15 for comprehensive summary

## Common Modifications

### Change input shape
```python
input_shape = (128, 128, 3)  # For smaller images
input_shape = (512, 512, 3)  # For larger images
input_shape = (100,)         # For 1D data
```

### Change batch sizes to test
```python
batch_sizes = [4, 8, 16, 32]  # Smaller range
batch_sizes = [64, 128, 256, 512, 1024]  # Larger range
```

### Change number of benchmark runs
```python
benchmark.benchmark_single_sample(X_test, num_runs=50)  # Faster
benchmark.benchmark_single_sample(X_test, num_runs=500)  # More accurate
```

### Change figure sizes
```python
plot_memory_usage(memory_tracker, figsize=(16, 6))  # Larger
plot_memory_usage(memory_tracker, figsize=(10, 4))  # Smaller
```
