# Performance Optimization - Complete Reference

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Memory Tracking](#memory-tracking)
5. [Training Time Tracking](#training-time-tracking)
6. [Batch Size Recommendations](#batch-size-recommendations)
7. [Inference Benchmarking](#inference-benchmarking)
8. [Visualizations](#visualizations)
9. [Reports](#reports)
10. [Keras Integration](#keras-integration)
11. [API Reference](#api-reference)
12. [Examples](#examples)
13. [Troubleshooting](#troubleshooting)

## Overview

The Performance Optimization module provides comprehensive tools for monitoring and optimizing HelmNet model performance:

- **Memory Usage Tracking** - Monitor CPU/GPU memory during training
- **Training Time Tracking** - Log execution times for each model
- **Batch Size Recommendations** - Find optimal batch size based on hardware
- **Inference Time Benchmarks** - Measure prediction speed and throughput

## Installation

### Step 1: Copy Module
```bash
cp v2_model_development_enhancements/performance_optimization.py ./
```

### Step 2: Import in Notebook
```python
from performance_optimization import *
```

### Step 3: Verify Installation
```python
# Check if module loads
memory_tracker = MemoryTracker()
print("✓ Performance optimization module loaded")
```

## Quick Start

### 5-Minute Setup

```python
from performance_optimization import *

# 1. Check hardware
memory_info = get_available_memory()
print(f"CPU: {memory_info['cpu_available_gb']} GB available")

# 2. Find optimal batch size
rec = recommend_batch_size(model, input_shape=(224, 224, 3))
print(f"Recommended batch size: {rec['recommended_batch_size']}")

# 3. Setup tracking
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

# 4. Train with tracking
model.fit(X_train, y_train, callbacks=[callback], epochs=20)

# 5. Benchmark inference
benchmark = InferenceBenchmark(model)
benchmark.benchmark_batch(X_test)

# 6. Generate report
report = generate_performance_report(model, X_test, memory_tracker, time_tracker, benchmark)
print_performance_summary(report)
```

## Memory Tracking

### MemoryTracker Class

Monitors CPU and GPU memory usage during training.

#### Constructor
```python
tracker = MemoryTracker()
```

#### Methods

##### get_memory_info()
Returns current memory usage.

```python
info = tracker.get_memory_info()
# Returns:
# {
#   'timestamp': '2024-03-29T10:30:45.123456',
#   'cpu_rss_mb': 2048.5,
#   'cpu_vms_mb': 3072.25,
#   'gpu_memory': {},
#   'cpu_percent': 45.2
# }
```

##### log_memory(epoch=None, batch=None)
Log memory at specific point.

```python
info = tracker.log_memory(epoch=1)
info = tracker.log_memory(epoch=1, batch=10)
```

**Parameters:**
- `epoch` (int, optional) - Epoch number
- `batch` (int, optional) - Batch number

**Returns:** Dictionary with memory info

##### get_peak_memory()
Get peak memory usage statistics.

```python
peak = tracker.get_peak_memory()
# Returns:
# {
#   'peak_cpu_rss_mb': 2048.5,
#   'peak_cpu_vms_mb': 3072.25,
#   'avg_cpu_rss_mb': 1856.75,
#   'avg_cpu_vms_mb': 2900.0,
#   'peak_cpu_percent': 85.5,
#   'avg_cpu_percent': 65.2
# }
```

**Returns:** Dictionary with peak memory statistics

##### get_memory_dataframe()
Get all memory logs as DataFrame.

```python
df = tracker.get_memory_dataframe()
# Columns: timestamp, cpu_rss_mb, cpu_vms_mb, gpu_memory, cpu_percent, epoch, batch
```

**Returns:** pandas DataFrame

### Usage Example

```python
# Track memory during training
tracker = MemoryTracker()

for epoch in range(10):
    tracker.log_memory(epoch=epoch)
    # ... training ...
    tracker.log_memory(epoch=epoch)

# Get statistics
peak = tracker.get_peak_memory()
print(f"Peak memory: {peak['peak_cpu_rss_mb']:.2f} MB")

# Visualize
df = tracker.get_memory_dataframe()
df.plot(x='timestamp', y='cpu_rss_mb')
```

## Training Time Tracking

### TrainingTimeTracker Class

Tracks execution time at epoch and batch levels.

#### Constructor
```python
timer = TrainingTimeTracker()
```

#### Methods

##### start_epoch(epoch)
Start timing an epoch.

```python
timer.start_epoch(0)
```

**Parameters:**
- `epoch` (int) - Epoch number

##### end_epoch(epoch)
End timing an epoch and return duration.

```python
duration = timer.end_epoch(0)  # Returns duration in seconds
```

**Parameters:**
- `epoch` (int) - Epoch number

**Returns:** Duration in seconds (float)

##### start_batch(batch)
Start timing a batch.

```python
timer.start_batch(0)
```

**Parameters:**
- `batch` (int) - Batch number

##### end_batch(batch)
End timing a batch and return duration.

```python
duration = timer.end_batch(0)  # Returns duration in seconds
```

**Parameters:**
- `batch` (int) - Batch number

**Returns:** Duration in seconds (float)

##### get_summary()
Get timing summary statistics.

```python
summary = timer.get_summary()
# Returns:
# {
#   'total_training_time_seconds': 2715.5,
#   'total_training_time_minutes': 45.26,
#   'avg_epoch_time_seconds': 2.72,
#   'min_epoch_time_seconds': 2.45,
#   'max_epoch_time_seconds': 3.12,
#   'avg_batch_time_ms': 12.5,
#   'min_batch_time_ms': 10.2,
#   'max_batch_time_ms': 15.8
# }
```

**Returns:** Dictionary with timing statistics

##### get_timings_dataframe()
Get timing data as DataFrames.

```python
epoch_df, batch_df = timer.get_timings_dataframe()
```

**Returns:** Tuple of (epoch_df, batch_df) pandas DataFrames

### Usage Example

```python
timer = TrainingTimeTracker()

for epoch in range(10):
    timer.start_epoch(epoch)
    
    for batch in range(100):
        timer.start_batch(batch)
        # ... training ...
        timer.end_batch(batch)
    
    timer.end_epoch(epoch)

# Get statistics
summary = timer.get_summary()
print(f"Total time: {summary['total_training_time_minutes']:.2f} minutes")
print(f"Avg per epoch: {summary['avg_epoch_time_seconds']:.2f} seconds")
```

## Batch Size Recommendations

### get_available_memory()

Get available GPU and CPU memory.

```python
memory_info = get_available_memory()
# Returns:
# {
#   'cpu_available_gb': 16.5,
#   'cpu_total_gb': 32.0,
#   'cpu_percent_used': 45.2,
#   'gpu_available': True,
#   'num_gpus': 1
# }
```

**Returns:** Dictionary with memory information

### recommend_batch_size()

Recommend batch size based on available hardware.

```python
rec = recommend_batch_size(
    model=model,
    input_shape=(224, 224, 3),
    memory_fraction=0.8,
    test_batch_sizes=[8, 16, 32, 64, 128, 256, 512]
)
```

**Parameters:**
- `model` (tf.keras.Model) - Keras model
- `input_shape` (tuple) - Input shape without batch dimension
- `memory_fraction` (float, default=0.8) - Fraction of available memory to use
- `test_batch_sizes` (list, default=[8, 16, 32, 64, 128, 256, 512]) - Batch sizes to test

**Returns:** Dictionary with recommendations

**Output:**
```python
{
  'memory_info': {...},
  'tested_batch_sizes': [8, 16, 32, 64, 128, 256],
  'successful_batch_sizes': [8, 16, 32, 64, 128],
  'failed_batch_sizes': [256],
  'recommended_batch_size': 128
}
```

### Usage Example

```python
# Find optimal batch size
rec = recommend_batch_size(model, input_shape=(224, 224, 3))

print(f"Tested: {rec['tested_batch_sizes']}")
print(f"Successful: {rec['successful_batch_sizes']}")
print(f"Recommended: {rec['recommended_batch_size']}")

# Use recommended batch size
batch_size = rec['recommended_batch_size']
model.fit(X_train, y_train, batch_size=batch_size)
```

## Inference Benchmarking

### InferenceBenchmark Class

Benchmark inference speed and throughput.

#### Constructor
```python
benchmark = InferenceBenchmark(model)
```

**Parameters:**
- `model` (tf.keras.Model) - Keras model to benchmark

#### Methods

##### benchmark_single_sample(X, num_runs=100)
Benchmark single sample inference.

```python
result = benchmark.benchmark_single_sample(X_test, num_runs=100)
# Returns:
# {
#   'type': 'single_sample',
#   'num_runs': 100,
#   'mean_time_ms': 15.23,
#   'std_time_ms': 2.15,
#   'min_time_ms': 12.5,
#   'max_time_ms': 22.3,
#   'median_time_ms': 14.8,
#   'throughput_samples_per_sec': 65.7
# }
```

**Parameters:**
- `X` (np.ndarray) - Input data
- `num_runs` (int, default=100) - Number of runs for averaging

**Returns:** Dictionary with benchmark results

##### benchmark_batch(X, batch_sizes=None, num_runs=10)
Benchmark batch inference at different batch sizes.

```python
results = benchmark.benchmark_batch(
    X_test,
    batch_sizes=[1, 8, 16, 32, 64, 128],
    num_runs=10
)
```

**Parameters:**
- `X` (np.ndarray) - Input data
- `batch_sizes` (list, default=[1, 8, 16, 32, 64, 128]) - Batch sizes to test
- `num_runs` (int, default=10) - Number of runs per batch size

**Returns:** List of dictionaries with benchmark results

##### get_results_dataframe()
Get all benchmark results as DataFrame.

```python
df = benchmark.get_results_dataframe()
# Columns: type, batch_size, num_runs, mean_time_ms, std_time_ms, 
#          min_time_ms, max_time_ms, median_time_ms, throughput_samples_per_sec
```

**Returns:** pandas DataFrame

### Usage Example

```python
# Benchmark inference
benchmark = InferenceBenchmark(model)

# Single sample
single = benchmark.benchmark_single_sample(X_test, num_runs=100)
print(f"Single sample: {single['mean_time_ms']:.2f} ms")

# Batch
batch_results = benchmark.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])
for result in batch_results:
    print(f"Batch {result['batch_size']}: {result['mean_time_ms']:.2f} ms")

# Get DataFrame
df = benchmark.get_results_dataframe()
print(df)
```

## Visualizations

### plot_memory_usage()

Plot memory usage over time.

```python
plot_memory_usage(memory_tracker, figsize=(14, 5))
```

**Parameters:**
- `memory_tracker` (MemoryTracker) - Memory tracker object
- `figsize` (tuple, default=(14, 5)) - Figure size

**Creates:**
- Subplot 1: CPU Memory (RSS and VMS) over time
- Subplot 2: CPU Utilization percentage over time

### plot_training_times()

Plot training times.

```python
plot_training_times(time_tracker, figsize=(14, 5))
```

**Parameters:**
- `time_tracker` (TrainingTimeTracker) - Time tracker object
- `figsize` (tuple, default=(14, 5)) - Figure size

**Creates:**
- Subplot 1: Time per epoch (bar chart)
- Subplot 2: Time per batch (scatter plot)

### plot_inference_benchmark()

Plot inference benchmark results.

```python
plot_inference_benchmark(benchmark, figsize=(14, 5))
```

**Parameters:**
- `benchmark` (InferenceBenchmark) - Benchmark object
- `figsize` (tuple, default=(14, 5)) - Figure size

**Creates:**
- Subplot 1: Inference time vs batch size (with error bands)
- Subplot 2: Throughput vs batch size

## Reports

### generate_performance_report()

Generate comprehensive performance report.

```python
report = generate_performance_report(
    model=model,
    X_test=X_test,
    memory_tracker=memory_tracker,
    time_tracker=time_tracker,
    benchmark=benchmark,
    model_name="Model 4"
)
```

**Parameters:**
- `model` (tf.keras.Model) - Keras model
- `X_test` (np.ndarray) - Test data
- `memory_tracker` (MemoryTracker) - Memory tracker
- `time_tracker` (TrainingTimeTracker) - Time tracker
- `benchmark` (InferenceBenchmark) - Benchmark object
- `model_name` (str, default="Model") - Model name for report

**Returns:** Dictionary with complete performance report

### print_performance_summary()

Print formatted performance summary.

```python
print_performance_summary(report)
```

**Parameters:**
- `report` (dict) - Report dictionary from generate_performance_report()

**Output:**
```
======================================================================
PERFORMANCE REPORT: Model 4
======================================================================

Timestamp: 2024-03-29T10:30:45.123456

📊 MODEL INFORMATION
  Total Parameters: 1,234,567
  Trainable Parameters: 1,200,000
  Non-trainable Parameters: 34,567

💾 MEMORY USAGE
  Peak CPU RSS: 2048.50 MB
  Peak CPU VMS: 3072.25 MB
  Avg CPU RSS: 1856.75 MB
  Peak CPU Usage: 85.50%

⏱️  TRAINING TIME
  Total Time: 45.26 minutes
  Avg per Epoch: 2.72 seconds
  Avg per Batch: 12.50 ms

⚡ INFERENCE PERFORMANCE
  Batch Size 32: 156.89ms (±5.43ms)
    Throughput: 203.9 samples/sec

======================================================================
```

## Keras Integration

### PerformanceTrackingCallback

Keras callback for automatic performance tracking.

```python
callback = PerformanceTrackingCallback(
    memory_tracker=memory_tracker,
    time_tracker=time_tracker
)

model.fit(X_train, y_train, callbacks=[callback], epochs=20)
```

**Parameters:**
- `memory_tracker` (MemoryTracker, optional) - Memory tracker
- `time_tracker` (TrainingTimeTracker, optional) - Time tracker

**Hooks:**
- `on_epoch_begin()` - Start epoch timer, log memory
- `on_epoch_end()` - End epoch timer, log memory
- `on_train_batch_begin()` - Start batch timer
- `on_train_batch_end()` - End batch timer

### Usage Example

```python
# Setup tracking
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

# Train with tracking
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[callback],
    epochs=20,
    batch_size=32
)

# Access tracked data
print(time_tracker.get_summary())
print(memory_tracker.get_peak_memory())
```

## API Reference

### Classes

| Class | Purpose |
|-------|---------|
| MemoryTracker | Track CPU/GPU memory usage |
| TrainingTimeTracker | Track training execution times |
| InferenceBenchmark | Benchmark inference speed |
| PerformanceTrackingCallback | Keras callback for auto-tracking |

### Functions

| Function | Purpose |
|----------|---------|
| get_available_memory() | Get available hardware memory |
| recommend_batch_size() | Find optimal batch size |
| generate_performance_report() | Generate comprehensive report |
| print_performance_summary() | Print formatted summary |
| plot_memory_usage() | Visualize memory usage |
| plot_training_times() | Visualize training times |
| plot_inference_benchmark() | Visualize inference performance |

## Examples

### Example 1: Quick Batch Size Check

```python
from performance_optimization import recommend_batch_size

rec = recommend_batch_size(model, input_shape=(224, 224, 3))
print(f"Recommended batch size: {rec['recommended_batch_size']}")
```

### Example 2: Track Training Performance

```python
from performance_optimization import *

memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

model.fit(X_train, y_train, callbacks=[callback], epochs=20)

plot_memory_usage(memory_tracker)
plot_training_times(time_tracker)
```

### Example 3: Benchmark Inference

```python
from performance_optimization import InferenceBenchmark, plot_inference_benchmark

benchmark = InferenceBenchmark(model)
benchmark.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])
plot_inference_benchmark(benchmark)
```

### Example 4: Compare Models

```python
from performance_optimization import InferenceBenchmark
import pandas as pd

models = {'Model 1': model1, 'Model 2': model2, 'Model 3': model3}
results = []

for name, model in models.items():
    bench = InferenceBenchmark(model)
    bench.benchmark_batch(X_test, batch_sizes=[32])
    
    df = bench.get_results_dataframe()
    batch_df = df[df['type'] == 'batch']
    
    if not batch_df.empty:
        results.append({
            'Model': name,
            'Latency (ms)': batch_df.iloc[0]['mean_time_ms'],
            'Throughput (samples/sec)': batch_df.iloc[0]['throughput_samples_per_sec']
        })

comparison = pd.DataFrame(results)
print(comparison)
```

### Example 5: Full Analysis Pipeline

```python
from performance_optimization import *

# 1. Check hardware
memory_info = get_available_memory()
print(f"Available: {memory_info['cpu_available_gb']} GB")

# 2. Find batch size
rec = recommend_batch_size(model, input_shape=(224, 224, 3))
batch_size = rec['recommended_batch_size']

# 3. Setup tracking
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

# 4. Train
model.fit(X_train, y_train, callbacks=[callback], batch_size=batch_size, epochs=20)

# 5. Benchmark
benchmark = InferenceBenchmark(model)
benchmark.benchmark_batch(X_test)

# 6. Report
report = generate_performance_report(model, X_test, memory_tracker, time_tracker, benchmark)
print_performance_summary(report)

# 7. Visualize
plot_memory_usage(memory_tracker)
plot_training_times(time_tracker)
plot_inference_benchmark(benchmark)
```

## Troubleshooting

### GPU Memory Not Detected

**Problem:** GPU memory tracking shows empty dictionary

**Solution:**
- Check TensorFlow GPU setup: `tf.config.list_physical_devices('GPU')`
- Verify CUDA/cuDNN installation
- GPU memory tracking may not work on all systems

### Batch Size Recommendation Fails

**Problem:** recommend_batch_size() raises error

**Solution:**
- Reduce test_batch_sizes range
- Check input_shape matches your data
- Ensure model is properly built
- Try smaller batch sizes first

### Inference Benchmark is Slow

**Problem:** Benchmarking takes too long

**Solution:**
- Reduce num_runs for faster testing
- Use smaller batch sizes for quick check
- Ensure model is on GPU if available
- Use smaller test dataset

### Memory Tracking Shows No Data

**Problem:** get_memory_dataframe() returns empty DataFrame

**Solution:**
- Call log_memory() at least once
- Check memory_tracker is passed to callback
- Verify callback is added to model.fit()

### Out of Memory During Benchmarking

**Problem:** OOM error during batch size recommendation

**Solution:**
- Reduce test_batch_sizes range
- Use smaller input_shape
- Close other applications
- Use GPU if available

## Performance Tips

1. **Memory Tracking**: Log at regular intervals (every epoch)
2. **Batch Size**: Test with actual input shape for accurate recommendations
3. **Inference**: Run multiple times (num_runs) for stable averages
4. **Benchmarking**: Warm up model before measuring (first prediction is slower)
5. **Reports**: Generate after training completes for full summary

## Next Steps

1. Copy `performance_optimization.py` to notebook directory
2. Import functions in notebook
3. Add tracking to training loop
4. Benchmark inference on test data
5. Generate performance report
6. Visualize results
7. Use insights for optimization

See PERFORMANCE_QUICK_START.md for quick reference and PERFORMANCE_NOTEBOOK_CELLS.md for copy-paste examples.
