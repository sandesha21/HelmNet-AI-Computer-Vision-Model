# Performance Optimization - Quick Start

## One-Liners

### Memory Tracking
```python
from performance_optimization import MemoryTracker

tracker = MemoryTracker()
info = tracker.log_memory(epoch=1)  # Log at any point
peak = tracker.get_peak_memory()    # Get peak usage
df = tracker.get_memory_dataframe() # Get all logs
```

### Training Time Tracking
```python
from performance_optimization import TrainingTimeTracker

timer = TrainingTimeTracker()
timer.start_epoch(0)
# ... training ...
duration = timer.end_epoch(0)
summary = timer.get_summary()
```

### Batch Size Recommendations
```python
from performance_optimization import recommend_batch_size

rec = recommend_batch_size(model, input_shape=(224, 224, 3))
print(f"Recommended batch size: {rec['recommended_batch_size']}")
```

### Inference Benchmarking
```python
from performance_optimization import InferenceBenchmark

bench = InferenceBenchmark(model)
single = bench.benchmark_single_sample(X_test)
batch_results = bench.benchmark_batch(X_test, batch_sizes=[8, 16, 32, 64])
df = bench.get_results_dataframe()
```

### Automatic Tracking with Keras
```python
from performance_optimization import PerformanceTrackingCallback, MemoryTracker, TrainingTimeTracker

memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

model.fit(X_train, y_train, callbacks=[callback], epochs=10)
```

### Visualizations
```python
from performance_optimization import plot_memory_usage, plot_training_times, plot_inference_benchmark

plot_memory_usage(memory_tracker)
plot_training_times(time_tracker)
plot_inference_benchmark(benchmark)
```

### Full Report
```python
from performance_optimization import generate_performance_report, print_performance_summary

report = generate_performance_report(model, X_test, memory_tracker, time_tracker, benchmark, "Model 4")
print_performance_summary(report)
```

## Common Tasks

### Track Memory During Training
```python
memory_tracker = MemoryTracker()
callback = PerformanceTrackingCallback(memory_tracker=memory_tracker)
model.fit(X_train, y_train, callbacks=[callback], epochs=10)
plot_memory_usage(memory_tracker)
```

### Find Optimal Batch Size
```python
rec = recommend_batch_size(model, input_shape=(224, 224, 3))
print(f"Successful batch sizes: {rec['successful_batch_sizes']}")
print(f"Recommended: {rec['recommended_batch_size']}")
```

### Benchmark Inference Speed
```python
bench = InferenceBenchmark(model)
bench.benchmark_single_sample(X_test, num_runs=100)
bench.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])
plot_inference_benchmark(bench)
```

### Compare Models Performance
```python
models = {'Model 1': model1, 'Model 2': model2, 'Model 3': model3}
for name, model in models.items():
    bench = InferenceBenchmark(model)
    bench.benchmark_batch(X_test)
    print(f"\n{name}:")
    print(bench.get_results_dataframe())
```

### Get Complete Performance Report
```python
report = generate_performance_report(
    model, X_test, 
    memory_tracker, time_tracker, benchmark,
    model_name="Best Model"
)
print_performance_summary(report)
```

## What Each Component Does

| Component | Purpose | Output |
|-----------|---------|--------|
| MemoryTracker | Monitor CPU/GPU memory | Peak/avg memory, DataFrame |
| TrainingTimeTracker | Track training duration | Per-epoch/batch times, summary |
| recommend_batch_size | Find optimal batch size | Recommended size, tested sizes |
| InferenceBenchmark | Measure inference speed | Latency, throughput, DataFrame |
| PerformanceTrackingCallback | Auto-track during training | Integrated with Keras |
| Visualizations | Plot performance metrics | Memory, time, inference charts |

## Integration with Training

```python
from performance_optimization import (
    MemoryTracker, TrainingTimeTracker, 
    PerformanceTrackingCallback, InferenceBenchmark,
    generate_performance_report, print_performance_summary,
    plot_memory_usage, plot_training_times, plot_inference_benchmark
)

# Setup tracking
memory_tracker = MemoryTracker()
time_tracker = TrainingTimeTracker()
callback = PerformanceTrackingCallback(memory_tracker, time_tracker)

# Train with tracking
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[callback],
    epochs=20,
    batch_size=32
)

# Benchmark inference
benchmark = InferenceBenchmark(model)
benchmark.benchmark_batch(X_test, batch_sizes=[1, 8, 16, 32, 64])

# Generate report
report = generate_performance_report(
    model, X_test, memory_tracker, time_tracker, benchmark, "Model 4"
)
print_performance_summary(report)

# Visualize
plot_memory_usage(memory_tracker)
plot_training_times(time_tracker)
plot_inference_benchmark(benchmark)
```

## Output Examples

### Memory Summary
```
Peak CPU RSS: 2048.50 MB
Peak CPU VMS: 3072.25 MB
Avg CPU RSS: 1856.75 MB
Peak CPU Usage: 85.50%
```

### Training Time Summary
```
Total Time: 45.25 minutes
Avg per Epoch: 2.26 seconds
Avg per Batch: 12.50 ms
```

### Inference Benchmark
```
Batch Size 1: 15.23ms (±2.15ms)
  Throughput: 65.7 samples/sec
Batch Size 8: 45.67ms (±3.21ms)
  Throughput: 175.1 samples/sec
Batch Size 32: 156.89ms (±5.43ms)
  Throughput: 203.9 samples/sec
```

## Tips

- **Memory Tracking**: Log at regular intervals (every epoch or every N batches)
- **Batch Size**: Test with your actual input shape for accurate recommendations
- **Inference**: Run multiple times (num_runs) for stable averages
- **Benchmarking**: Warm up the model before measuring (first prediction is slower)
- **Reports**: Generate after training completes for full summary

## Troubleshooting

**GPU memory not detected**
- Check TensorFlow GPU setup
- Verify CUDA/cuDNN installation
- GPU memory tracking may not work on all systems

**Batch size recommendation fails**
- Reduce test_batch_sizes range
- Check input_shape matches your data
- Ensure model is properly built

**Inference benchmark is slow**
- Reduce num_runs for faster testing
- Use smaller batch sizes for quick check
- Ensure model is on GPU if available

## Next Steps

1. Copy `performance_optimization.py` to notebook directory
2. Import functions in notebook
3. Add tracking to your training loop
4. Benchmark inference on test data
5. Generate performance report
6. Visualize results
7. Use insights for optimization

See PERFORMANCE_IMPLEMENTATION_SUMMARY.md for detailed examples.
