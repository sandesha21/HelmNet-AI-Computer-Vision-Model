"""
Performance Optimization Module for HelmNet v2
Tracks memory usage, training time, batch size recommendations, and inference benchmarks
"""

import os
import time
import psutil
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# ============================================================================
# 1. MEMORY TRACKING
# ============================================================================

class MemoryTracker:
    """Track GPU and CPU memory usage during training"""
    
    def __init__(self):
        self.memory_log = []
        self.start_time = None
        self.process = psutil.Process(os.getpid())
    
    def get_memory_info(self) -> Dict:
        """Get current memory usage"""
        # CPU memory
        cpu_memory = self.process.memory_info()
        cpu_rss_mb = cpu_memory.rss / 1024 / 1024  # Resident Set Size in MB
        cpu_vms_mb = cpu_memory.vms / 1024 / 1024  # Virtual Memory Size in MB
        
        # GPU memory
        gpu_memory = {}
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_memory[f'GPU_{i}'] = tf.config.experimental.get_memory_info('GPU:0')
        except:
            pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_rss_mb': cpu_rss_mb,
            'cpu_vms_mb': cpu_vms_mb,
            'gpu_memory': gpu_memory,
            'cpu_percent': self.process.cpu_percent(interval=0.1)
        }
    
    def log_memory(self, epoch: int = None, batch: int = None):
        """Log memory at specific point"""
        info = self.get_memory_info()
        info['epoch'] = epoch
        info['batch'] = batch
        self.memory_log.append(info)
        return info
    
    def get_peak_memory(self) -> Dict:
        """Get peak memory usage"""
        if not self.memory_log:
            return {}
        
        df = pd.DataFrame(self.memory_log)
        return {
            'peak_cpu_rss_mb': df['cpu_rss_mb'].max(),
            'peak_cpu_vms_mb': df['cpu_vms_mb'].max(),
            'avg_cpu_rss_mb': df['cpu_rss_mb'].mean(),
            'avg_cpu_vms_mb': df['cpu_vms_mb'].mean(),
            'peak_cpu_percent': df['cpu_percent'].max(),
            'avg_cpu_percent': df['cpu_percent'].mean()
        }
    
    def get_memory_dataframe(self) -> pd.DataFrame:
        """Get memory log as DataFrame"""
        return pd.DataFrame(self.memory_log)


# ============================================================================
# 2. TRAINING TIME TRACKING
# ============================================================================

class TrainingTimeTracker:
    """Track training execution times"""
    
    def __init__(self):
        self.timings = {}
        self.epoch_times = []
        self.batch_times = []
    
    def start_epoch(self, epoch: int):
        """Start timing an epoch"""
        self.timings[f'epoch_{epoch}_start'] = time.time()
    
    def end_epoch(self, epoch: int) -> float:
        """End timing an epoch and return duration"""
        start = self.timings.get(f'epoch_{epoch}_start')
        if start is None:
            return 0
        
        duration = time.time() - start
        self.epoch_times.append({
            'epoch': epoch,
            'duration_seconds': duration,
            'duration_minutes': duration / 60
        })
        return duration
    
    def start_batch(self, batch: int):
        """Start timing a batch"""
        self.timings[f'batch_{batch}_start'] = time.time()
    
    def end_batch(self, batch: int) -> float:
        """End timing a batch and return duration"""
        start = self.timings.get(f'batch_{batch}_start')
        if start is None:
            return 0
        
        duration = time.time() - start
        self.batch_times.append({
            'batch': batch,
            'duration_seconds': duration,
            'duration_ms': duration * 1000
        })
        return duration
    
    def get_summary(self) -> Dict:
        """Get timing summary"""
        epoch_df = pd.DataFrame(self.epoch_times)
        batch_df = pd.DataFrame(self.batch_times)
        
        summary = {}
        
        if not epoch_df.empty:
            summary['total_training_time_seconds'] = epoch_df['duration_seconds'].sum()
            summary['total_training_time_minutes'] = epoch_df['duration_seconds'].sum() / 60
            summary['avg_epoch_time_seconds'] = epoch_df['duration_seconds'].mean()
            summary['min_epoch_time_seconds'] = epoch_df['duration_seconds'].min()
            summary['max_epoch_time_seconds'] = epoch_df['duration_seconds'].max()
        
        if not batch_df.empty:
            summary['avg_batch_time_ms'] = batch_df['duration_ms'].mean()
            summary['min_batch_time_ms'] = batch_df['duration_ms'].min()
            summary['max_batch_time_ms'] = batch_df['duration_ms'].max()
        
        return summary
    
    def get_timings_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get timing data as DataFrames"""
        return pd.DataFrame(self.epoch_times), pd.DataFrame(self.batch_times)


# ============================================================================
# 3. BATCH SIZE RECOMMENDATIONS
# ============================================================================

def get_available_memory() -> Dict:
    """Get available GPU and CPU memory"""
    memory_info = {}
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_available_gb'] = cpu_memory.available / 1024 / 1024 / 1024
    memory_info['cpu_total_gb'] = cpu_memory.total / 1024 / 1024 / 1024
    memory_info['cpu_percent_used'] = cpu_memory.percent
    
    # GPU memory
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            memory_info['gpu_available'] = True
            memory_info['num_gpus'] = len(gpus)
        else:
            memory_info['gpu_available'] = False
            memory_info['num_gpus'] = 0
    except:
        memory_info['gpu_available'] = False
        memory_info['num_gpus'] = 0
    
    return memory_info


def recommend_batch_size(
    model: tf.keras.Model,
    input_shape: Tuple,
    memory_fraction: float = 0.8,
    test_batch_sizes: List[int] = None
) -> Dict:
    """
    Recommend batch size based on available hardware
    
    Args:
        model: Keras model
        input_shape: Input shape (without batch dimension)
        memory_fraction: Fraction of available memory to use (0-1)
        test_batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with recommendations
    """
    if test_batch_sizes is None:
        test_batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    
    memory_info = get_available_memory()
    recommendations = {
        'memory_info': memory_info,
        'tested_batch_sizes': [],
        'successful_batch_sizes': [],
        'failed_batch_sizes': [],
        'recommended_batch_size': None
    }
    
    # Test each batch size
    for batch_size in test_batch_sizes:
        try:
            # Create dummy input
            dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)
            
            # Try prediction
            _ = model.predict(dummy_input, verbose=0)
            
            recommendations['tested_batch_sizes'].append(batch_size)
            recommendations['successful_batch_sizes'].append(batch_size)
            
        except (tf.errors.ResourceExhaustedError, RuntimeError, MemoryError):
            recommendations['tested_batch_sizes'].append(batch_size)
            recommendations['failed_batch_sizes'].append(batch_size)
            break
    
    # Recommend the largest successful batch size
    if recommendations['successful_batch_sizes']:
        recommendations['recommended_batch_size'] = recommendations['successful_batch_sizes'][-1]
    
    return recommendations


# ============================================================================
# 4. INFERENCE TIME BENCHMARKS
# ============================================================================

class InferenceBenchmark:
    """Benchmark inference speed and throughput"""
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.results = []
    
    def benchmark_single_sample(self, X: np.ndarray, num_runs: int = 100) -> Dict:
        """Benchmark single sample inference"""
        times = []
        
        # Warmup
        _ = self.model.predict(X[:1], verbose=0)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            _ = self.model.predict(X[:1], verbose=0)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        result = {
            'type': 'single_sample',
            'num_runs': num_runs,
            'mean_time_ms': times.mean(),
            'std_time_ms': times.std(),
            'min_time_ms': times.min(),
            'max_time_ms': times.max(),
            'median_time_ms': np.median(times),
            'throughput_samples_per_sec': 1000 / times.mean()
        }
        
        self.results.append(result)
        return result
    
    def benchmark_batch(self, X: np.ndarray, batch_sizes: List[int] = None, num_runs: int = 10) -> List[Dict]:
        """Benchmark batch inference at different batch sizes"""
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
        
        batch_results = []
        
        for batch_size in batch_sizes:
            if batch_size > len(X):
                continue
            
            times = []
            
            # Warmup
            _ = self.model.predict(X[:batch_size], verbose=0)
            
            # Benchmark
            for _ in range(num_runs):
                start = time.time()
                _ = self.model.predict(X[:batch_size], verbose=0)
                times.append((time.time() - start) * 1000)  # Convert to ms
            
            times = np.array(times)
            
            result = {
                'type': 'batch',
                'batch_size': batch_size,
                'num_runs': num_runs,
                'mean_time_ms': times.mean(),
                'std_time_ms': times.std(),
                'min_time_ms': times.min(),
                'max_time_ms': times.max(),
                'median_time_ms': np.median(times),
                'throughput_samples_per_sec': (batch_size * 1000) / times.mean()
            }
            
            batch_results.append(result)
            self.results.append(result)
        
        return batch_results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all benchmark results as DataFrame"""
        return pd.DataFrame(self.results)


# ============================================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_memory_usage(memory_tracker: MemoryTracker, figsize: Tuple = (14, 5)):
    """Plot memory usage over time"""
    df = memory_tracker.get_memory_dataframe()
    
    if df.empty:
        print("No memory data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # CPU Memory
    axes[0].plot(range(len(df)), df['cpu_rss_mb'], label='RSS (Resident)', marker='o')
    axes[0].plot(range(len(df)), df['cpu_vms_mb'], label='VMS (Virtual)', marker='s')
    axes[0].set_xlabel('Measurement')
    axes[0].set_ylabel('Memory (MB)')
    axes[0].set_title('CPU Memory Usage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CPU Percent
    axes[1].plot(range(len(df)), df['cpu_percent'], marker='o', color='orange')
    axes[1].set_xlabel('Measurement')
    axes[1].set_ylabel('CPU Usage (%)')
    axes[1].set_title('CPU Utilization')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_times(time_tracker: TrainingTimeTracker, figsize: Tuple = (14, 5)):
    """Plot training times"""
    epoch_df, batch_df = time_tracker.get_timings_dataframe()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Epoch times
    if not epoch_df.empty:
        axes[0].bar(epoch_df['epoch'], epoch_df['duration_seconds'], color='steelblue')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Time per Epoch')
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Batch times
    if not batch_df.empty:
        axes[1].scatter(batch_df['batch'], batch_df['duration_ms'], alpha=0.6, color='coral')
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('Time per Batch')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_inference_benchmark(benchmark: InferenceBenchmark, figsize: Tuple = (14, 5)):
    """Plot inference benchmark results"""
    df = benchmark.get_results_dataframe()
    
    if df.empty:
        print("No benchmark data to plot")
        return
    
    batch_df = df[df['type'] == 'batch'].copy()
    
    if batch_df.empty:
        print("No batch benchmark data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Inference time vs batch size
    axes[0].plot(batch_df['batch_size'], batch_df['mean_time_ms'], marker='o', linewidth=2, markersize=8)
    axes[0].fill_between(
        batch_df['batch_size'],
        batch_df['mean_time_ms'] - batch_df['std_time_ms'],
        batch_df['mean_time_ms'] + batch_df['std_time_ms'],
        alpha=0.3
    )
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Inference Time (ms)')
    axes[0].set_title('Inference Time vs Batch Size')
    axes[0].grid(True, alpha=0.3)
    
    # Throughput vs batch size
    axes[1].plot(batch_df['batch_size'], batch_df['throughput_samples_per_sec'], marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Throughput (samples/sec)')
    axes[1].set_title('Throughput vs Batch Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. COMPREHENSIVE PERFORMANCE REPORT
# ============================================================================

def generate_performance_report(
    model: tf.keras.Model,
    X_test: np.ndarray,
    memory_tracker: MemoryTracker,
    time_tracker: TrainingTimeTracker,
    benchmark: InferenceBenchmark,
    model_name: str = "Model"
) -> Dict:
    """Generate comprehensive performance report"""
    
    report = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        },
        'memory': memory_tracker.get_peak_memory(),
        'training_time': time_tracker.get_summary(),
        'inference': benchmark.get_results_dataframe().to_dict('records') if not benchmark.get_results_dataframe().empty else []
    }
    
    return report


def print_performance_summary(report: Dict):
    """Print formatted performance summary"""
    print("\n" + "="*70)
    print(f"PERFORMANCE REPORT: {report['model_name']}")
    print("="*70)
    
    print(f"\nTimestamp: {report['timestamp']}")
    
    # Model Info
    print("\n📊 MODEL INFORMATION")
    print(f"  Total Parameters: {report['model_info']['total_params']:,}")
    print(f"  Trainable Parameters: {report['model_info']['trainable_params']:,}")
    print(f"  Non-trainable Parameters: {report['model_info']['non_trainable_params']:,}")
    
    # Memory
    if report['memory']:
        print("\n💾 MEMORY USAGE")
        print(f"  Peak CPU RSS: {report['memory'].get('peak_cpu_rss_mb', 0):.2f} MB")
        print(f"  Peak CPU VMS: {report['memory'].get('peak_cpu_vms_mb', 0):.2f} MB")
        print(f"  Avg CPU RSS: {report['memory'].get('avg_cpu_rss_mb', 0):.2f} MB")
        print(f"  Peak CPU Usage: {report['memory'].get('peak_cpu_percent', 0):.2f}%")
    
    # Training Time
    if report['training_time']:
        print("\n⏱️  TRAINING TIME")
        if 'total_training_time_minutes' in report['training_time']:
            print(f"  Total Time: {report['training_time']['total_training_time_minutes']:.2f} minutes")
        if 'avg_epoch_time_seconds' in report['training_time']:
            print(f"  Avg per Epoch: {report['training_time']['avg_epoch_time_seconds']:.2f} seconds")
        if 'avg_batch_time_ms' in report['training_time']:
            print(f"  Avg per Batch: {report['training_time']['avg_batch_time_ms']:.2f} ms")
    
    # Inference
    if report['inference']:
        print("\n⚡ INFERENCE PERFORMANCE")
        for result in report['inference']:
            if result['type'] == 'batch':
                print(f"  Batch Size {result['batch_size']}: {result['mean_time_ms']:.2f}ms (±{result['std_time_ms']:.2f}ms)")
                print(f"    Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# 7. KERAS CALLBACK FOR AUTOMATIC TRACKING
# ============================================================================

class PerformanceTrackingCallback(tf.keras.callbacks.Callback):
    """Keras callback for automatic performance tracking"""
    
    def __init__(self, memory_tracker: MemoryTracker = None, time_tracker: TrainingTimeTracker = None):
        super().__init__()
        self.memory_tracker = memory_tracker or MemoryTracker()
        self.time_tracker = time_tracker or TrainingTimeTracker()
        self.batch_count = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.time_tracker.start_epoch(epoch)
        self.memory_tracker.log_memory(epoch=epoch)
    
    def on_epoch_end(self, epoch, logs=None):
        duration = self.time_tracker.end_epoch(epoch)
        self.memory_tracker.log_memory(epoch=epoch)
        if logs:
            logs['epoch_time_seconds'] = duration
    
    def on_train_batch_begin(self, batch, logs=None):
        self.time_tracker.start_batch(self.batch_count)
    
    def on_train_batch_end(self, batch, logs=None):
        self.time_tracker.end_batch(self.batch_count)
        self.batch_count += 1
