# ============================================================================
# SECTION 8: REPRODUCIBILITY
# ============================================================================
# This module contains all reproducibility-related functions and configurations
# for the HelmNet v2 notebook

import os
import sys
import platform
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# ============================================================================
# 8.1 SEED MANAGEMENT
# ============================================================================

RANDOM_SEED = 812

def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set all random seeds for complete reproducibility.
    
    This function ensures deterministic behavior across:
    - NumPy random number generation
    - TensorFlow/Keras operations
    - Python's random module
    - CUDA operations (if GPU is available)
    
    Args:
        seed (int): Random seed value. Default: 812
    
    Returns:
        None
    
    Note:
        Must be called before any model training or data operations
        that involve randomness.
    """
    # Set Python's random seed
    import random
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set TensorFlow's random seed
    tf.random.set_seed(seed)
    
    # Set Keras random seed
    keras.utils.set_random_seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"✓ All random seeds set to {seed}")
    print("✓ Deterministic behavior enabled")


# ============================================================================
# 8.2 ENVIRONMENT INFORMATION
# ============================================================================

def get_environment_info() -> dict:
    """
    Collect comprehensive environment information for reproducibility.
    
    Returns:
        dict: Dictionary containing system and library version information
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'machine': platform.machine(),
        },
        'libraries': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'tensorflow': tf.__version__,
            'keras': keras.__version__,
        },
        'gpu': {
            'available': len(tf.config.list_physical_devices('GPU')) > 0,
            'count': len(tf.config.list_physical_devices('GPU')),
            'devices': [str(d) for d in tf.config.list_physical_devices('GPU')],
        }
    }
    
    # Try to get CUDA version if available
    try:
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        env_info['gpu']['cuda_version'] = cuda_version.split('\n')[0]
    except:
        env_info['gpu']['cuda_version'] = 'Not available'
    
    return env_info


def display_environment_info() -> None:
    """
    Display formatted environment information for documentation.
    
    This function prints a comprehensive report of the execution environment,
    which is essential for reproducibility and debugging.
    """
    env_info = get_environment_info()
    
    print("\n" + "="*70)
    print("ENVIRONMENT INFORMATION")
    print("="*70 + "\n")
    
    # System Information
    print("SYSTEM INFORMATION:")
    print(f"  Timestamp: {env_info['timestamp']}")
    print(f"  Platform: {env_info['system']['platform']}")
    print(f"  Python Version: {env_info['system']['python_version']}")
    print(f"  Processor: {env_info['system']['processor']}")
    print(f"  Machine: {env_info['system']['machine']}\n")
    
    # Library Versions
    print("LIBRARY VERSIONS:")
    print(f"  NumPy: {env_info['libraries']['numpy']}")
    print(f"  Pandas: {env_info['libraries']['pandas']}")
    print(f"  TensorFlow: {env_info['libraries']['tensorflow']}")
    print(f"  Keras: {env_info['libraries']['keras']}\n")
    
    # GPU Information
    print("GPU INFORMATION:")
    print(f"  GPU Available: {'Yes' if env_info['gpu']['available'] else 'No'}")
    print(f"  GPU Count: {env_info['gpu']['count']}")
    if env_info['gpu']['devices']:
        print(f"  GPU Devices:")
        for device in env_info['gpu']['devices']:
            print(f"    - {device}")
    print(f"  CUDA Version: {env_info['gpu']['cuda_version']}\n")
    
    print("="*70 + "\n")


# ============================================================================
# 8.3 DATA VERSIONING
# ============================================================================

def get_data_version_info(images_path: str, labels_path: str) -> dict:
    """
    Collect data versioning information.
    
    Args:
        images_path (str): Path to images file
        labels_path (str): Path to labels file
    
    Returns:
        dict: Dictionary containing data version information
    """
    import hashlib
    
    def get_file_hash(filepath: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    data_info = {
        'images': {
            'path': images_path,
            'size_mb': os.path.getsize(images_path) / (1024 * 1024),
            'hash': get_file_hash(images_path),
            'modified': datetime.fromtimestamp(os.path.getmtime(images_path)).isoformat(),
        },
        'labels': {
            'path': labels_path,
            'size_kb': os.path.getsize(labels_path) / 1024,
            'hash': get_file_hash(labels_path),
            'modified': datetime.fromtimestamp(os.path.getmtime(labels_path)).isoformat(),
        }
    }
    
    return data_info


def display_data_version_info(images_path: str, labels_path: str) -> None:
    """
    Display formatted data versioning information.
    
    Args:
        images_path (str): Path to images file
        labels_path (str): Path to labels file
    """
    data_info = get_data_version_info(images_path, labels_path)
    
    print("\n" + "="*70)
    print("DATA VERSIONING INFORMATION")
    print("="*70 + "\n")
    
    print("IMAGES FILE:")
    print(f"  Path: {data_info['images']['path']}")
    print(f"  Size: {data_info['images']['size_mb']:.2f} MB")
    print(f"  SHA256 Hash: {data_info['images']['hash']}")
    print(f"  Last Modified: {data_info['images']['modified']}\n")
    
    print("LABELS FILE:")
    print(f"  Path: {data_info['labels']['path']}")
    print(f"  Size: {data_info['labels']['size_kb']:.2f} KB")
    print(f"  SHA256 Hash: {data_info['labels']['hash']}")
    print(f"  Last Modified: {data_info['labels']['modified']}\n")
    
    print("="*70 + "\n")


# ============================================================================
# 8.4 PARAMETER DOCUMENTATION
# ============================================================================

class ModelParameters:
    """
    Centralized configuration for all model hyperparameters.
    
    This class serves as a single source of truth for all model parameters,
    making it easy to modify and track configurations across experiments.
    """
    
    # ========== RANDOM SEED ==========
    RANDOM_SEED = 812
    
    # ========== DATA PARAMETERS ==========
    IMAGE_SIZE = (200, 200)
    GRAYSCALE = True
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    STRATIFIED = True
    
    # ========== NORMALIZATION PARAMETERS ==========
    NORMALIZE_RANGE = (0, 1)  # Normalize to [0, 1]
    
    # ========== MODEL 1: SIMPLE CNN ==========
    MODEL1_NAME = "Simple CNN"
    MODEL1_EPOCHS = 50
    MODEL1_BATCH_SIZE = 32
    MODEL1_LEARNING_RATE = 0.001
    MODEL1_OPTIMIZER = "adam"
    
    # ========== MODEL 2: VGG-16 BASE ==========
    MODEL2_NAME = "VGG-16 (Base)"
    MODEL2_EPOCHS = 50
    MODEL2_BATCH_SIZE = 32
    MODEL2_LEARNING_RATE = 0.0001
    MODEL2_OPTIMIZER = "adam"
    MODEL2_FREEZE_LAYERS = True
    
    # ========== MODEL 3: VGG-16 + FFNN ==========
    MODEL3_NAME = "VGG-16 + FFNN"
    MODEL3_EPOCHS = 50
    MODEL3_BATCH_SIZE = 32
    MODEL3_LEARNING_RATE = 0.0001
    MODEL3_OPTIMIZER = "adam"
    MODEL3_FREEZE_LAYERS = True
    MODEL3_DENSE_UNITS = [512, 256]
    MODEL3_DROPOUT_RATE = 0.5
    
    # ========== MODEL 4: VGG-16 + FFNN + AUGMENTATION ==========
    MODEL4_NAME = "VGG-16 + FFNN + Augmentation"
    MODEL4_EPOCHS = 50
    MODEL4_BATCH_SIZE = 32
    MODEL4_LEARNING_RATE = 0.0001
    MODEL4_OPTIMIZER = "adam"
    MODEL4_FREEZE_LAYERS = True
    MODEL4_DENSE_UNITS = [512, 256]
    MODEL4_DROPOUT_RATE = 0.5
    
    # Data Augmentation Parameters
    MODEL4_AUGMENTATION = {
        'rotation_range': 20,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest',
    }
    
    # ========== EARLY STOPPING PARAMETERS ==========
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_RESTORE_BEST = True
    
    # ========== EVALUATION PARAMETERS ==========
    EVALUATION_METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'confusion_matrix', 'classification_report'
    ]
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert all parameters to a dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def display(cls) -> None:
        """Display all parameters in a formatted manner."""
        params = cls.to_dict()
        
        print("\n" + "="*70)
        print("MODEL PARAMETERS DOCUMENTATION")
        print("="*70 + "\n")
        
        # Group parameters by category
        categories = {
            'RANDOM': [],
            'DATA': [],
            'NORMALIZATION': [],
            'MODEL1': [],
            'MODEL2': [],
            'MODEL3': [],
            'MODEL4': [],
            'EARLY_STOPPING': [],
            'EVALUATION': [],
        }
        
        for key, value in params.items():
            for category in categories:
                if key.startswith(category):
                    categories[category].append((key, value))
                    break
        
        for category, items in categories.items():
            if items:
                print(f"{category} PARAMETERS:")
                for key, value in sorted(items):
                    # Format the key nicely
                    formatted_key = key.replace('_', ' ').title()
                    print(f"  {formatted_key}: {value}")
                print()
        
        print("="*70 + "\n")
    
    @classmethod
    def save_to_csv(cls, filepath: str) -> None:
        """Save parameters to a CSV file for documentation."""
        params = cls.to_dict()
        df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        df.to_csv(filepath, index=False)
        print(f"✓ Parameters saved to {filepath}")
    
    @classmethod
    def save_to_json(cls, filepath: str) -> None:
        """Save parameters to a JSON file for documentation."""
        import json
        params = cls.to_dict()
        
        # Convert non-serializable objects to strings
        for key, value in params.items():
            if isinstance(value, dict):
                params[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ Parameters saved to {filepath}")


# ============================================================================
# 8.5 REPRODUCIBILITY REPORT
# ============================================================================

def generate_reproducibility_report(
    images_path: str,
    labels_path: str,
    output_dir: str = "."
) -> None:
    """
    Generate a comprehensive reproducibility report.
    
    This function creates a complete documentation of the execution environment,
    data versioning, and model parameters for full reproducibility.
    
    Args:
        images_path (str): Path to images file
        labels_path (str): Path to labels file
        output_dir (str): Directory to save the report. Default: current directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Display all information
    display_environment_info()
    display_data_version_info(images_path, labels_path)
    ModelParameters.display()
    
    # Save parameters to files
    ModelParameters.save_to_csv(os.path.join(output_dir, 'model_parameters.csv'))
    ModelParameters.save_to_json(os.path.join(output_dir, 'model_parameters.json'))
    
    # Create a comprehensive text report
    report_path = os.path.join(output_dir, 'reproducibility_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HELMNET V2 - REPRODUCIBILITY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Environment Info
        env_info = get_environment_info()
        f.write("ENVIRONMENT INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Timestamp: {env_info['timestamp']}\n")
        f.write(f"Platform: {env_info['system']['platform']}\n")
        f.write(f"Python Version: {env_info['system']['python_version']}\n")
        f.write(f"TensorFlow Version: {env_info['libraries']['tensorflow']}\n")
        f.write(f"GPU Available: {env_info['gpu']['available']}\n")
        f.write(f"GPU Count: {env_info['gpu']['count']}\n\n")
        
        # Data Info
        data_info = get_data_version_info(images_path, labels_path)
        f.write("DATA VERSIONING INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Images Path: {data_info['images']['path']}\n")
        f.write(f"Images Hash: {data_info['images']['hash']}\n")
        f.write(f"Labels Path: {data_info['labels']['path']}\n")
        f.write(f"Labels Hash: {data_info['labels']['hash']}\n\n")
        
        # Parameters
        f.write("MODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        params = ModelParameters.to_dict()
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Reproducibility report saved to {report_path}")
