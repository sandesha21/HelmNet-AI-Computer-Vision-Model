# ============================================================================
# MODEL DEVELOPMENT ENHANCEMENTS FOR V2 NOTEBOOK
# ============================================================================
# Copy this entire cell into your notebook after imports
# ============================================================================

import os
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from datetime import datetime

# ============================================================================
# 1. MODEL ARCHITECTURE DOCUMENTATION
# ============================================================================

def print_model_architecture(model, model_name):
    """
    Print enhanced model architecture with detailed information.
    Use this instead of just model.summary()
    """
    print("\n" + "="*80)
    print(f"MODEL ARCHITECTURE: {model_name}")
    print("="*80 + "\n")
    
    # Standard summary
    model.summary()
    
    # Additional statistics
    print("\n" + "-"*80)
    print("ARCHITECTURE STATISTICS")
    print("-"*80)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB (assuming float32)")
    print("="*80 + "\n")

# ============================================================================
# 2. HYPERPARAMETER DOCUMENTATION
# ============================================================================

HYPERPARAMETER_DOCS = {
    "model_1": {
        "name": "Simple CNN",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "justification": {
            "epochs": "20 epochs sufficient for baseline convergence on 631 samples",
            "batch_size": "32 balances gradient stability and memory efficiency",
            "learning_rate": "1e-3 provides good convergence speed for simple architecture",
            "optimizer": "Adam adapts learning rate per parameter, ideal for CNNs",
            "loss": "Binary Crossentropy optimal for binary classification"
        }
    },
    "model_2": {
        "name": "VGG-16 (Frozen Base)",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "justification": {
            "epochs": "20 epochs for transfer learning with frozen weights",
            "batch_size": "32 maintains consistency across models",
            "learning_rate": "1e-4 (lower) - frozen base requires minimal updates",
            "optimizer": "Adam with lower LR for fine-tuning pre-trained features",
            "loss": "Binary Crossentropy for binary classification"
        }
    },
    "model_3": {
        "name": "VGG-16 + FFNN",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "dropout": 0.5,
        "justification": {
            "epochs": "20 epochs for transfer learning with custom head",
            "batch_size": "32 for consistency",
            "learning_rate": "1e-4 for frozen base + custom head training",
            "dropout": "0.5 prevents overfitting in dense layers",
            "dense_layers": "256→128 progressively reduces dimensionality"
        }
    },
    "model_4": {
        "name": "VGG-16 + FFNN + Augmentation",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "dropout": 0.5,
        "augmentation": {
            "rotation_range": 20,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "brightness_range": [0.8, 1.2]
        },
        "justification": {
            "rotation": "Handles workers at different angles",
            "zoom": "Handles varying distances from camera",
            "horizontal_flip": "Handles workers from different sides",
            "brightness": "Handles varying lighting conditions"
        }
    }
}

def print_hyperparameter_docs(model_name):
    """Print hyperparameter documentation and justification."""
    if model_name not in HYPERPARAMETER_DOCS:
        print(f"Documentation not found for {model_name}")
        return
    
    doc = HYPERPARAMETER_DOCS[model_name]
    print("\n" + "="*80)
    print(f"HYPERPARAMETER DOCUMENTATION: {doc['name']}")
    print("="*80)
    
    print("\nHyperparameters:")
    for key, value in doc.items():
        if key not in ["name", "justification", "augmentation"]:
            print(f"  {key}: {value}")
    
    print("\nJustification:")
    for param, reason in doc.get("justification", {}).items():
        print(f"  • {param}: {reason}")
    
    if "augmentation" in doc:
        print("\nData Augmentation:")
        for aug, value in doc["augmentation"].items():
            print(f"  • {aug}: {value}")
    
    print("="*80 + "\n")

# ============================================================================
# 3. CALLBACKS FACTORY
# ============================================================================

def create_callbacks(model_name, checkpoint_dir="./model_checkpoints"):
    """
    Create training callbacks for model development.
    
    Includes:
    - EarlyStopping: Stops when validation loss plateaus
    - ModelCheckpoint: Saves best model weights
    - ReduceLROnPlateau: Reduces learning rate when loss plateaus
    - TensorBoard: Logs training metrics
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early Stopping - prevents overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    # Model Checkpointing - saves best weights
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.h5")
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    # Learning Rate Scheduling - adaptive learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    
    # TensorBoard Logging
    log_dir = os.path.join(
        "./logs",
        f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    return [early_stopping, model_checkpoint, reduce_lr, tensorboard]

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
INTEGRATION STEPS:

1. Copy this entire file into your notebook after imports

2. Before training Model 1, add:
   print_model_architecture(model_1, "Model 1: Simple CNN")
   print_hyperparameter_docs("model_1")

3. Update Model 1 training:
   history_1 = model_1.fit(
       X_train_normalized, y_train,
       epochs=20,
       batch_size=32,
       validation_data=(X_val_normalized, y_val),
       callbacks=create_callbacks("Model_1"),  # ADD THIS LINE
       verbose=2
   )

4. Repeat for Models 2-4

5. Load best model:
   from tensorflow.keras.models import load_model
   best_model = load_model('./model_checkpoints/Model_4_best.h5')

6. View TensorBoard:
   tensorboard --logdir ./logs
"""

print("✓ Model development enhancements loaded")
