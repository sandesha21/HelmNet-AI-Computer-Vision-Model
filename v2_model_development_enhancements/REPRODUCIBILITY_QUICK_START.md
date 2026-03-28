# Reproducibility Quick Start Guide

## Overview

This guide provides a quick reference for implementing reproducibility in HelmNet v2. The reproducibility section ensures that all experiments can be exactly reproduced by documenting seeds, environment, data, and parameters.

## What is Reproducibility?

Reproducibility means that running the same code with the same data on the same system produces identical results. This is essential for:
- Scientific validity and peer review
- Debugging and troubleshooting
- Production deployment
- Regulatory compliance
- Team collaboration

## Quick Implementation (5 minutes)

### Step 1: Import the Reproducibility Module

```python
import sys
sys.path.append('v2_model_development_enhancements')

from reproducibility_section import (
    set_all_seeds,
    display_environment_info,
    display_data_version_info,
    ModelParameters,
    generate_reproducibility_report,
    RANDOM_SEED
)
```

### Step 2: Set Random Seeds

```python
# Set all seeds for reproducibility
set_all_seeds(RANDOM_SEED)  # RANDOM_SEED = 812
```

**What this does:**
- Sets Python's random seed
- Sets NumPy's random seed
- Sets TensorFlow's random seed
- Enables deterministic GPU operations

### Step 3: Display Environment Information

```python
# Display system and library information
display_environment_info()
```

**Output includes:**
- Platform and Python version
- Library versions (TensorFlow, NumPy, Pandas, Keras)
- GPU availability and CUDA version
- Timestamp of execution

### Step 4: Display Data Versioning

```python
# Display data integrity information
display_data_version_info('images_proj.npy', 'Labels_proj.csv')
```

**Output includes:**
- File paths and sizes
- SHA256 hashes (fingerprints)
- Last modification times
- Data integrity verification

### Step 5: Display Model Parameters

```python
# Display all model hyperparameters
ModelParameters.display()
```

**Output includes:**
- Random seed configuration
- Data preprocessing parameters
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Early stopping configuration

### Step 6: Generate Reproducibility Report

```python
# Generate comprehensive reproducibility report
generate_reproducibility_report(
    images_path='images_proj.npy',
    labels_path='Labels_proj.csv',
    output_dir='reproducibility_artifacts'
)
```

**Creates:**
- `reproducibility_artifacts/model_parameters.csv`
- `reproducibility_artifacts/model_parameters.json`
- `reproducibility_artifacts/reproducibility_report.txt`

## Key Parameters

### Random Seed
```python
RANDOM_SEED = 812
```
- Controls all randomness in the pipeline
- Must be set before any random operations
- Same seed = same results

### Data Parameters
```python
IMAGE_SIZE = (200, 200)           # Image dimensions
GRAYSCALE = True                  # Convert to grayscale
TRAIN_SPLIT = 0.70               # 70% training data
VAL_SPLIT = 0.15                 # 15% validation data
TEST_SPLIT = 0.15                # 15% test data
STRATIFIED = True                # Maintain class balance
```

### Model Training Parameters
```python
# Model 1: Simple CNN
MODEL1_EPOCHS = 50
MODEL1_BATCH_SIZE = 32
MODEL1_LEARNING_RATE = 0.001

# Model 2: VGG-16 Base
MODEL2_EPOCHS = 50
MODEL2_BATCH_SIZE = 32
MODEL2_LEARNING_RATE = 0.0001

# Model 3: VGG-16 + FFNN
MODEL3_EPOCHS = 50
MODEL3_BATCH_SIZE = 32
MODEL3_LEARNING_RATE = 0.0001
MODEL3_DENSE_UNITS = [512, 256]
MODEL3_DROPOUT_RATE = 0.5

# Model 4: VGG-16 + FFNN + Augmentation
MODEL4_EPOCHS = 50
MODEL4_BATCH_SIZE = 32
MODEL4_LEARNING_RATE = 0.0001
MODEL4_DENSE_UNITS = [512, 256]
MODEL4_DROPOUT_RATE = 0.5
```

### Data Augmentation Parameters (Model 4)
```python
MODEL4_AUGMENTATION = {
    'rotation_range': 20,              # ±20 degrees
    'zoom_range': 0.2,                 # 0.8x to 1.2x
    'horizontal_flip': True,           # Flip horizontally
    'brightness_range': [0.8, 1.2],   # Brightness variation
    'fill_mode': 'nearest',            # Fill mode for rotations
}
```

### Early Stopping Parameters
```python
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 10          # Stop after 10 epochs without improvement
EARLY_STOPPING_RESTORE_BEST = True    # Use best model weights
```

## Modifying Parameters

### Option 1: Edit the Class (Recommended)

Edit `reproducibility_section.py`:

```python
class ModelParameters:
    RANDOM_SEED = 812  # Change this value
    MODEL1_EPOCHS = 50  # Change this value
    # ... other parameters
```

### Option 2: Override at Runtime

```python
# Create a copy and modify
params = ModelParameters.to_dict()
params['RANDOM_SEED'] = 999
params['MODEL1_EPOCHS'] = 100

# Use modified parameters in your code
```

## Saving and Sharing Parameters

### Save to CSV (for Excel/Sheets)
```python
ModelParameters.save_to_csv('model_parameters.csv')
```

### Save to JSON (for programmatic access)
```python
ModelParameters.save_to_json('model_parameters.json')
```

### Generate Full Report
```python
generate_reproducibility_report(
    images_path='images_proj.npy',
    labels_path='Labels_proj.csv',
    output_dir='reproducibility_artifacts'
)
```

## Reproducibility Checklist

Before running experiments:

- [ ] Random seeds set (RANDOM_SEED = 812)
- [ ] Environment information displayed
- [ ] Data versioning verified
- [ ] All parameters documented
- [ ] Reproducibility report generated
- [ ] Artifacts saved to `reproducibility_artifacts/`

## Troubleshooting

### Issue: Results still differ between runs

**Solution:** Ensure all seeds are set BEFORE any random operations:
```python
set_all_seeds(RANDOM_SEED)  # Must be first!
# Then load data and train models
```

### Issue: GPU results differ from CPU results

**Solution:** This is expected due to different numerical precision. Document which hardware was used:
```python
display_environment_info()  # Shows GPU availability
```

### Issue: Different results on different machines

**Solution:** Ensure same library versions:
```python
display_environment_info()  # Shows all library versions
```

## Best Practices

1. **Always set seeds first** - Before any random operations
2. **Document environment** - Record system and library versions
3. **Version your data** - Use file hashes to verify data integrity
4. **Centralize parameters** - Use `ModelParameters` class
5. **Generate reports** - Create reproducibility artifacts
6. **Share artifacts** - Include with published results
7. **Test reproducibility** - Run experiments twice and compare

## Integration with Notebook

Add these cells to your notebook in order:

1. **Markdown:** Section header and overview
2. **Code:** Import reproducibility module
3. **Code:** Set random seeds
4. **Code:** Display environment information
5. **Code:** Display data versioning
6. **Code:** Display model parameters
7. **Code:** Save parameters to files
8. **Code:** Generate reproducibility report
9. **Markdown:** Reproducibility checklist

See `REPRODUCIBILITY_NOTEBOOK_CELLS.md` for complete cell content.

## Files Generated

```
reproducibility_artifacts/
├── model_parameters.csv          # Parameters in CSV format
├── model_parameters.json         # Parameters in JSON format
└── reproducibility_report.txt    # Comprehensive text report
```

## Next Steps

1. Add reproducibility cells to notebook
2. Run all cells to generate artifacts
3. Verify `reproducibility_artifacts/` directory is created
4. Include artifacts with published results
5. Share with collaborators and reviewers

## References

- [TensorFlow Reproducibility Guide](https://www.tensorflow.org/guide/random_seed)
- [NumPy Random Seed Documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html)
- [Reproducible Research Best Practices](https://www.nature.com/articles/d41586-021-00592-0)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `REPRODUCIBILITY_NOTEBOOK_CELLS.md` for detailed cell content
3. Examine `reproducibility_section.py` for implementation details
4. Check generated `reproducibility_report.txt` for environment details
