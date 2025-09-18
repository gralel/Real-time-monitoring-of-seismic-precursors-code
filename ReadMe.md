# Seismic Geomagnetic Signal Classification

## Quick Start Guide

### Project Structure

```
your_project/
├── data/                         # Dataset directory
│   ├── window-7/
│   │   ├── data_0.npy           # Class 0 samples (7-day)
│   │   └── data_1.npy           # Class 1 samples (7-day)
│   ├── window-14/
│   │   ├── data_0.npy           # Class 0 samples (14-day)
│   │   └── data_1.npy           # Class 1 samples (14-day)
│   └── window-30/
│       ├── data_0.npy           # Class 0 samples (30-day)
│       └── data_1.npy           # Class 1 samples (30-day)
│
├── results/                      # Model outputs
│   ├── gru_models/              # GRU model results
│   ├── lstm_models/             # LSTM model results
│   ├── rnn_models/              # RNN model results
│   ├── mlp_models/              # MLP model results
│   ├── transformer_models/      # Transformer model results
│   └── performance_visualization/  # Visualization outputs
│       ├── Individual_Confusion_Matrices/  # Confusion matrix plots
│       ├── Model_Comparison_Analysis/      # Comparative analysis plots
│       ├── Performance_Analysis/           # Training metrics plots
│       └── ROC_analysis/                   # ROC curve plots
│
├── logs/                         # Training logs
│   └── training_*.log           # Detailed training logs
│
├── GRU.ipynb                    # Complete GRU pipeline
├── LSTM.ipynb                   # Complete LSTM pipeline
├── RNN.ipynb                    # Complete RNN pipeline
├── MLP.ipynb                    # Complete MLP pipeline
├── Transformer.ipynb            # Complete Transformer pipeline
├── Result_drawing.ipynb         # Results visualization and plotting
└── README.md                    # Project documentation and usage guide
```

### Quick Usage Instructions

This project implements deep learning models for seismic geomagnetic signal classification. Each notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) contains the same 6-module structure:

#### Module Structure Overview

1. **Module 1: Environment Setup and Configuration** - Initializes the computational environment and ensures reproducibility
2. **Module 2: Data Exploration and Analysis** - Analyzes class distributions and handles data imbalance
3. **Module 3: Training Pipeline Utilities** - Provides utility functions for model training
4. **Module 4: Core Model Classes** - Defines model architectures with F1-weighted ensemble methods
5. **Module 5: Training Pipeline and K-Fold Cross-Validation** - Implements training functions with cross-validation
6. **Module 6: Training Orchestration and Ensemble Pipeline** - Orchestrates the complete training pipeline

#### Required Path Modifications

Before running any notebook, you need to modify the following paths to match your project location:

1. **Module 2 - Data Analysis** (Find and modify this line):
   ```python
   analyzer = DatasetAnalyzer(base_path="your_project/data")
   # Change to your actual data directory path
   # For example, if your project is at: C:/Users/Tian/Desktop/地磁论文代码运行测试
   # Then change to: analyzer = DatasetAnalyzer(base_path="C:/Users/Tian/Desktop/地磁论文代码运行测试/data")
   ```

2. **Module 5 - Training Functions** (Find and modify this line):
   ```python
   output_dir: str = "your_project/results"
   # Change to your actual results directory path
   # For example, if your project is at: C:/Users/Tian/Desktop/地磁论文代码运行测试
   # Then change to: output_dir: str = r"C:\Users\Tian\Desktop\地磁论文代码运行测试\results"
   ```

3. **Module 6 - Training Execution** (Find and modify this line):
   ```python
   base_path = "your_project"
   # Change to your actual project root directory
   # For example, if your project is at: C:/Users/Tian/Desktop/地磁论文代码运行测试
   # Then change to: base_path = r"C:\Users\Tian\Desktop\地磁论文代码运行测试"
   ```

**Note**: The example path "地磁论文代码运行测试" is equivalent to "your_project" in the directory structure above. Simply replace it with your actual project path.

#### Running the Training Code

1. Choose a model notebook (e.g., GRU.ipynb)
2. Run Module 1 to setup the environment
3. Modify the paths in Modules 2, 5, and 6 as described above
4. Run all modules sequentially (Modules 1-6)
5. The training will automatically process 7-day, 14-day, and 30-day windows
6. Results will be saved in the `results/{model_name}_models/` directory

#### Visualizing Results

After training all models, you can generate comprehensive visualizations using `Result_drawing.ipynb`:

1. Open `Result_drawing.ipynb`
2. Modify the base directory path in all four modules:
   ```python
   BASE_DIR = Path(r"your_project/results")
   # Change to your actual results directory
   # For example: BASE_DIR = Path(r"C:\Users\Tian\Desktop\地磁论文代码运行测试\results")
   ```
3. Run all four visualization modules sequentially
4. The visualizations will be saved in `results/performance_visualization/` subdirectories

The visualization notebook contains four modules that generate different analysis plots:
- Module 1: Training and validation metrics over epochs
- Module 2: Model comparison using radar charts and scatter plots
- Module 3: ROC curves for all models and time windows
- Module 4: Confusion matrices averaged from cross-validation

You can customize the plotting styles and add additional analyses based on your requirements.

For detailed information about each module's functionality and customization options, please refer to the comprehensive documentation below.

## Code Framework Documentation

### Module 1: Environment Setup and Configuration

#### Overview

This module initializes the computational environment for deep learning model training, ensuring reproducibility and optimal resource utilization across different hardware configurations.

#### Dependencies

##### Required Packages

| Package | Version Range | Purpose |
|---------|--------------|---------|
| Python | 3.9.0 - 3.11.x | Base interpreter |
| PyTorch | 2.0.0 - 2.3.1 | Deep learning framework |
| NumPy | 1.21.0+ (supports 2.x) | Numerical computations |
| scikit-learn | 1.0.0 - 1.5.x | Model evaluation metrics |

##### Optional Packages

| Package | Purpose |
|---------|---------|
| psutil | System resource monitoring |
| packaging | Version compatibility checking |
| CUDA | GPU acceleration (11.8 - 12.1) |

#### Installation

##### Using pip

```bash
# PyTorch with CUDA 12.1 support
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install numpy scikit-learn==1.5.0

# Optional packages
pip install psutil>=5.8.0 packaging
```

##### Using conda

```bash
# PyTorch with CUDA support
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Core dependencies
conda install numpy scikit-learn=1.5.0

# Optional packages
conda install -c conda-forge psutil packaging
```

#### Core Functionality

##### 1. Environment Verification
- Detects Python, PyTorch, NumPy, and CUDA versions
- Validates hardware configuration (CPU/GPU)
- Checks package compatibility

##### 2. Reproducibility Configuration
- Sets random seeds across all libraries (Python, NumPy, PyTorch, CUDA)
- Configures deterministic operations for consistent results
- Supports both standard and strict determinism modes

##### 3. Device Management
- Automatically selects optimal computation device (GPU/CPU)
- Configures GPU memory allocation limits
- Provides fallback mechanisms for resource-constrained environments

##### 4. Resource Monitoring
- Real-time memory usage tracking
- Batch size optimization based on available memory
- GPU cache management utilities

#### Usage

To use this module for model training experiments:

1. **Open the desired model notebook:**
   - `GRU.ipynb` - Gated Recurrent Unit model
   - `LSTM.ipynb` - Long Short-Term Memory model  
   - `RNN.ipynb` - Recurrent Neural Network model
   - `MLP.ipynb` - Multi-Layer Perceptron model
   - `Transformer.ipynb` - Transformer model

2. **Execute the following initialization sequence:**

```python
# Import the environment setup module
from environment_setup import *

# Verify environment compatibility
env_info = check_environment()
print_environment_info(env_info)

# Configure reproducibility - essential for paper reproduction
set_random_seeds(42)  # Fixed seed ensures consistent results

# Select computation device automatically
device = get_device()
print(f"Using device: {device}")

# Optional: Monitor available resources before training
memory_stats = get_memory_usage(device)
print(f"Available memory: {memory_stats['free_gb']:.2f} GB")
```

#### Key Functions

| Function | Description | Example |
|----------|-------------|---------|
| `check_environment()` | Verify system setup | `env = check_environment()` |
| `set_random_seeds(seed)` | Ensure reproducibility | `set_random_seeds(42)` |
| `get_device()` | Select GPU/CPU | `device = get_device()` |
| `get_memory_usage(device)` | Check memory | `stats = get_memory_usage(device)` |
| `clear_gpu_memory()` | Free GPU cache | `clear_gpu_memory()` |

#### NumPy Compatibility

The module automatically handles differences between NumPy 1.x and 2.x versions through a compatibility layer. No manual intervention required.

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not available | Ensure NVIDIA drivers (525.60.13+) are installed |
| Import errors | Install missing packages using pip/conda commands above |
| Out of memory | Reduce batch size or set `gpu_memory_fraction` parameter |
| Version conflicts | Check compatibility with `check_version_compatibility()` |
| Non-deterministic results | Use `set_random_seeds(42, strict_determinism=True)` |

#### Expected Output

When running Module 1 successfully, you should see:

```
PyTorch Environment Information
========================================
Software:
  Python:    3.11.7
  PyTorch:   2.3.0
  NumPy:     1.26.4 (Mode: numpy_1)

Hardware:
  CPUs:      16
  CUDA:      12.1
  GPUs:      1
  
Using device: cuda:0 (NVIDIA GeForce RTX 4090, 24.00 GB)
========================================
```

#### Notes

- Always run Module 1 before training any model to ensure proper environment setup
- The same seed value (42) should be used across all experiments for reproducibility
- GPU memory fraction can be adjusted if running multiple experiments simultaneously



### Module 2: Data Exploration and Analysis

#### Overview

This module provides comprehensive analysis of class distributions in binary classification datasets, with specialized support for temporal window-based data organization and imbalance detection. It automatically evaluates dataset characteristics and generates tailored recommendations for handling class imbalance.

**Implementation Note**: Module 2 is embedded directly in each model notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) as the second module following Module 1 (Environment Setup). Each notebook is self-contained and includes all necessary code.

#### Data Imbalance Explanation

The dataset exhibits class imbalance (non 1:1 ratio) due to several practical considerations:

1. **Multiple Station Recording**: A single seismic event may be recorded by multiple geomagnetic stations. All valid recordings are utilized to maximize training data availability.

2. **Data Quality Filtering**: Records containing excessive NaN values are excluded from the dataset to ensure training stability and model reliability.

3. **Instrument Reliability**: Some data loss occurs due to station equipment malfunctions or maintenance periods, affecting the available sample distribution.

4. **Data Confidentiality**: Due to data sensitivity requirements, only preprocessed `.npy` files are provided. Raw station-specific comparisons between seismic and non-seismic signals from the same geographic regions cannot be visualized in the public release.

These factors contribute to the natural class imbalance observed in the dataset, which the training pipeline addresses through:
- Weighted loss functions
- Balanced sampling strategies  
- Ensemble methods with F1-based weighting

#### Core Functionality

##### 1. Distribution Analysis
- Quantitative assessment of class distributions across temporal windows (7/14/30 days)
- Sample count and proportion calculation for each class
- Data validation and integrity checking

##### 2. Imbalance Detection
- Automatic severity categorization (balanced/mild/moderate/severe)
- Imbalance ratio calculation for each window
- Cross-window comparison and assessment

##### 3. Mitigation Strategies
- Class weight calculation using multiple methods (inverse/sqrt/effective)
- Loss function recommendations based on severity
- Sampling strategy suggestions for balanced training

##### 4. Implementation Guidance
- PyTorch-ready weight tensors
- Code snippets for immediate use
- Metric recommendations for imbalanced scenarios

#### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥ 1.21.0 | Data processing and analysis |
| Python | ≥ 3.9.0 | Base interpreter |

#### Data Format Requirements

| Aspect | Requirement | Description |
|--------|-------------|-------------|
| File Format | NumPy arrays (.npy) | Binary format for efficiency |
| Array Structure | 2D array | Shape: (n_samples, n_features + 1) |
| Label Position | Last column | Labels in final column (index -1) |
| Label Values | Binary (0, 1) | Integer labels only |
| File Naming | `data_0.npy`, `data_1.npy` | Class-specific files |

#### Usage

To analyze your dataset:

1. **Open any model notebook** (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, or Transformer.ipynb)

2. **Run Module 2 (Data Analysis) in the notebook:**
   - Module 2 is embedded directly in each notebook as the second module
   - No separate import needed - the code is self-contained within the notebook

```python
# Module 2 code is already included in the notebook
# Simply execute the Module 2 cells which contain:

# The DatasetAnalyzer class and analysis functions
class DatasetAnalyzer:
    # ... (full implementation in notebook)

# Quick analysis function
def quick_analyze(base_path=None, window_periods=['7', '14', '30']):
    # ... (implementation in notebook)

# Run the analysis
results = quick_analyze(base_path="./data")

# Or for detailed analysis
analyzer = DatasetAnalyzer(base_path="./data")
results = analyzer.analyze_all_windows()

# Access specific window results
window_7_analysis = results['7']
if window_7_analysis:
    print(f"Imbalance ratio: {window_7_analysis.imbalance_analysis.imbalance_ratio:.2f}:1")
    print(f"Suggested weights: {window_7_analysis.imbalance_analysis.suggested_weights}")
```

**Important**: Each notebook is self-contained and includes all necessary code. You don't need to install or import Module 2 separately - just run the cells in order.

#### Imbalance Classification Criteria

| Severity | Ratio Range | Strategy | Impact |
|----------|-------------|----------|---------|
| Balanced | < 1.5:1 | Standard training | No special handling |
| Mild | 1.5:1 - 3:1 | Class weights | Adjust loss function |
| Moderate | 3:1 - 10:1 | Weights + sampling | Balanced batches |
| Severe | > 10:1 | Comprehensive | Multiple techniques |

#### PyTorch Integration

The module provides ready-to-use weights for PyTorch:

```python
# Get weights from analysis
weights = results['7'].imbalance_analysis.suggested_weights

# Apply in PyTorch model
import torch
import torch.nn as nn

weights_tensor = torch.tensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# For severe imbalance
if results['7'].imbalance_analysis.severity == 'severe':
    # Consider Focal Loss or other specialized techniques
    from focal_loss import FocalLoss
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
```

#### Expected Output

```
======================================================================
Dataset Distribution Analysis
Base Path: /path/to/your/data
======================================================================

Analyzing window: 7 days
----------------------------------------
Total samples: 2,660
Class distribution: [1661, 999]
Class proportions: 62.4%, 37.6%

Imbalance Analysis:
  Ratio: 1.66:1
  Severity: MILD
  Strategy: Apply class weight adjustment
  Weights: [0.801, 1.331]

======================================================================
Summary
======================================================================
Windows analyzed: 3
Most imbalanced: Window 14 (2.27:1)
Assessment: Moderate imbalance

======================================================================
Implementation Guide
======================================================================

# Window 7 days:
weights = torch.tensor([0.801, 1.331])
criterion = nn.CrossEntropyLoss(weight=weights)
```

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| Data not found | Verify path and directory structure matches requirements |
| Invalid data format | Ensure .npy files contain 2D arrays with labels in last column |
| Only one class detected | Check if data files are correctly separated by class |
| Memory errors | Process windows sequentially rather than all at once |

#### Notes

- Analysis results are stored in memory and can be accessed programmatically
- Logging information is automatically written to console
- The module validates data integrity before analysis
- Recommendations are based on empirical best practices for imbalanced learning

### Module 3: Training Pipeline Utilities

#### Overview

This module provides essential utility functions for deep learning model training, including device configuration, data processing, model initialization, and checkpoint management. It supports multi-GPU training and implements various strategies for handling imbalanced datasets.

**Implementation Note**: Module 3 is embedded directly in each model notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) as the third module following Module 1 (Environment Setup) and Module 2 (Data Analysis). Each notebook contains the complete implementation tailored to its specific model architecture.

#### Core Functionality

##### 1. Setup & Configuration
- **Logging System**: Comprehensive logging to file and console
- **Device Management**: Automatic GPU detection and multi-GPU configuration
- **Reproducibility**: Random seed control across all libraries

##### 2. Data Processing
- **Class Weight Calculation**: Multiple methods for imbalanced data (inverse/sqrt/effective/balanced)
- **Balanced Sampling**: WeightedRandomSampler for training balance
- **Data Reshaping**: Model-specific data transformation

##### 3. Data Loading
- **Standard DataLoader**: Optimized for single/multi-GPU setups
- **Balanced DataLoader**: Automatic minority class oversampling
- **Auto-scaling**: Batch size and workers scale with GPU count

##### 4. Model Operations
- **Model Creation**: Unified interface with DataParallel support
- **Metric Calculation**: Comprehensive binary classification metrics
- **Parameter Counting**: Model size analysis

##### 5. Checkpoint Management
- **Save/Load States**: Complete training state preservation
- **Multi-GPU Compatibility**: Handles DataParallel models correctly

#### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0.0 | Deep learning framework |
| NumPy | ≥ 1.21.0 | Numerical operations |
| scikit-learn | ≥ 1.0.0 | Metrics calculation |

#### Output Directory Structure

Module 3 creates and manages the following output directories:

```
├── results/                      # Model-specific outputs
│   ├── gru_models/              # GRU: checkpoints, classifications, metrics
│   ├── lstm_models/             # LSTM: checkpoints, classifications, metrics
│   ├── rnn_models/              # RNN: checkpoints, classifications, metrics
│   ├── mlp_models/              # MLP: checkpoints, classifications, metrics
│   └── transformer_models/      # Transformer: checkpoints, classifications, metrics
│
└── logs/                         # Training logs (auto-created)
    └── training_*.log           # Timestamped training logs
```

**Note**: Checkpoints are saved within each model's results directory (e.g., `results/gru_models/checkpoint_*.pth`). For complete project structure including data organization, see Module 2.

#### Usage

To use the utility functions:

1. **Open the desired model notebook** (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, or Transformer.ipynb)

2. **Run Module 3 (Utilities) in the notebook:**

```python
# Module 3 code is embedded in each notebook
# Execute the Module 3 cells which contain all utility functions

# Setup environment
logger = setup_logging(
    log_file='training.log',
    log_dir='./logs',
    include_timestamp=True
)
device, gpu_ids = setup_device(gpu_ids=None)  # Auto-detect GPUs
set_seed(42, strict=False)

# Calculate class weights for imbalanced data
weights = calculate_class_weights(
    labels=train_labels,
    method='sqrt',  # Options: 'inverse', 'sqrt', 'effective', 'balanced'
    normalize=True
)

# Create balanced data loader
train_loader = create_data_loaders_balanced(
    dataset=train_dataset,
    batch_size=32,
    num_gpus=len(gpu_ids) if gpu_ids else 1,
    is_train=True,
    oversample=True
)

# Model initialization with multi-GPU support
model = create_model(
    model_class=YourModelClass,  # GRUClassifier, LSTMClassifier, etc.
    input_size=input_features,
    hidden_sizes=[512, 256],
    output_size=2,
    dropout_prob=0.3,
    device=device,
    device_ids=gpu_ids
)

# Save checkpoint during training
checkpoint_path = save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=current_epoch,
    loss=current_loss,
    checkpoint_dir='./results/gru_models',  # Saved in model-specific results directory
    additional_info={'best_f1': best_f1_score}
)

# Load checkpoint for resuming/inference
model, optimizer, epoch, loss, info = load_checkpoint(
    checkpoint_path='./results/gru_models/best_model.pth',
    model=model,
    optimizer=optimizer,
    device=device
)
```

#### Model-Specific Functions

Each notebook contains model-specific data reshaping functions:

| Model | Reshape Function | Purpose |
|-------|-----------------|---------|
| GRU | `reshape_data_for_gru(X, seq_length)` | Transform to (batch, seq_length, features) |
| LSTM | `reshape_data_for_lstm(X, seq_length)` | Transform to (batch, seq_length, features) |
| RNN | `reshape_data_for_rnn(X, seq_length)` | Transform to (batch, seq_length, features) |
| MLP | `prepare_mlp_data(X, flatten=True)` | Flatten input for feedforward network |
| Transformer | `reshape_data_for_transformer(X, seq_length)` | Transform to (batch, seq_length, features) |
| Transformer | `add_positional_encoding(X, d_model)` | Add positional embeddings |

#### Key Functions Reference

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `setup_logging()` | log_file, log_dir, level | Logger | Configure logging system |
| `setup_device()` | gpu_ids | (device, gpu_ids) | Configure computation device |
| `set_seed()` | seed, strict | None | Set random seeds |
| `calculate_class_weights()` | labels, method | Tensor | Compute class weights |
| `create_balanced_sampler()` | dataset, oversample | Sampler | Create balanced sampler |
| `create_data_loaders()` | dataset, batch_size | DataLoader | Standard data loader |
| `create_data_loaders_balanced()` | dataset, batch_size | DataLoader | Balanced data loader |
| `create_model()` | model_class, params | Module | Initialize model |
| `save_checkpoint()` | model, optimizer, epoch | Path | Save training state |
| `load_checkpoint()` | path, model | Tuple | Load training state |
| `calculate_binary_metrics()` | y_true, y_class | Dict | Compute all metrics |
| `run_epoch()` | model, loader, device | Dict | Execute training/validation epoch |

#### Multi-GPU Configuration

The module automatically handles multi-GPU setups:

```python
# Automatic detection and configuration
device, gpu_ids = setup_device()  # Uses all available GPUs

# Or specify GPUs
device, gpu_ids = setup_device(gpu_ids=[0, 1])  # Use GPU 0 and 1

# Batch size auto-scaling
# If using 2 GPUs with batch_size=32, effective batch = 64
train_loader = create_data_loaders_balanced(
    dataset=dataset,
    batch_size=32,  # Per GPU
    num_gpus=len(gpu_ids) if gpu_ids else 1
)
```

#### Class Weight Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `inverse` | n_samples / (n_classes × class_counts) | Standard weighting |
| `sqrt` | sqrt(n_samples / (n_classes × class_counts)) | Moderate adjustment |
| `effective` | (1 - β) / (1 - β^class_counts), β=0.999 | Severe imbalance |
| `balanced` | Same as inverse | sklearn-compatible |

#### Default Settings

- **Model Output Directory**: `./results/{model_type}_models/` (includes checkpoints and classifications)
- **Log Directory**: `./logs` (override via `LOG_DIR` environment variable)
- **Max Workers**: 16 (auto-scaled based on dataset size and GPU count)
- **Batch Size Scaling**: `effective_batch = batch_size × num_gpus`
- **Gradient Clipping**: Default max_grad_norm = 1.0

#### Expected Metrics Output

The `calculate_binary_metrics()` function returns:

```python
{
    'f1': 0.85,           # Weighted F1 score
    'precision': 0.82,    # Weighted precision
    'recall': 0.88,       # Weighted recall (sensitivity)
    'specificity': 0.79,  # True negative rate
    'mcc': 0.67,          # Matthews correlation coefficient
    'norm_mcc': 0.84,     # Normalized MCC [0,1]
    'class_acc': [0.79, 0.88]  # Per-class accuracy
}
```

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or use gradient accumulation |
| Single sample batch error | Module automatically skips single-sample batches |
| DataParallel not working | Ensure model is created before wrapping with DataParallel |
| Checkpoint loading fails | Check device mapping and model architecture match |
| Imbalanced training | Use `create_data_loaders_balanced()` with appropriate method |

#### Notes

- All functions include comprehensive error handling and logging
- Type hints are provided for better IDE support and code clarity
- The module handles both single-GPU and multi-GPU scenarios transparently
- Checkpoint saving includes timestamp for version control
- Metrics calculation includes both standard and specialized measures for imbalanced data

### Module 4: Core Model Classes

#### Overview

This module implements neural network architectures for binary classification of seismic geomagnetic signals. It includes five model architectures (GRU, LSTM, RNN, MLP, Transformer) and an F1-weighted ensemble method for combining multiple models.

**Implementation Note**: Module 4 is embedded directly in each model notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) as the fourth module. Each notebook contains the model implementation specific to its architecture.

#### Data Dimensions

The models are designed to work with three temporal windows:

| Window Period | Total Features | Sequence Length | Features per Timestep |
|--------------|----------------|-----------------|---------------------|
| 7 days | 10,080 | 7 | 1,440 |
| 14 days | 20,160 | 14 | 1,440 |
| 30 days | 43,200 | 30 | 1,440 |

**Note**: 
- MLP uses flattened input (total features directly)
- Sequential models (GRU/LSTM/RNN/Transformer) reshape data to (batch, sequence_length, features_per_timestep)

#### Model Architectures

##### Sequential Models (GRU, LSTM, RNN)

**Common Features**:
- Multi-layer architecture with configurable hidden dimensions (e.g., [512, 256])
- BatchNorm1d after each recurrent layer
- Dropout regularization (default 0.3)
- Optional bidirectional processing
- Xavier/Orthogonal weight initialization

**Key Differences**:
- **GRU**: Gated Recurrent Unit - fewer parameters, faster training
- **LSTM**: Long Short-Term Memory - better for long sequences
- **RNN**: Vanilla RNN - simplest architecture, configurable activation (tanh/relu)

##### MLP (Multi-Layer Perceptron)

- Feedforward architecture without temporal processing
- Direct input of flattened features (10,080/20,160/43,200)
- Multiple activation functions (ReLU, Tanh, GELU, ELU, LeakyReLU, SELU)
- BatchNorm and Dropout between layers

##### Transformer

- Self-attention mechanism for temporal modeling
- Configurable model dimension (d_model=512) and attention heads (nhead=8)
- Positional encoding for sequence awareness
- Global average pooling for aggregation
- 2 encoder layers (optimized for efficiency)

##### F1WeightedEnsemble

All models support ensemble combination:
- Weights models by their F1 scores
- Strategies: weighted_mean, voting, max confidence
- Provides uncertainty estimates
- Inference-only (no retraining needed)

#### Configuration Examples

##### Sequential Models (GRU/LSTM/RNN)
```python
model_config = {
    'input_size': 1440,                  # Features per timestep
    'hidden_sizes': [512, 256],          # Hidden layer dimensions
    'output_size': 2,                    # Binary classification
    'dropout_prob': 0.3,                 # Dropout rate
    'use_batch_norm': True,              # Enable BatchNorm
    'bidirectional': False,              # Unidirectional by default
}
```

##### MLP Configuration
```python
# Adjust input_size based on window
model_config_7d = {'input_size': 10080, 'hidden_sizes': [512, 256], ...}
model_config_14d = {'input_size': 20160, 'hidden_sizes': [512, 256], ...}
model_config_30d = {'input_size': 43200, 'hidden_sizes': [512, 256], ...}
```

##### Transformer Configuration
```python
model_config = {
    'input_size': 1440,                  # Features per timestep
    'd_model': 512,                      # Model dimension
    'nhead': 8,                          # Attention heads
    'num_encoder_layers': 2,             # Encoder layers
    'dim_feedforward': 2048,             # FFN dimension
}
```

#### Usage

To use the model classes:

1. **Open the desired model notebook** (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, or Transformer.ipynb)

2. **Run Module 4 (Model Classes) in the notebook:**

```python
# Module 4 contains the model class definition
# Example for GRU (similar for other models)

# Initialize model based on window size
# For 7-day window with GRU/LSTM/RNN/Transformer:
model = GRUClassifier(
    input_size=1440,              # Features per timestep (after reshape)
    hidden_sizes=[512, 256],      # Architecture
    output_size=2,                # Binary classification
    dropout_prob=0.3,
    use_batch_norm=True
)

# For 7-day window with MLP:
model = MLPClassifier(
    input_size=10080,             # Total flattened features
    hidden_sizes=[512, 256],
    output_size=2,
    dropout_prob=0.3,
    use_batch_norm=True
)

# Save model with checkpoint utility
model.save_checkpoint(
    filepath='./results/gru_models/model_best.pth',
    optimizer=optimizer,
    epoch=50,
    metrics={'f1': 0.87, 'accuracy': 0.92}
)

# Load from checkpoint
model = GRUClassifier.load_checkpoint('./results/gru_models/model_best.pth')

# Create ensemble from K-fold models
ensemble = F1WeightedEnsemble(
    models=[fold1_model, fold2_model, fold3_model, fold4_model, fold5_model],
    weights=[0.85, 0.87, 0.86, 0.84, 0.88],  # F1 scores from validation
    strategy='weighted_mean'
)

# Inference with uncertainty estimation
classifications, confidence, variance = ensemble.classify_with_confidence(test_data)
```

#### Model Selection Guidelines

| Model | Best For | Characteristics |
|-------|----------|-----------------|
| **GRU** | Baseline, efficiency | Fewer parameters than LSTM, faster training |
| **LSTM** | Long sequences | Better gradient flow, more complex gating |
| **RNN** | Simple patterns | Fastest, prone to vanishing gradients |
| **MLP** | Non-temporal patterns | No sequence modeling, fastest inference |
| **Transformer** | Complex dependencies | Attention mechanism, parallel processing |
| **Ensemble** | Production deployment | Best accuracy, uncertainty quantification |

#### Key Methods

All models implement:

| Method | Description |
|--------|-------------|
| `forward(x)` | Forward pass for classification |
| `save_checkpoint()` | Save model state and configuration |
| `load_checkpoint()` | Load model from saved state |
| `get_model_info()` | Return parameter count and architecture |
| `monitor_batch_norm_stats()` | Track BatchNorm statistics |

#### Training Recommendations

##### Architecture Selection
- Start with GRU for baseline
- Use LSTM for sequences > 20 timesteps
- Use Transformer for complex temporal patterns
- Use MLP when temporal order is less important

##### Hyperparameters
- Hidden sizes: [512, 256] works well for most cases
- Dropout: 0.3 for regularization
- BatchNorm: Enable for training stability
- Learning rate: 1e-3 to 1e-4 range

##### Ensemble Training
1. Train 5 models using K-fold cross-validation
2. Track F1 scores on validation sets
3. Create ensemble weighted by F1 scores
4. Use ensemble for final classification

#### Output Directory

Models save outputs to:
```
results/
├── gru_models/          # GRU checkpoints and classifications
├── lstm_models/         # LSTM checkpoints and classifications
├── rnn_models/          # RNN checkpoints and classifications
├── mlp_models/          # MLP checkpoints and classifications
└── transformer_models/  # Transformer checkpoints and classifications
```

#### Notes

- All models support multi-GPU training via DataParallel
- BatchNorm is applied differently for sequential vs non-sequential models
- Models automatically handle variable sequence lengths
- Ensemble models must be in eval mode for proper BatchNorm behavior
- Each notebook contains the complete model implementation specific to its architecture

### Module 5: Training Pipeline and K-Fold Cross-Validation

#### Overview

This module implements the complete training pipeline with stratified K-fold cross-validation for binary classification. It includes epoch execution, metrics evaluation, and model checkpointing with support for imbalanced datasets and multi-GPU training.

**Implementation Note**: Module 5 is embedded directly in each model notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) as the fifth module. Each notebook contains the training functions specific to its architecture.

#### Core Components

##### 1. Metrics Evaluation
- **Primary Metric**: F1-score (weighted) for model selection
- **Secondary Metrics**: MCC, Precision, Recall, Specificity
- **Class-wise Monitoring**: Per-class accuracy tracking
- **Comprehensive Output**: All metrics saved for post-analysis

##### 2. Training Features
- **Class Balancing**: Weighted loss functions and balanced sampling
- **Gradient Clipping**: Default max_norm=1.0 for stability
- **Multi-GPU Support**: Automatic DataParallel configuration
- **Cross-Validation**: Stratified K-fold preserving class distribution
- **Early Stopping**: Via learning rate schedulers
- **Checkpointing**: Automatic saving of best models

##### 3. K-Fold Strategy
- 5-fold stratified cross-validation (default)
- Maintains class proportions in each fold
- Independent training for each fold
- Ensemble creation from fold models

#### Main Training Functions

Each model has its specialized training function:

| Model | K-Fold Function | Epoch Function | Full Training Function |
|-------|----------------|----------------|----------------------|
| GRU | `kfold_train_eval_gru()` | `run_epoch()` | `train_eval_gru()` |
| LSTM | `kfold_train_eval_lstm()` | `run_epoch()` | `train_eval_lstm()` |
| RNN | `kfold_train_eval_rnn()` | `run_epoch()` | `train_eval_rnn()` |
| MLP | `kfold_train_eval_mlp()` | `run_epoch()` | `train_eval_mlp()` |
| Transformer | `kfold_train_eval_transformer()` | `run_epoch()` | `train_eval_transformer()` |

**Common Function**: `calculate_binary_metrics(y_true, y_class)` - Computes all evaluation metrics

#### Usage

To train models with K-fold cross-validation:

1. **Open the desired model notebook** (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, or Transformer.ipynb)

2. **Run Module 5 (Training Pipeline) in the notebook:**

```python
# Module 5 contains the training functions
# Execute the Module 5 cells which implement K-fold training

# Dataset preparation (common for all models)
X_tensor = torch.FloatTensor(X_scaled)  # Input features
y_tensor = torch.LongTensor(y)          # Labels
dataset = TensorDataset(X_tensor, y_tensor)

# Execute K-fold training (example for GRU, similar for others)
results = kfold_train_eval_gru(
    model_class=GRUClassifier,
    dataset=dataset,
    output_dir='./results/gru_models',
    loss_fn=nn.CrossEntropyLoss(weight=class_weights),  # Weighted for imbalance
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs={'lr': 0.001, 'weight_decay': 0.01},
    scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_kwargs={'mode': 'min', 'patience': 10, 'factor': 0.1},
    epochs=100,
    device=device,
    device_ids=gpu_ids,
    batch_size=32,
    num_folds=5,
    random_state=42,
    # Model-specific parameters
    input_size=1440,         # For sequential models
    hidden_sizes=[512, 256],
    output_size=2,
    dropout_prob=0.3
)

# Analyze results
mean_f1 = np.mean(results['val_f1s'])
std_f1 = np.std(results['val_f1s'])
print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
```

#### Model-Specific Parameters

##### Sequential Models (GRU/LSTM/RNN)
```python
# Common parameters
'input_size': features_per_timestep,  # After reshape
'hidden_sizes': [512, 256],
'dropout_prob': 0.3,
'epochs': 100

# RNN-specific
'nonlinearity': 'tanh'  # For RNN only
```

##### MLP
```python
# Flattened input
'input_size': 10080/20160/43200,  # Based on window
'hidden_sizes': [512, 256],
'dropout_prob': 0.3,
'activation': 'relu',
'epochs': 100
```

##### Transformer
```python
# Attention-based parameters
'input_size': features_per_timestep,
'd_model': 512,
'nhead': 8,
'num_encoder_layers': 2,
'dim_feedforward': 2048,
'dropout_prob': 0.1,
'epochs': 100

# Special scheduler
scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
scheduler_kwargs={'T_max': 100}
```

#### Output Files Structure

Each model saves comprehensive training artifacts:

```
results/
├── {model}_models/                        # Model-specific directory
│   ├── {Model}_{window}_fold_{k}.pth     # Trained model weights
│   ├── {Model}_{window}_fold_{k}_train_probs.npy   # Training probabilities
│   ├── {Model}_{window}_fold_{k}_test_probs.npy    # Validation probabilities
│   ├── {Model}_{window}_fold_{k}_train_labels.npy  # Training labels
│   ├── {Model}_{window}_fold_{k}_test_labels.npy   # Validation labels
│   ├── {Model}_{window}_fold_{k}_logs.txt          # Training logs
│   ├── {Model}_{window}_summary.json               # Aggregate metrics
│   ├── {Model}_{window}_ensemble.pth               # Ensemble model
│   └── {Model}_{window}_ensemble_config.json       # Ensemble configuration
```

Example: `GRUModel_7day_fold_1.pth` for GRU model, 7-day window, fold 1

#### Training Process Flow

1. **Data Splitting**: Stratified K-fold split maintaining class balance
2. **Fold Training**: Each fold trained independently
3. **Metrics Tracking**: Per-epoch metrics logged
4. **Model Saving**: Best model per fold saved
5. **Ensemble Creation**: F1-weighted combination of fold models
6. **Results Aggregation**: Summary statistics computed

#### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_folds` | 5 | Number of cross-validation folds |
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Maximum training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `random_state` | 42 | Random seed for reproducibility |

#### Metrics Computation

The `calculate_binary_metrics()` function computes:

```python
{
    'f1': weighted F1-score,
    'precision': weighted precision,
    'recall': weighted recall (sensitivity),
    'specificity': true negative rate,
    'mcc': Matthews correlation coefficient,
    'norm_mcc': normalized MCC [0,1],
    'class_acc': [class_0_accuracy, class_1_accuracy]
}
```

#### Learning Rate Schedulers

Different models use optimized schedulers:

| Model | Scheduler | Configuration |
|-------|-----------|---------------|
| GRU/LSTM/RNN/MLP | ReduceLROnPlateau | `mode='min', patience=10, factor=0.1` |
| Transformer | CosineAnnealingLR | `T_max=epochs` |

#### Results Analysis

After training, analyze the cross-validation results:

```python
# Extract metrics from results
val_f1_scores = results['val_f1s']
val_mcc_scores = results['val_mccs']

# Compute statistics
print(f"F1 Score: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}")
print(f"MCC: {np.mean(val_mcc_scores):.4f} ± {np.std(val_mcc_scores):.4f}")

# Load summary JSON for detailed analysis
with open('./results/gru_models/GRUModel_7day_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"Best fold: {summary['best_fold']}")
    print(f"Best F1: {summary['best_val_f1']:.4f}")
```

#### Training Tips

##### For Imbalanced Data
- Use weighted loss functions with class weights
- Enable balanced sampling in data loaders
- Monitor per-class accuracy
- Focus on F1 and MCC over accuracy

##### For Optimal Performance
- Start with default hyperparameters
- Use learning rate scheduling
- Enable gradient clipping
- Monitor validation metrics for early stopping signs

##### For Reproducibility
- Set random seeds consistently
- Save all configuration parameters
- Log comprehensive metrics
- Store model checkpoints regularly

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size or use gradient accumulation |
| Poor convergence | Adjust learning rate or try different optimizer |
| Overfitting | Increase dropout, add weight decay, reduce model size |
| Class imbalance issues | Verify class weights are applied correctly |
| Slow training | Check DataLoader workers, enable GPU, reduce logging frequency |

#### Notes

- Training automatically handles single-sample batches (skips for BatchNorm stability)
- All models support both CPU and multi-GPU training
- Ensemble models are created automatically after K-fold training
- Comprehensive logging ensures full reproducibility
- Results are saved incrementally to prevent data loss

### Module 6: Training Orchestration and Ensemble Pipeline

#### Overview

This module orchestrates the complete end-to-end training pipeline for seismic geomagnetic signal classification. It coordinates data loading, model training across multiple temporal windows, K-fold cross-validation, and F1-weighted ensemble creation for all five model architectures.

**Implementation Note**: Module 6 is embedded directly in each model notebook (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, Transformer.ipynb) as the sixth module. Each notebook contains the complete pipeline implementation specific to its architecture.

#### Pipeline Components

##### 1. Main Training Orchestrator
- **Multi-Window Processing**: Automatically trains on 7-day, 14-day, and 30-day windows
- **Data Integration**: Loads and preprocesses data for each window
- **Class Balance**: Applies sqrt-scaled weights for imbalanced datasets
- **Device Management**: Configures GPU/CPU with multi-GPU support

##### 2. F1-Weighted Ensemble System
- **Performance-Based Weighting**: Weights models by validation F1 scores
- **Automatic Creation**: Builds ensembles after K-fold training
- **Configuration Persistence**: Saves ensemble parameters and metrics
- **Model Loading**: Utilities for loading trained ensembles

##### 3. Comprehensive Metrics
- Tracks F1, Precision, Recall, Specificity, MCC across folds
- Saves individual fold classifications and labels
- Generates summary statistics and best model selection
- Exports JSON configuration for reproducibility

#### Main Functions

Each model has its orchestration function:

| Model | Main Function | Ensemble Function |
|-------|--------------|-------------------|
| GRU | `main_gru(optimize=False)` | `perform_model_ensemble()` |
| LSTM | `main_lstm(optimize=False)` | `perform_model_ensemble()` |
| RNN | `main_rnn(optimize=False)` | `perform_model_ensemble()` |
| MLP | `main_mlp(optimize=False)` | `perform_model_ensemble()` |
| Transformer | `main_transformer(optimize=False)` | `perform_model_ensemble()` |

**Utility Function**: `load_ensemble_model()` - Load saved ensemble with configuration

#### Configuration Parameters

##### Common Training Parameters

```python
# Standard configuration for model comparison
training_config = {
    'batch_size': 32,               # Training batch size
    'learning_rate': 0.001,         # Initial learning rate
    'weight_decay': 0.01,           # L2 regularization
    'max_grad_norm': 1.0,           # Gradient clipping
    'epochs': 100,                  # Maximum epochs (adjust as needed)
    'num_folds': 5                  # K-fold cross-validation
}
```

##### Model-Specific Parameters

###### Sequential Models (GRU/LSTM/RNN)
```python
{
    'hidden_sizes': [512, 256],    # Hidden layer dimensions
    'num_layers': 2,                # Number of recurrent layers
    'dropout_prob': 0.3,            # Dropout rate
    'nonlinearity': 'tanh'          # RNN-specific activation
}
```

###### MLP
```python
{
    'hidden_sizes': [512, 256],    # Hidden layer dimensions
    'dropout_prob': 0.3,            # Dropout rate
    'activation': 'relu'            # Activation function
}
```

###### Transformer
```python
{
    'd_model': 512,                 # Model dimension
    'nhead': 8,                     # Attention heads
    'num_encoder_layers': 2,        # Encoder layers
    'dim_feedforward': 2048,        # FFN dimension
    'dropout_prob': 0.1             # Lower dropout for attention
}
```

**Note**: Parameters are standardized for fair model comparison. Adjust based on your specific requirements and computational resources.

#### Time Window Configuration

All models process three temporal windows:

```python
# Sequential models (GRU/LSTM/RNN/Transformer)
window_mapping = {
    "window-7": {"name": "7day", "seq_length": 7},
    "window-14": {"name": "14day", "seq_length": 14},
    "window-30": {"name": "30day", "seq_length": 30}
}

# MLP (flattened features)
window_mapping = {
    "window-7": {"name": "7day", "features": 10080},
    "window-14": {"name": "14day", "features": 20160},
    "window-30": {"name": "30day", "features": 43200}
}
```

#### Usage

To execute the complete training pipeline:

1. **Open the desired model notebook** (GRU.ipynb, LSTM.ipynb, RNN.ipynb, MLP.ipynb, or Transformer.ipynb)

2. **Run Module 6 (Training Orchestration) in the notebook:**

```python
# Module 6 contains the complete pipeline
# Simply execute the main function

# Train all windows with default configuration
main_gru(optimize=False)  # For GRU (similar for other models)

# The pipeline will:
# 1. Setup logging and device configuration
# 2. Process each time window (7, 14, 30 days)
# 3. Load and prepare data with class weights
# 4. Execute K-fold cross-validation
# 5. Create F1-weighted ensemble
# 6. Save all models and metrics

# Load trained ensemble for inference
ensemble = load_ensemble_model(
    model_path="./results/gru_models/GRUModel_7day_ensemble.pth",
    config_path="./results/gru_models/GRUModel_7day_ensemble_config.json",
    device=device,
    model_class=GRUClassifier,
    device_ids=gpu_ids  # Optional for multi-GPU
)

# Use ensemble for classification
ensemble.eval()
with torch.no_grad():
    classifications = ensemble(test_data)
```

#### Pipeline Execution Flow

```
1. Initialize
   ├── Setup logging
   ├── Configure device (GPU/CPU)
   └── Create output directories

2. For each time window (7, 14, 30 days):
   ├── Load data (data_0.npy, data_1.npy)
   ├── Calculate class weights
   ├── Reshape data (model-specific)
   ├── Create dataset and tensors
   ├── Execute K-fold training
   │   └── For each fold:
   │       ├── Train model
   │       ├── Validate performance
   │       └── Save checkpoint
   ├── Create ensemble from folds
   └── Save ensemble configuration

3. Generate summary
   └── Export metrics and configuration
```

#### Output Structure

Each model creates organized outputs:

```
results/
└── {model}_models/
    # Individual fold models
    ├── {Model}Model_{window}_fold_{k}.pth
    ├── {Model}Model_{window}_fold_{k}_train_probs.npy
    ├── {Model}Model_{window}_fold_{k}_test_probs.npy
    ├── {Model}Model_{window}_fold_{k}_train_labels.npy
    ├── {Model}Model_{window}_fold_{k}_test_labels.npy
    
    # Summary and logs
    ├── {Model}Model_{window}_fold_{k}_logs.txt
    ├── {Model}Model_{window}_summary.json
    
    # Ensemble models
    ├── {Model}Model_{window}_ensemble.pth
    └── {Model}Model_{window}_ensemble_config.json
```

Example: `GRUModel_7day_fold_1.pth` for GRU model, 7-day window, fold 1

#### Ensemble Configuration

The ensemble system automatically:

1. **Calculates F1 weights** from validation performance
2. **Normalizes weights** to sum to 1.0
3. **Saves configuration** with all parameters and metrics
4. **Stores fold metrics** for analysis

Example ensemble config structure:
```json
{
    "model_name": "GRUModel",
    "window_name": "7day",
    "ensemble_weights": [0.198, 0.202, 0.201, 0.199, 0.200],
    "ensemble_method": "f1_weighted",
    "model_params": {...},
    "fold_metrics": {...},
    "average_metrics": {...}
}
```

#### Optimizer and Scheduler Configuration

Default configurations optimized for each model:

| Model | Optimizer | Scheduler |
|-------|-----------|-----------|
| GRU/LSTM/RNN/MLP | AdamW | ReduceLROnPlateau(mode='min', patience=10) |
| Transformer | AdamW | CosineAnnealingLR(T_max=epochs) |

#### Class Weight Calculation

All models use sqrt-scaled weights for imbalanced data:

```python
# Automatic calculation based on class distribution
class_weights = sqrt(n_samples / (n_classes * class_counts))
weighted_loss = nn.CrossEntropyLoss(weight=class_weights)
```

#### Customization Options

##### Adjusting Training Parameters

**Important**: There is no universal "best" parameter set. The default parameters are chosen for model comparison purposes. Optimal parameters depend on your specific data characteristics, computational resources, and performance requirements.

```python
# Example of how to modify parameters (not necessarily better)
# These are just examples showing different directions you could explore:

# Example 1: Different model capacity
training_config['hidden_sizes'] = [768, 384]  # Larger (may overfit)
# OR
training_config['hidden_sizes'] = [256, 128]  # Smaller (may underfit)

# Example 2: Different training duration
training_config['epochs'] = 50   # Faster training (may underfit)
# OR
training_config['epochs'] = 200  # Longer training (may overfit)

# Example 3: Different regularization
training_config['dropout_prob'] = 0.1  # Less regularization
# OR
training_config['dropout_prob'] = 0.5  # More regularization

# Always validate changes through cross-validation!
```

**Parameter Selection Guidelines**:
- Start with default parameters as baseline
- Change one parameter at a time
- Monitor validation metrics, not just training metrics
- Watch for overfitting (training >> validation performance)
- Consider computational cost vs. performance gain

##### Selecting Specific Windows

```python
# Process only specific windows
for window_period in ['7', '30']:  # Skip 14-day
    # Training code
```

##### Custom Data Path

```python
# Modify in main function
base_path = "/your/custom/data/path"
data_base_path = os.path.join(base_path, 'data')
```

#### Performance Optimization

##### For Faster Training
- Reduce `epochs` (e.g., 50 instead of 100)
- Increase `batch_size` if memory allows
- Use fewer folds (e.g., 3 instead of 5)
- Reduce model size (smaller `hidden_sizes`)

##### For Better Accuracy
- Increase `epochs` with early stopping
- Use smaller `batch_size` for better gradients
- Experiment with different `learning_rate`
- Try different weight decay values

##### For Limited Memory
- Reduce `batch_size`
- Use gradient accumulation
- Decrease model dimensions
- Process windows sequentially

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size or model dimensions |
| Slow convergence | Adjust learning rate or try different scheduler |
| Poor ensemble performance | Check individual fold performances |
| Missing output files | Verify data path and permissions |
| Device errors | Check CUDA availability and GPU memory |

#### Notes

- The pipeline automatically handles all aspects of training
- Ensemble weights are normalized F1 scores from validation
- All randomness is controlled for reproducibility
- Logs are comprehensive for debugging and analysis
- Parameters are standardized across models for fair comparison
- Adjust parameters based on your specific use case and resources

## Visualization Code Documentation

### Overview

The `Result_drawing.ipynb` notebook provides comprehensive visualization tools for analyzing the performance of trained models. It contains four specialized modules that generate different types of performance analysis plots for seismic geomagnetic signal classification.

### Module 1: Training and Validation Metrics Analysis

This module provides comprehensive performance visualization for deep learning models (GRU, LSTM, RNN, MLP, Transformer) across different temporal windows (7-day, 14-day, 30-day).

#### Functionality
- Parses training logs from all K-fold cross-validation folds
- Calculates weighted statistics using F1-based ensemble weights
- Generates 2×2 subplot figures showing training/validation accuracy and loss
- Automatically sorts legend entries by validation accuracy
- Uses transparent legend backgrounds with visible borders

#### Key Features
- **Weighted Averaging**: Uses ensemble weights from model configurations
- **Dynamic Scaling**: Automatically adjusts y-axis ranges based on data
- **Professional Styling**: Scientific publication-ready figures with 600 DPI
- **Comprehensive Metrics**: Tracks accuracy and loss for both training and validation

#### Output
- Saves plots to `results/performance_visualization/Performance_Analysis/`
- Generates one figure per time window: `Model_Performance_{window}.png`

#### Path Configuration
```python
BASE_DIR = Path(r"your_project/results")
# Change to your actual results directory
# Example: BASE_DIR = Path(r"C:\Users\Tian\Desktop\地磁论文代码运行测试\results")
```

### Module 2: Model Comparison Under Different Time Windows

This module provides comprehensive comparative analysis and visualization of model performance across different temporal windows using radar charts and scatter plots.

#### Functionality
- Loads ensemble configuration files containing averaged metrics
- Generates radar charts for multi-metric comparison
- Creates scatter plots for metric correlations and trade-offs
- Produces stability plots showing mean vs. standard deviation

#### Visualization Types
1. **Radar Charts**: Compare models across 5 metrics (F1, Precision, Recall, Specificity, Norm MCC)
2. **Stability Plots**: F1 score mean vs. standard deviation
3. **Correlation Plots**: Norm MCC vs. F1 score relationship
4. **Trade-off Analysis**: Precision vs. Recall balance

#### Key Features
- **Performance-Based Sorting**: Models ranked by performance metrics
- **Color Coding**: Consistent color scheme across all plots
- **Professional Legends**: Separate legend figures for different plot types
- **Statistical Analysis**: Includes mean and standard deviation from cross-validation

#### Output
- Saves plots to `results/performance_visualization/Model_Comparison_Analysis/`
- Generates 12 figures total (3 radar charts + 9 scatter plots)
- Includes separate legend files for radar and scatter plots

### Module 3: ROC Curve Analysis and Visualization

This module generates comparative ROC curves for multiple deep learning models across different temporal windows.

#### Functionality
- Loads test probabilities and labels from all cross-validation folds
- Calculates ROC curves for each fold and averages them
- Interpolates curves to common FPR points for comparison
- Computes AUC scores with standard errors

#### Key Features
- **Multi-Model Comparison**: All models displayed on same axes
- **Three-Panel Layout**: One subplot per time window
- **AUC-Based Sorting**: Legend entries sorted by performance
- **Statistical Robustness**: Uses averaged results from 5-fold cross-validation
- **Transparent Legends**: Professional appearance with transparent backgrounds

#### Output
- Saves plot to `results/performance_visualization/ROC_analysis/`
- Single figure with 3 subplots: `multi_model_roc_curves.png`
- High resolution (600 DPI) for publication quality

### Module 4: Confusion Matrix Analysis with Cross-Validation Averaging

This module generates individual confusion matrices for multiple deep learning models across different temporal windows, using averaged results from 5-fold cross-validation.

#### Functionality
- Loads test predictions and labels from all folds
- Calculates confusion matrices for each fold
- Averages confusion matrices across folds
- Creates individual plots for each model-window combination

#### Key Features
- **Individual Plots**: Separate confusion matrix for each model and window
- **Averaged Results**: Robust evaluation using K-fold averaging
- **Consistent Scaling**: Global color scale across all matrices
- **Professional Layout**: Fixed dimensions for consistent presentation
- **Horizontal Colorbar**: Separate colorbar figure for reference

#### Output
- Saves plots to `results/performance_visualization/Individual_Confusion_Matrices/`
- Generates 15 confusion matrix plots (5 models × 3 windows)
- Includes a horizontal colorbar: `horizontal_colorbar.png`

### Required Path Modifications

Before running the visualization notebook, modify the base directory path in each module:

```python
# In all four modules, find and modify:
BASE_DIR = Path(r"your_project/results")

# Change to your actual results directory
# For example, if your project is at: C:/Users/Tian/Desktop/地磁论文代码运行测试
# Then change to: BASE_DIR = Path(r"C:\Users\Tian\Desktop\地磁论文代码运行测试\results")
```

**Note**: All four visualization modules use the same variable name `BASE_DIR` (uppercase) for consistency.

### Dependencies for Visualization

```bash
# Required packages
pip install numpy matplotlib seaborn scipy scikit-learn pathlib

# Optional for enhanced plotting
pip install pandas
```

### Running the Visualizations

1. Ensure all model training is complete (all 5 notebooks executed)
2. Open `Result_drawing.ipynb`
3. Modify the `BASE_DIR` path in each module
4. Run all modules sequentially
5. Check the `results/performance_visualization/` directory for outputs

### Customization Options

#### Modify Color Schemes
```python
COLORS = {
    'GRU': '#E41A1C',         # Red
    'LSTM': '#377EB8',        # Blue
    'MLP': '#4DAF4A',         # Green
    'RNN': '#984EA3',         # Purple
    'Transformer': '#FF7F00'  # Orange
}
```

#### Adjust Figure Sizes
```python
plt.rcParams.update({
    'figure.figsize': (18, 16),  # Modify as needed
    'font.size': 12,             # Adjust font sizes
    'axes.labelsize': 32,        # Axis label size
})
```

#### Change Output Resolution
```python
plt.savefig(output_file, dpi=600)  # Adjust DPI for different resolutions
```

### Troubleshooting Visualization Issues

| Issue | Solution |
|-------|----------|
| Missing plots | Check that training generated all required .npy and .json files |
| Path errors | Verify BASE_DIR points to correct results directory |
| Memory issues | Process windows sequentially rather than all at once |
| Font issues | Install Arial font or modify font family in rcParams |
| Color issues | Ensure matplotlib and seaborn versions are compatible |

### Notes

- All visualization modules are independent and can be run separately
- The code automatically handles missing data gracefully
- Figures are optimized for scientific publication standards
- Custom analysis can be added by extending existing modules
- All plots use consistent styling for professional presentation