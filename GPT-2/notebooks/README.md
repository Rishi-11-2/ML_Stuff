# GPT-2 Training Notebooks

This directory contains a series of Jupyter notebooks that document the GPT-2 model training process. The notebooks are organized in a sequential manner, with each notebook building upon the previous one.

## Available Notebooks

### 1. `train_gpt2_1.ipynb`
- **Purpose**: Initial model setup and data preparation
- **Key Features**:
  - Model architecture definition
  - Dataset loading and preprocessing
  - Basic training loop implementation

### 2. `train_gpt2_2.ipynb`
- **Purpose**: Enhanced training with optimizations
- **Key Features**:
  - Learning rate scheduling
  - Gradient accumulation
  - Basic model evaluation

### 3. `train_gpt2_3.ipynb`
- **Purpose**: Advanced training techniques
- **Key Features**:
  - Mixed precision training
  - Gradient checkpointing
  - Model checkpointing

### 4. `train_gpt2_4.ipynb`
- **Purpose**: Final model training and evaluation
- **Key Features**:
  - Full-scale training
  - Comprehensive evaluation metrics
  - Text generation examples
1. **Before Committing**
   - Clear all outputs
   - Restart kernel and run all cells to ensure reproducibility
   - Add descriptive markdown cells

2. **Performance Tips**
   - Use `torch.no_grad()` for inference
   - Clear CUDA cache when needed: `torch.cuda.empty_cache()`
   - Use smaller batch sizes for interactive exploration

3. **Visualization**
   - Use interactive widgets for parameter exploration
   - Include both high-level overviews and detailed views
   - Document visualization functions for reuse

## Contributing

1. Follow the existing notebook structure
2. Add clear section headers
3. Include example usage
4. Document any additional dependencies
5. Keep visualizations clear and informative

## Troubleshooting

- **Kernel Dies**: Try reducing batch size or sequence length
- **CUDA Out of Memory**: Clear cache or restart kernel
- **Installation Issues**: Ensure all dependencies are installed in the correct environment
