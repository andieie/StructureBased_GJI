# Environment Setup Instructions

This project is designed to work with the `gempyenvf` conda environment. Follow these instructions to set up the required dependencies.

## Option 1: Using the provided conda environment file (Recommended)

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate gempyenvf
```

## Option 2: Using pip requirements

If you prefer to use pip or need to install packages in an existing environment:

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

## Option 3: Manual setup

If you need to set up the environment manually:

```bash
# Create a new conda environment
conda create -n gempyenvf python=3.11.8

# Activate the environment
conda activate gempyenvf

# Install PyGIMLi (geophysical inversion library)
conda install -c gimli pygimli

# Install GemPy (geological modeling)
conda install -c conda-forge gempy

# Install other dependencies
conda install -c conda-forge numpy matplotlib scipy pandas pyvista vtk jupyterlab
```

## Important Notes

1. **PyGIMLi**: This is a key dependency for geophysical inversion. It's available through the `gimli` conda channel.

2. **GemPy**: Used for geological modeling and is available through conda-forge.

3. **Custom utilsGeo**: The project includes custom utility modules in the `utilsGeo/` directory that provide additional functionality.

4. **Python Version**: This project is tested with Python 3.11.8. Other versions may work but are not guaranteed.

## Verification

To verify your installation is working correctly, try running:

```python
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import gempy
print("All dependencies imported successfully!")
```

## Troubleshooting

- If PyGIMLi installation fails, try installing from source following instructions at: https://www.pygimli.org/installation.html
- For GemPy issues, refer to: https://docs.gempy.org/installation.html
- Ensure you have the latest conda version: `conda update conda`