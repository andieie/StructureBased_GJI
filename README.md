# StructureBased_GJI
Structure-based or Geometric inversion using implicit models. Code base with synthetic examples.

## Quick Setup

### Requirements
- **Python 3.11** (recommended via conda)
- **PyGIMLi 1.4.0+** (geophysical modeling)
- **GemPy 2.2.0** (geological modeling)
- **NumPy, Matplotlib, SciPy** (scientific computing)

### Installation
```bash
# Option 1: Using conda environment (recommended)
conda env create -f environment.yml
conda activate gempyenvf

# Option 2: Using pip
pip install -r requirements.txt
```

### Project Structure
```
StructureBased_GJI/
├── modelling_class/              # Core geometric inversion class
│   └── GeometricInversion2d.py   # Main inversion implementation
├── utilsGeo/                     # Utility functions
│   ├── get_gempy_model.py        # GemPy-PyGIMLi integration
│   ├── utils.py                  # Data processing utilities
│   └── plotting_params.py        # Visualization settings
└── synthetic_examples/           # Complete workflow examples
    ├── createdata_homo_xhole.py  # Generate homogeneous synthetic data
    ├── createdata_hetero_xhole.py # Generate heterogeneous synthetic data
    ├── data/                     # Output directory for synthetic data
    └── inversion/
        └── GI_xhole_2D_tt.py     # Run geometric inversion
```

### Running the Examples

1. **Generate synthetic data:**
   ```bash
   cd synthetic_examples/
   python createdata_homo_xhole.py    # Creates homogeneous crosshole data
   python createdata_hetero_xhole.py  # Creates heterogeneous crosshole data
   ```

2. **Run geometric inversion:**
   ```bash
   cd synthetic_examples/inversion/
   python GI_xhole_2D_tt.py          # Performs structure-based inversion
   ```

### Key Components

- **GeometricInversion2d**: Main class implementing structure-based joint inversion
- **GemPy Integration**: Implicit geological modeling with fault interface parameterization  
- **PyGIMLi Forward Operators**: Travel time and ERT simulation on dynamically updated meshes
- **Synthetic Examples**: Complete crosshole survey scenarios with controllable geological complexity

## Citation
If you use this code in your research, please cite:

Balza, A. et al. (2025). *Structure-based joint inversion for implicit geological models.* Geophysical Journal International.

BibTeX:
@article{BalzaMorales2025StructureBased,
  title        = {Structure-based geophysical inversion using implicit geological models},
  author       = {Balza Morales, Andrea and Förderer, Aaron and Wellmann, Florian and Maurer, Hansruedi and Wagner, Florian M.},
  year         = {2025},
  journal      = {Geophysical Journal International},
  note         = {Submitted manuscript},
  month        = {July},
  institution  = {RWTH Aachen University and ETH Zurich},
}


Funding for this project was provided by the European
Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie
grant agreement No. 956965, through the EASYGO project (https://easygo-itn.eu). 

Disclaimer: GITHUB Co-pilot was used for the documentation and re-factoring of the original code.