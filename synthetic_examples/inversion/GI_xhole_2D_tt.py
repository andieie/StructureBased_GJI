"""
Geometric Inversion for Cross-hole Travel Time Data
=================================================

This script performs geometric inversion on cross-hole travel time data to recover
fault geometry by optimizing the positions of interface points. The inversion uses
a custom forward operator that combines geological modeling (GemPy) with PyGIMLi
travel time simulation.

Data Setup:
-----------
Before running this script, ensure you have the following directory structure:

StructureBased_GJI/
├── modelling_class/
├── utilsGeo/
└── synthetic_examples/
    ├── data/                         # Synthetic data files
    │   ├── XHOLE_HOMO_2D_TT_DATA_c500.sgt
    │   └── schemes/                  # Optional: measurement schemes
    │       └── scheme.sgt
    ├── createdata_homo_xhole.py      # Data generation scripts
    ├── createdata_hetero_xhole.py
    └── inversion/
        └── GI_xhole_2D_tt.py        # This script

If data files are not found, the script will provide instructions on where to place them.
The synthetic data should be generated using the createdata_*.py scripts.

Author: Structure-Based Inversion Research Group  
Date: October 2025
License: See LICENSE file for details
"""

# %% IMPORTS AND SETUP
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import gempy as gp 
from datetime import datetime
import sys
import os
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Add project root to Python path for module imports
import sys
import os

def find_project_root(current_file=__file__, marker_files=['modelling_class', 'utilsGeo']):
    """
    Find the project root directory by looking for specific marker directories.
    
    Parameters:
    -----------
    current_file : str
        The current file path (__file__)
    marker_files : list
        List of directories/files that indicate the project root
        
    Returns:
    --------
    str
        Absolute path to the project root
    """
    current_dir = os.path.dirname(os.path.abspath(current_file))
    
    while current_dir != os.path.dirname(current_dir):  # Not at filesystem root
        # Check if marker files exist in current directory
        if all(os.path.exists(os.path.join(current_dir, marker)) for marker in marker_files):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Fallback to relative path if markers not found
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add project root to path
project_root = find_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import custom modules
from utilsGeo import get_gempy_model
from modelling_class import GeometricInversion2d
import pygimli.physics.ert as ert
import pygimli.physics.traveltime as tt
from utilsGeo.plotting_params import scatter_kwargs, pg_show_kwargs, set_style, set_legend_white

# Set plotting style
set_style(fs=15)

# Configuration parameters
LAMBDA = 1  # Constraint strength factor
# Use relative path for data - synthetic data stored in synthetic_examples directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SYNTHETIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(SYNTHETIC_DIR, 'data/')
DATASET = 'XHOLE_HOMO_2D_TT'
METHOD = 'TravelTime'
HOMOGENEOUS = True

# Create unique filename for results
experiment_name = f'XHOLE_d{DATASET}lam{LAMBDA}x'

print(f"Starting geometric inversion experiment: {experiment_name}")
print(f"Method: {METHOD}")
print(f"Lambda: {LAMBDA}")
print(f"Homogeneous model: {HOMOGENEOUS}")
# %%============================================================================
# DATA LOADING AND INITIAL SETUP
# ============================================================================

def load_synthetic_data(dataset_path, dataset_name):
    """
    Load synthetic travel time data.
    
    Parameters
    ----------
    dataset_path : str
        Path to data directory
    dataset_name : str
        Name of dataset file
        
    Returns
    -------
    pg.DataContainer
        Loaded travel time data
        
    Notes
    -----
    If the data file doesn't exist, the script will provide instructions
    on where to place the synthetic data files.
    """
    # Ensure data directory exists
    os.makedirs(dataset_path, exist_ok=True)
    
    data_file = os.path.join(dataset_path, dataset_name + '_DATA_c500.sgt')
    
    try:
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print(f"Please ensure synthetic data files are placed in: {dataset_path}")
            print(f"Expected file: {dataset_name}_DATA_c500.sgt")
            raise FileNotFoundError(f"Synthetic data file not found: {data_file}")
            
        data = pg.load(data_file)
        print(f"Successfully loaded data: {data.size()} measurements")
        return data
    except Exception as e:
        print(f"Error loading data from {data_file}: {e}")
        raise

# Load synthetic data
try:
    syndata = load_synthetic_data(DATA_PATH, DATASET)
except FileNotFoundError:
    print("\n" + "="*60)
    print("DATA SETUP REQUIRED")
    print("="*60)
    print("The synthetic data files were not found.")
    print("Please ensure you have the required data structure:")
    print(f"\n{PROJECT_ROOT}/synthetic_examples/")
    print("├── data/")
    print("│   ├── XHOLE_HOMO_2D_TT_DATA_c500.sgt")
    print("│   └── schemes/               # Optional")
    print("│       └── scheme.sgt")
    print("├── createdata_homo_xhole.py")
    print("├── createdata_hetero_xhole.py")
    print("└── inversion/")
    print("    └── GI_xhole_2D_tt.py")
    print("\nYou can generate synthetic data using:")
    print("- createdata_homo_xhole.py")
    print("- createdata_hetero_xhole.py")
    print("="*60)
    raise

# %%============================================================================
# GEOLOGICAL MODEL SETUP
# ============================================================================

def create_starting_geological_model():
    """
    Create the starting geological model with fault interface points.
    
    Returns
    -------
    tuple
        (geo_model, initplc, initmap, move_points, fixed_point, all_points)
    """
    # Define starting fault interface points
    fault_points = [[[5, 11], [7.5, 11], [10, 11]], []]
    
    # Model domain parameters
    extent = [-10, 20, 0, 20, 0, 25]
    resolution = [100, 10, 100]
    
    # Create surface interpolation points for 3D model
    sur_int = {}
    surfaces = ['hanging_wall', 'foot_wall']
    fixed_point = [10, 11]  # This point will remain fixed during inversion
    
    for sur, inter in zip(surfaces, fault_points):
        point_list = []
        for y in np.linspace(extent[2] + 3, extent[3] - 3, 3):
            for i in inter:
                point_list.append([i[0], y, i[1]])
        sur_int[sur] = point_list
    
    # Create 2D section for analysis
    section = {'section1': ([-10, 10], [20, 10], [100, 100])}
    
    # Generate GemPy geological model
    geo_model = get_gempy_model.make_gempy_model(
        sur_int, 2, extent, resolution, section, plot=True)
    
    # Extract 2D geometry
    initplc, initmap = get_gempy_model.get_geometry_2d(
        geo_model, 'section1', show=False)
    
    print('Starting orientations:', geo_model.orientations)
    
    # Get surface points for inversion
    dfpoints = gp.get_data(geo_model, 'surface_points')
    subset = dfpoints.index[(dfpoints['Y'] == 10)].tolist()
    
    # Define which points will be moved during inversion (excluding fixed point)
    move_points = subset[:-1]
    n_points = len(move_points)
    
    print(f"Number of moving interface points: {n_points}")
    print(f"Moving point indices: {move_points}")
    
    return geo_model, initplc, initmap, move_points, fixed_point, n_points, sur_int

geo_model, initplc, initmap, move_points, fixed_point, n_points, sur_int = create_starting_geological_model()


# %%============================================================================
# PHYSICAL PROPERTY AND CONSTRAINT SETUP
# ============================================================================

def setup_physical_properties(sur_int):
    """
    Define physical properties for geological units.
    
    Parameters
    ----------
    sur_int : dict
        Dictionary containing surface intersection points for each lithology
    
    Returns
    -------
    tuple
        (paramap, constraints, master_dict)
    """
    # Physical property mapping (velocities for travel time)
    paramap = {}
    paramap['hanging_wall'] = 2100  # m/s
    paramap['foot_wall'] = 2600     # m/s
    
    # Inversion constraints (Z-direction movement only)
    constraints = ['I10']  # Interface constraint
    
    # Create master dictionary with all model information
    def create_master_dict(initmap, paramap, sur_int):
        """
        Create comprehensive model dictionary.
        
        Structure: {lithology: [unit_id, parameter_value, surface_points]}
        """
        md = {}
        for litho, id_val in initmap.items():
            points = sur_int.get(litho)
            md[litho] = [id_val, paramap.get(litho), points]
        return md
    
    # Create master dictionary using passed sur_int
    master_dict = create_master_dict(initmap, paramap, sur_int)
    
    print('Master Dictionary Structure:')
    for unit, data in master_dict.items():
        print(f"  {unit}: [unit_id={data[0]}, velocity={data[1]} m/s, points={len(data[2]) if data[2] else 0}]")
    
    return paramap, constraints, master_dict

# Setup physical properties and constraints
paramap, constraints, master_dict = setup_physical_properties(sur_int)

# %%============================================================================
# MESH CREATION AND PARAMETER MAPPING
# ============================================================================

def create_parameter_vector(mesh, initmap, paramap):
    """
    Create parameter vector for mesh cells based on geological units.
    
    Parameters
    ----------
    mesh : pg.Mesh
        PyGIMLi mesh object
    initmap : dict
        Mapping of geological units to marker IDs
    paramap : dict
        Mapping of geological units to physical properties
        
    Returns
    -------
    numpy.ndarray
        Parameter values for each mesh cell
    """
    cells = []
    for unit_name, param_value in paramap.items():
        unit_id = initmap.get(unit_name)
        temp = [unit_id[0], param_value]
        cells.append(temp)
    
    # Map parameters to mesh cells
    param_vector = pg.solver.parseMapToCellArray(cells, mesh)
    
    print(f"Parameter mapping complete:")
    print(f"  Total mesh cells: {mesh.cellCount()}")
    print(f"  Parameter vector length: {len(param_vector)}")
    
    return param_vector

def create_inversion_mesh():
    """
    Create mesh for inversion including sensor positions.
    
    Returns
    -------
    tuple
        (mesh, parameter_vector, sensors)
    """
    # Get sensor positions from synthetic data
    sensors = syndata.sensorPositions()
    
    # Add sensors to geometry
    for sensor_pos in sensors:
        initplc.createNode(sensor_pos)
        initplc.createNode(sensor_pos - [0, 0.1, 0])  # Add depth nodes
    
    # Create mesh
    mesh = pg.meshtools.createMesh(initplc)
    
    # Set boundary markers
    for boundary in mesh.boundaries():
        if (boundary.center().x() == mesh.xmin() or 
            boundary.center().x() == mesh.xmax() or
            boundary.center().y() == mesh.ymin() or 
            boundary.center().y() == mesh.ymax()):
            boundary.setMarker(0)
    
    # Create parameter vector
    param_vector = create_parameter_vector(mesh, initmap, paramap)
    
    # Visualize mesh with parameters
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _, cb = pg.show(mesh, param_vector, ax=ax, **pg_show_kwargs['tt'])
    ax.scatter(sensors[:, 0], sensors[:, 1])
    ax.set_title('Inversion Mesh with Parameters')
    plt.show()
    
    print(f"Mesh created successfully:")
    print(f"  Nodes: {mesh.nodeCount()}")
    print(f"  Cells: {mesh.cellCount()}")
    print(f"  Sensors: {len(sensors)}")
    
    return mesh, param_vector, sensors

mesh_inv, param_vector, sensors = create_inversion_mesh()

# %%============================================================================
# STARTING MODEL CREATION FOR GEOMETRIC INVERSION
# ============================================================================

def create_starting_model():
    """
    Create the starting model for geometric inversion.
    
    The starting model (sps_geo) contains initial perturbations for the 
    interface points that will be optimized during the geometric inversion.
    These small perturbations provide a starting guess for the fault geometry.
    
    Returns
    -------
    numpy.ndarray
        Starting geometry parameters (sps_geo) - small perturbations for each 
        moving interface point in the Z-direction
    """
    # Create starting geometry parameters (small perturbation for interface points)
    sps_geo = np.zeros(len(move_points)) + 0.1
    
    print('Starting model (sps_geo):', sps_geo)
    print('Length of starting model vector:', len(sps_geo))
    print('Note: Each value represents initial Z-perturbation for interface points')
    
    return sps_geo

# Create starting model - this is required for geometric inversion
sps_geo = create_starting_model()

# %%============================================================================
# GEOMETRIC INVERSION FRAMEWORK INITIALIZATION  
# ============================================================================


def create_forward_operator():
    """
    Create the forward operator for geometric inversion.
    
    Returns
    -------
    GeometricModelling2D
        Forward operator instance
    """
    # Create timestamp for saving results
    now = datetime.now()
    datetime_stamp = now.strftime("%Y%m%d_%H%M%S")
    saving_path = f'results/{METHOD}/{datetime_stamp}/'

    # Create forward operator using sps_geo as starting model
    fop = GeometricInversion2d.GeometricModelling2D(
        saving_path, geo_model, METHOD, 'Z', 'homogenous', 'section1',
        syndata, sps_geo, move_points, master_dict, constraints, 
        [], showInter=True  # Empty list for paracons since we removed pilot points
    )

    print(f"Geometric inversion framework initialized:")
    print(f"  Method: {METHOD}")
    print(f"  Dimension shift: Z")
    print(f"  Moving points: {len(move_points)}")
    print(f"  Saving path: {saving_path}")
    print(f"  Model type: homogenous") 
    print(f"  Starting model length: {len(sps_geo)}")
    
    return fop, saving_path

fop, saving_path = create_forward_operator()

# %%============================================================================
# SENSOR POSITIONS AND VISUALIZATION OF STARTING MODEL
# ============================================================================

def setup_sensor_visualization():
    """
    Setup sensor positions and create visualization of starting model.
    
    Returns
    -------
    tuple
        (receivers, shots, error)
    """
    # Separate receivers and shots based on x-coordinate
    receivers = []
    shots = []
    for sensor_pos in syndata.sensorPositions().array():
        if sensor_pos[0] > 6:
            shots.append(sensor_pos)
        else:
            receivers.append(sensor_pos)
    
    receivers = np.array(receivers)
    shots = np.array(shots)
    
    # Get data error estimates
    error = syndata['err'] / 100
    
    # Create visualization of starting model
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    if METHOD == 'TravelTime':
        _, cb = pg.show(mesh_inv, param_vector, ax=ax, **pg_show_kwargs['tt'])
        ax.scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
        ax.scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    else:
        ax, cb = pg.show(mesh_inv, param_vector, cMap='Spectral_r', cMin=500, cMax=2000)
        ax.scatter(sensors[:, 0], sensors[:, 1], s=50, c='black', label='Sensors')
    
    # Add interface points (need to get points properly)
    def get_interface_points(geo_model):
        """Extract interface points from geological model."""
        x = np.array(geo_model.surface_points.df['X'])
        y = np.array(geo_model.surface_points.df['Y'])
        z = np.array(geo_model.surface_points.df['Z'])
        return x, y, z
    
    points_x, points_y, points_z = get_interface_points(geo_model)
    
    # Show fixed and moving points
    ax.scatter(fixed_point[0], fixed_point[1], **scatter_kwargs['fixed_points'])
    
    actual_points_x = points_x[move_points]
    actual_points_z = points_z[move_points]
    ax.scatter(actual_points_x, actual_points_z, **scatter_kwargs['moving_points'])
    
    # Formatting
    cb.ax.tick_params(labelsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cb.ax.set_xlabel('Velocities (m/s)', fontsize=20)
    
    legend = ax.legend(fontsize=20)
    for text in legend.get_texts():
        text.set_color('white')
    
    ax.set_xlabel('X (m)', fontsize=20)
    ax.set_ylabel('Z (m)', fontsize=20)
    ax.set_title('Starting model', fontsize=25)
    ax.set_xlim([0, 12])
    ax.set_ylim([2.50, 20])
    
    fig.savefig(saving_path + f'/Starting_model_{METHOD}_flat.svg', dpi=300)
    plt.show()
    
    print(f"Sensor setup complete:")
    print(f"  Receivers: {len(receivers)}")
    print(f"  Shots: {len(shots)}")
    print(f"  Moving interface points: {len(move_points)}")
    
    return receivers, shots, error

# Setup sensor positions and visualization
receivers, shots, error = setup_sensor_visualization()

# %%============================================================================
# INVERSION SETUP AND EXECUTION
# ============================================================================

def setup_inversion():
    """
    Setup and configure the inversion framework.
    
    Returns
    -------
    pg.frameworks.Inversion
        Configured inversion object
    """
    # Configure PyGIMLi settings
    pg.setThreadCount(8)
    pg.setLogLevel(0)
    
    # Create inversion framework
    inv = pg.frameworks.Inversion(fop, verbose=True)
    
    # Set regularization parameter (lambda)
    inv._inv.setLambda(LAMBDA)
    inv.minDPhi = -10
    
    # Configure transformations
    trans_linear = pg.trans.Trans()  # Linear transformation for geometry
    trans_log = pg.trans.TransLog()  # Log transformation for data
    
    inv.fop.modelTrans = trans_linear
    inv.fop.dataTrans = trans_log
    
    # Set inversion parameters
    inv.maxIter = 10
    inv.stopAtChi1 = True
    
    # Create constraints
    fop.createConstraints()
    
    print(f"Inversion setup complete:")
    print(f"  Lambda: {LAMBDA}")
    print(f"  Max iterations: {inv.maxIter}")
    print(f"  Stop at chi2=1: {inv.stopAtChi1}")
    print(f"  Model transformation: Linear")
    print(f"  Data transformation: Log")
    
    return inv

inv = setup_inversion()

# %%============================================================================
# MISFIT TRACKING AND VISUALIZATION
# ============================================================================

def create_misfit_callback():
    """
    Create callback function to track and visualize misfit during inversion.
    
    Returns
    -------
    function
        Misfit callback function
    """
    misfits = []
    
    def save_misfit(iteration, inversion_obj):
        """
        Save and visualize misfit at each iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        inversion_obj : object
            Inversion object containing response and data
        """
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        
        # Calculate misfit
        d_predicted = inversion_obj.response
        d_observed = inversion_obj.dataVals
        misfit = d_observed - d_predicted
        misfits.append(misfit)
        
        # Plot misfit as matrix
        gci = pg.viewer.mpl.dataview.drawVecMatrix(
            ax[1], inversion_obj.fop.scheme["g"], inversion_obj.fop.scheme["s"],
            misfit, cMin=-0.0002, cMax=0.0002, cMap='coolwarm'
        )
        
        # Plot current model
        if iteration == 0:
            pg.show(fop.meshes_dict[iteration], fop.param_dict[iteration], 
                   ax=ax[0], **pg_show_kwargs['tt'])
        else:
            pg.show(fop.meshes_dict[iteration-1], fop.param_dict[iteration-1], 
                   ax=ax[0], **pg_show_kwargs['tt'])
        
        # Add sensor and point markers
        ax[0].scatter(fop.shots[:, 0], fop.shots[:, 1], **scatter_kwargs['shots'])
        ax[0].scatter(fop.receivers[:, 0], fop.receivers[:, 1], **scatter_kwargs['receivers'])
        ax[0].scatter(fop.pX[fop.move_points], 
                     fop.pZ[fop.move_points] + inversion_obj.model, 
                     **scatter_kwargs['moving_points'])
        ax[0].scatter(fixed_point[0], fixed_point[1], **scatter_kwargs['fixed_points'])
        ax[0].scatter(fop.pX[fop.move_points], fop.pZ[fop.move_points], 
                     label='Starting points', c='limegreen', s=30)
        
        # Set titles and labels
        ax[1].set_title(f'CHI2 of {np.round(inversion_obj.chi2(), 3)}')
        ax[0].set_title(f'Model at iteration {iteration}')
        
        # Save figure
        fig.savefig(fop.path + f'/images/{iteration}_iteration.png', dpi=300)
        plt.show()
        
        print(f"Iteration {iteration}: Chi2 = {inversion_obj.chi2():.3f}")
    
    return save_misfit, misfits

# Create misfit callback function
save_misfit_callback, misfits = create_misfit_callback()

# %%============================================================================
# INVERSION EXECUTION
# ============================================================================

def run_geometric_inversion():
    """
    Execute the geometric inversion and save results.
    
    Returns
    -------
    tuple
        (result, chi2_history)
    """
    # Set callback for misfit tracking
    inv.setPostStep(save_misfit_callback)
    
    print("Starting geometric inversion...")
    print(f"  Initial data: {len(syndata['t'])} travel time measurements")
    print(f"  Lambda: {LAMBDA}")
    
    # Run inversion
    if METHOD == 'TravelTime':
        result = inv.run(syndata["t"], lam=LAMBDA, showprogress=True, style=None)
    else:
        result = inv.run(syndata["rhoa"], lam=LAMBDA, showprogress=True, style=None)
    
    # Get chi2 history
    chi2_history = inv.chi2History
    
    # Plot chi2 evolution
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(chi2_history, 'o-')
    ax.set_title(f'$\\chi^2$ History - Final $\\chi^2$: {np.round(chi2_history[-1], 2)}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\\chi^2$')
    fig.savefig(fop.path + '/images/chi2History.jpg', dpi=300)
    plt.show()
    
    # Save results
    np.save(fop.path + '/chi2.npy', chi2_history)
    np.save(fop.path + '/inversion_result.npy', result)
    
    print(f"Inversion completed:")
    print(f"  Final chi2: {chi2_history[-1]:.3f}")
    print(f"  Iterations: {len(chi2_history)}")
    print(f"  Results saved to: {fop.path}")
    
    return result, chi2_history

result, chi2_history = run_geometric_inversion()

# %%============================================================================
# RESULTS VISUALIZATION AND COMPARISON
# ============================================================================

def visualize_results():
    """
    Create comprehensive visualization of inversion results.
    """
    # Define true model points for comparison (if available)
    truemodel_points_tt = [[3, 14.5], [4, 14.0], [5, 13.5], [6, 13], 
                          [7, 12.5], [8, 12], [9, 11.5], [10, 11]]
    truemodel_points = np.stack(truemodel_points_tt)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Starting model
    pg.show(mesh_inv, param_vector, ax=axes[0], **pg_show_kwargs['tt'])
    axes[0].scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    axes[0].scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
    axes[0].scatter(fixed_point[0], fixed_point[1], **scatter_kwargs['fixed_points'])
    
    # Get initial interface points
    points_x, points_y, points_z = get_interface_points(geo_model)
    axes[0].scatter(points_x[move_points], points_z[move_points], 
                   label='Initial interface', c='limegreen', s=50)
    axes[0].set_title('Starting Model')
    axes[0].legend()
    
    # Plot 2: Final model (need to get this from fop)
    if hasattr(fop, 'meshes_dict') and len(fop.meshes_dict) > 0:
        final_key = max(fop.meshes_dict.keys())
        pg.show(fop.meshes_dict[final_key], fop.param_dict[final_key], 
               ax=axes[1], **pg_show_kwargs['tt'])
    else:
        pg.show(mesh_inv, param_vector, ax=axes[1], **pg_show_kwargs['tt'])
    
    axes[1].scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    axes[1].scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
    axes[1].scatter(fixed_point[0], fixed_point[1], **scatter_kwargs['fixed_points'])
    
    # Plot final interface points (initial + result perturbation)
    final_z_positions = points_z[move_points] + result
    axes[1].scatter(points_x[move_points], final_z_positions, 
                   label='Final interface', c='red', s=50)
    axes[1].set_title('Final Model')
    axes[1].legend()
    
    # Plot 3: Comparison with true model (if available)
    pg.show(mesh_inv, param_vector, ax=axes[2], **pg_show_kwargs['tt'])
    axes[2].scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    axes[2].scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
    axes[2].scatter(points_x[move_points], points_z[move_points], 
                   label='Initial', c='limegreen', s=50)
    axes[2].scatter(points_x[move_points], final_z_positions, 
                   label='Inverted', c='red', s=50)
    axes[2].scatter(truemodel_points[:, 0], truemodel_points[:, 1], 
                   label='True model', c='blue', s=50, marker='s')
    axes[2].set_title('Model Comparison')
    axes[2].legend()
    
    # Format all plots
    for ax in axes:
        ax.set_xlim([0, 12])
        ax.set_ylim([2.5, 20])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
    
    plt.tight_layout()
    fig.savefig(fop.path + '/final_comparison.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\nInversion Summary:")
    print(f"  Interface point movement (mean): {np.mean(result):.3f} m")
    print(f"  Interface point movement (std): {np.std(result):.3f} m")
    print(f"  Maximum movement: {np.max(np.abs(result)):.3f} m")
    print(f"  Final chi2: {chi2_history[-1]:.3f}")

# Helper function to get interface points
def get_interface_points(geo_model):
    """Extract interface points from geological model."""
    x = np.array(geo_model.surface_points.df['X'])
    y = np.array(geo_model.surface_points.df['Y']) 
    z = np.array(geo_model.surface_points.df['Z'])
    return x, y, z

# Run visualization
visualize_results()

print("\n" + "="*60)
print("GEOMETRIC INVERSION COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Results saved to: {saving_path}")
print(f"Final chi2: {chi2_history[-1]:.3f}")
print(f"Total iterations: {len(chi2_history)}")
print("="*60)



