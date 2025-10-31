"""
Synthetic Cross-hole Heterogeneous Data Generation and Inversion
===============================================================

This script generates synthetic cross-hole geophysical data (ERT and Travel Time) 
for a heterogeneous two-layer geological model with spatially varying parameters.

Author: Andrea
Date: October 2025
"""

# %% IMPORTS AND SETUP
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.meshtools import mesh
import pygimli.physics.ert as ert
import pygimli.physics.traveltime as tt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pygimli.utils import sparseMatrix2coo
from pygimli.viewer.mpl import createColorBar
import sys
import os

# Add project root to Python path for module imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import custom modules
from utilsGeo import get_gempy_model as ggm
from utilsGeo import utils
from utilsGeo.plotting_params import scatter_kwargs, pg_show_kwargs, set_style
import gempy as gp

# Set plotting style
set_style(20)
plt.rcParams.update({'font.size': 25})

# Use relative path for output data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
synthetic_path = os.path.join(os.path.dirname(__file__), 'data/')

# %%============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_parameter_grid(plc, np_x=20, np_y=20):
    """
    Create a regular grid for parameter interpolation.
    
    Parameters
    ----------
    plc : pygimli.Mesh
        Geological model mesh boundary
    np_x : int, optional
        Number of x-coordinates, by default 20
    np_y : int, optional
        Number of y-coordinates, by default 20
        
    Returns
    -------
    tuple
        (grid, np_all) - Parameter grid and total number of points
    """
    np_all = np_x * np_y
    
    # Create grid exactly as in original script
    x = np.linspace(plc.xMin(), plc.xMax(), np_x + 1)
    y = np.linspace(plc.yMin(), plc.yMax(), np_y + 1)
    grid = mt.createGrid(x=x, y=y)
    
    print(f"Grid bounds: x=[{plc.xMin():.2f}, {plc.xMax():.2f}], y=[{plc.yMin():.2f}, {plc.yMax():.2f}]")
    print(f"Grid size: {len(x)}x{len(y)} = {np_all} points")
    
    return grid, np_all


def generate_heterogeneous_ert_parameters(np_all, seed=28):
    """
    Generate heterogeneous resistivity parameters for ERT simulation.
    
    Parameters
    ----------
    np_all : int
        Total number of parameter points
    seed : int, optional
        Random seed for reproducibility, by default 28
        
    Returns
    -------
    tuple
        (para_cz, para_br) - Parameter arrays for hanging wall and foot wall
    """
    np.random.seed(seed)
    # Hanging wall: lower resistivities
    para_cz = np.random.uniform(low=100, high=500, size=np_all)
    # Foot wall: higher resistivities  
    para_br = np.random.uniform(low=1000, high=5000, size=np_all)
    return para_cz, para_br


def generate_heterogeneous_tt_parameters(np_all, seed=28):
    """
    Generate heterogeneous velocity parameters for travel time simulation.
    
    Parameters
    ----------
    np_all : int
        Total number of parameter points
    seed : int, optional
        Random seed for reproducibility, by default 28
        
    Returns
    -------
    tuple
        (para_cz, para_br) - Velocity arrays for hanging wall and foot wall
    """
    # Velocity ranges matching original script
    br_low = 1700
    br_high = 2200
    cz_low = 2800
    cz_high = 3200
    
    np.random.seed(seed)
    # Hanging wall: higher velocities
    para_cz = np.random.uniform(low=cz_low, high=cz_high, size=np_all)
    # Foot wall: lower velocities
    para_br = np.random.uniform(low=br_low, high=br_high, size=np_all)
    return para_cz, para_br


def interpolate_parameters_to_mesh(mesh, grid, para_cz, para_br):
    """
    Interpolate heterogeneous parameters from grid to mesh cells.
    
    Parameters
    ----------
    mesh : pygimli.Mesh
        Target mesh for parameter assignment
    grid : pygimli.Mesh
        Parameter grid
    para_cz : numpy.array
        Parameter values for hanging wall (marker 2)
    para_br : numpy.array
        Parameter values for foot wall (marker 1)
        
    Returns
    -------
    tuple
        (all_values, paramesh_cz, paramesh_br) - Full parameter array and individual layers
    """
    cellmarkers = mesh.cellMarkers()
    
    # Interpolate parameters to mesh - using mt.interpolate as in original
    paramesh_cz_full = mt.interpolate(mesh, grid, para_cz)
    paramesh_br_full = mt.interpolate(mesh, grid, para_br)
    
    # Assign parameters based on geological unit markers (exactly as original)
    all_values = np.where(cellmarkers == 2, paramesh_cz_full, paramesh_br_full)
    
    # Extract parameters for each layer for visualization (exactly as original)
    paramesh_cz = paramesh_cz_full[cellmarkers == 2]
    paramesh_br = paramesh_br_full[cellmarkers == 1]
    
    return all_values, paramesh_cz, paramesh_br

# %%============================================================================
# GEOLOGICAL MODEL SETUP
# ============================================================================

def create_geological_model():
    """
    Create a simple fault-based geological model using GemPy.
    
    Returns
    -------
    tuple
        (geo_model, plc, map, truemodel_points)
    """
    # Define fault points
    fault_points = [[[3,14.5], [4,14.0], [5,13.5], [6,13], [7,12.5], 
                     [8,12], [9,11.5], [10,11]], []]
    
    # Model extent and resolution
    extent = [-10, 20, 0, 20, 0, 25]
    resolution = [100, 10, 100]
    
    # Create surface interpolation points
    sur_int = {}
    surfaces = ['hanging_wall', 'foot_wall']
    truemodel_points = np.stack(fault_points[0])
    
    for sur, inter in zip(surfaces, fault_points):
        point_list = []
        for y in np.linspace(extent[2]+3, extent[3]-3, 3):
            for i in inter:
                point_list.append([i[0], y, i[1]])
        sur_int[sur] = point_list
    
    # Create 2D section
    section = {'section1': ([-10,10], [20,10], [100,100])}
    
    # Generate GemPy model
    geo_model = ggm.make_gempy_model(sur_int, 2, extent, resolution, section, plot=True)
    plc, map_data = ggm.get_geometry_2d(geo_model, 'section1', show=True)
    
    return geo_model, plc, map_data, truemodel_points



# %%============================================================================
# SENSOR ARRAY SETUP
# ============================================================================

def create_crosshole_sensors():
    """
    Create cross-hole sensor arrays for both wells.
    
    Returns
    -------
    numpy.ndarray
        Combined sensor positions for both wells
    """
    # Well parameters
    w1_x = 5
    w2_x = 10
    electrode_spacing = 0.5
    y_values = np.arange(5, 20, electrode_spacing)
    
    # Create well positions
    x_w1 = np.ones_like(y_values) * w1_x
    x_w2 = np.ones_like(y_values) * w2_x
    w1 = np.column_stack((x_w1, y_values))
    w2 = np.column_stack((x_w2, y_values))
    
    return np.vstack((w1, w2))



# %%============================================================================
# TRAVEL TIME DATA FUNCTIONS
# ============================================================================

def setup_tt_sensors_and_scheme():
    """
    Set up travel time sensors and measurement scheme.
    
    Returns
    -------
    tuple
        (tt_scheme, sensors_tt) - Modified scheme and sensor positions
        
    Notes
    -----
    If no external scheme file is available, uses a fallback approach.
    """
    # Check for scheme file in data directory (relative to script location)
    scheme_file = 'data/scheme.sgt'
    
    print(f"Looking for scheme file...")
    print(f"  Expected scheme file: {scheme_file}")
    
    # Check if directories exist
    if os.path.exists('data'):
        print(f"✓ Data directory exists")
    else:
        print(f"✗ Data directory does not exist")
    
    # Try to load the scheme file
    if os.path.exists(scheme_file):
        try:
            print(f"✓ Scheme file found! Loading from: {scheme_file}")
            tt_scheme = pg.load(scheme_file)
            print(f"✓ Successfully loaded scheme with {tt_scheme.size()} measurements")
        except Exception as e:
            print(f"✗ Error loading scheme file: {e}")
            print("Using fallback approach...")
            tt_scheme = pg.DataContainer()  # Fallback
    else:
        print(f"✗ No scheme file found at: {scheme_file}")
        print("Using fallback approach...")
        tt_scheme = pg.DataContainer()  # Fallback
        print(f"  To use a custom scheme, place it at: {scheme_file}")
    
    # Handle scheme loading
    if hasattr(tt_scheme, 'sensorPositions') and tt_scheme.size() > 0:
        tt_sensors = tt_scheme.sensorPositions().array()
        print(f"✓ Using scheme sensors: {len(tt_sensors)} sensor positions")
    else:
        print("⚠ Creating default sensor configuration...")
        # Default simple sensor array (users should modify this)
        tt_sensors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])  # Placeholder
        print(f"  Using {len(tt_sensors)} default sensor positions")
    
    x = tt_sensors[:,0]
    y = tt_sensors[:,1]
    z = np.zeros_like(x)
    
    # Generate complete gun-source pairs
    guns = np.unique(tt_scheme['g'])
    sources = np.unique(tt_scheme['s'])
    complete_pairs = list(itertools.product(guns, sources))
    complete_pairs = np.array(complete_pairs)
    
    print(f"Total possible gun-source pairs: {len(complete_pairs)}")
    
    # Update scheme with complete pairs
    tt_scheme['g'] = complete_pairs[:,0]
    tt_scheme['s'] = complete_pairs[:,1]
    
    # Create sensor array
    sensors_tt = np.column_stack((x, y, z))
    tt_scheme.setSensorPositions(sensors_tt)
    
    return tt_scheme, sensors_tt


def simulate_tt_data_hetero(mesh_TT, tt_scheme, para_cz, para_br, grid):
    """
    Simulate travel time data for heterogeneous model.
    
    Parameters
    ----------
    mesh_TT : pygimli.Mesh
        Mesh for travel time simulation
    tt_scheme : pg.DataContainer
        Travel time measurement scheme
    para_cz : numpy.array
        Hanging wall velocity parameters
    para_br : numpy.array
        Foot wall velocity parameters
    grid : pygimli.Mesh
        Parameter grid for interpolation
        
    Returns
    -------
    tuple
        (data_tt_hetero, all_values_vel, paramesh_cz, paramesh_br) - Simulated data and velocity arrays
    """
    # Interpolate parameters to mesh
    all_values_vel, paramesh_cz, paramesh_br = interpolate_parameters_to_mesh(
        mesh_TT, grid, para_cz, para_br)
    
    print(f"Interpolated velocity range: {all_values_vel.min():.1f} - {all_values_vel.max():.1f}")
    
    # Simulate travel time data
    data_tt_hetero = tt.simulate(
        mesh=mesh_TT, 
        scheme=tt_scheme, 
        slowness=1./all_values_vel,
        noiseLevel=0.001, 
        noiseAbs=1e-5, 
        seed=1337
    )
    
    return data_tt_hetero, all_values_vel, paramesh_cz, paramesh_br

# %%============================================================================
# ERT DATA FUNCTIONS
# ============================================================================

def simulate_ert_data_hetero(mesh_ERT, scheme, para_cz, para_br, grid):
    """
    Simulate ERT data for heterogeneous model.
    
    Parameters
    ----------
    mesh_ERT : pygimli.Mesh
        Mesh for ERT simulation
    scheme : pg.DataContainer
        ERT measurement scheme
    para_cz : numpy.array
        Hanging wall resistivity parameters
    para_br : numpy.array
        Foot wall resistivity parameters
    grid : pygimli.Mesh
        Parameter grid for interpolation
        
    Returns
    -------
    tuple
        (syndata, all_values, paramesh_cz, paramesh_br) - Simulated data and parameter arrays
    """
    
    # Interpolate parameters to mesh
    all_values, paramesh_cz, paramesh_br = interpolate_parameters_to_mesh(
        mesh_ERT, grid, para_cz, para_br)
    
    print(f"Interpolated resistivity range: {all_values.min():.1f} - {all_values.max():.1f}")
    
    # Simulate ERT data
    syndata = ert.simulate(
        mesh_ERT, 
        scheme=scheme, 
        res=all_values, 
        noiseLevel=1, 
        seed=1337
    )
    
    # Remove negative apparent resistivity values
    syndata.remove(syndata['rhoa'] < 0)
    
    return syndata, all_values, paramesh_cz, paramesh_br




# %%============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_heterogeneous_ert_model(mesh_ERT, paramesh_cz, paramesh_br, sensors, 
                                 truemodel_points, synthetic_path):
    """
    Plot heterogeneous ERT resistivity model with dual colorbars.
    
    Parameters
    ----------
    mesh_ERT : pygimli.Mesh
        ERT mesh
    paramesh_cz : numpy.array
        Hanging wall resistivity parameters
    paramesh_br : numpy.array
        Foot wall resistivity parameters
    sensors : numpy.array
        Sensor positions
    truemodel_points : numpy.array
        True fault positions
    synthetic_path : str
        Path to save figure
    """
    cellmarkers = mesh_ERT.cellMarkers()
    
    fig, ax = plt.subplots(figsize=(15, 10), tight_layout=True)
    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="5%", pad=0.50)
    cax_left.yaxis.set_ticks_position('left')
    cax_right = divider.append_axes("right", size="5%", pad=0.15)
    
    # Create submeshes for each layer
    mesh_1 = mesh_ERT.createSubMesh(mesh_ERT.cells(cellmarkers == 1))
    mesh_2 = mesh_ERT.createSubMesh(mesh_ERT.cells(cellmarkers == 2))
    
    # Plot each layer with different colormaps
    pg.show(mesh_1, paramesh_br, ax=ax, cMap='magma', cmin=1000, cmax=5000, colorBar=False)
    pg.show(mesh_2, paramesh_cz, ax=ax, cMap='viridis', cmin=100, cmax=500, colorBar=False)
    
    # Create colorbars
    cb_left = pg.viewer.mpl.colorbar.createColorBarOnly(
        cMin=100, cMax=500, cMap='viridis', ax=cax_left, orientation='vertical')
    cb_right = pg.viewer.mpl.colorbar.createColorBarOnly(
        cMin=1000, cMax=5000, cMap='magma', ax=cax_right, orientation='vertical')
    
    # Customize colorbars
    cb_left.set_ylabel(f'Hanging wall {pg.unit("res")}', rotation=90, labelpad=20)
    cb_right.set_ylabel(f'Foot wall {pg.unit("res")}', rotation=90, labelpad=25)
    cb_left.yaxis.set_ticks_position('left')
    cb_left.yaxis.set_label_position('left')
    
    # Set ticks and labels
    ax.set_xticks([0,2,4,6,8,10,12,14])
    ax.set_yticks([0,5,10,15,20])
    cax_left.set_yticks([100, 200, 300, 400, 500])
    cax_right.set_yticks([1000, 2000, 3000, 4000, 5000])
    
    # Set limits and labels
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    
    # Add sensors and fault points
    ax.scatter(sensors[:,0], sensors[:,1], s=10, c='black', marker='D', label='Sensors')
    ax.scatter(truemodel_points[:,0], truemodel_points[:,1], 
               s=100, c='black', label='True fault')
    
    ax.legend()
    fig.savefig(synthetic_path + 'ERT_HeteroModel.png')
    return fig, ax


def plot_heterogeneous_tt_model(mesh_TT, paramesh_cz, paramesh_br, shots, 
                                receivers, truemodel_points, synthetic_path):
    """
    Plot heterogeneous travel time velocity model with dual colorbars.
    
    Parameters
    ----------
    mesh_TT : pygimli.Mesh
        Travel time mesh
    paramesh_cz : numpy.array
        Hanging wall velocity parameters
    paramesh_br : numpy.array
        Foot wall velocity parameters
    shots : numpy.array
        Shot positions
    receivers : numpy.array
        Receiver positions
    truemodel_points : numpy.array
        True fault positions
    synthetic_path : str
        Path to save figure
    """
    cellmarkers = mesh_TT.cellMarkers()
    cz_low, cz_high = 2800, 3200
    br_low, br_high = 1700, 2200
    
    fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="5%", pad=0.50)
    cax_left.yaxis.set_ticks_position('left')
    cax_right = divider.append_axes("right", size="5%", pad=0.15)
    
    # Create submeshes for each layer
    mesh_1 = mesh_TT.createSubMesh(mesh_TT.cells(cellmarkers == 1))
    mesh_2 = mesh_TT.createSubMesh(mesh_TT.cells(cellmarkers == 2))
    
    # Plot each layer with different colormaps
    pg.show(mesh_2, paramesh_cz, ax=ax, cMap='viridis', cmin=cz_low, cmax=cz_high, colorBar=False)
    pg.show(mesh_1, paramesh_br, ax=ax, cMap='coolwarm', cmin=br_low, cmax=br_high, colorBar=False)
    
    # Create colorbars
    cb_left = pg.viewer.mpl.colorbar.createColorBarOnly(
        cMin=cz_low, cMax=cz_high, cMap='viridis', ax=cax_left, orientation='vertical')
    cb_right = pg.viewer.mpl.colorbar.createColorBarOnly(
        cMin=br_low, cMax=br_high, cMap='coolwarm', ax=cax_right, orientation='vertical')
    
    # Customize colorbars
    cb_left.set_ylabel(f'Hanging wall {pg.unit("vel")}', rotation=90, labelpad=20, fontsize=25)
    cb_right.set_ylabel(f'Foot wall {pg.unit("vel")}', rotation=90, labelpad=25, fontsize=25)
    cb_left.yaxis.set_ticks_position('left')
    cb_left.yaxis.set_label_position('left')
    cb_left.tick_params(labelsize=20)
    cb_right.tick_params(labelsize=20)
    
    # Set ticks and labels
    ax.set_xticks([0,2,4,6,8,10,12,14])
    ax.set_yticks([0,5,10,15,20])
    ax.set_xlim(0, 13)
    ax.set_ylim(4, 18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('X (m)', fontsize=15)
    ax.set_ylabel('Z (m)', fontsize=15)
    ax.set_title('Heterogeneous - True model', fontsize=28)
    
    # Add sensors and fault points
    ax.scatter(truemodel_points[:,0], truemodel_points[:,1], s=100, c='black', label='True model')
    ax.scatter(shots[:,0], shots[:,1], **scatter_kwargs['shots'])
    ax.scatter(receivers[:,0], receivers[:,1], **scatter_kwargs['receivers'])
    
    legend = ax.legend(fontsize=17)
    for text in legend.get_texts():
        text.set_color("white")
        
    fig.savefig(synthetic_path + 'TT_HeteroModel.svg')
    return fig, ax


def showVA(data, usePos=True, ax=None, **kwargs):
    """
    Show apparent velocity as image plot.
    
    Parameters
    ----------
    data : pg.DataContainer
        Data container with 's', 'g' sensor indices and 't' traveltimes
    usePos : bool, optional
        Use sensor positions for axes labels, by default True
    ax : matplotlib.Axes, optional
        Axes to plot on, by default None
        
    Returns
    -------
    tuple
        (ax, colorbar) - Axes and colorbar objects
    """
    ax, _ = pg.show(ax=ax)
    gci = drawVA(ax, data=data, usePos=usePos, **kwargs)
    cBar = createColorBar(gci, **kwargs)
    return ax, cBar


def drawVA(ax, data, vals=None, usePos=True, pseudosection=False, **kwargs):
    """
    Draw apparent velocities as matrix into an axis.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        Target axes
    data : pg.DataContainer
        Data container with sensor information
    vals : array-like, optional
        Travel times, by default None (uses data['t'])
    usePos : bool, optional
        Use sensor positions for tick labels, by default True
    pseudosection : bool, optional
        Show in pseudosection style, by default False
        
    Returns
    -------
    matplotlib.image.AxesImage
        Image object for colorbar creation
    """
    if isinstance(vals, str):
        vals = data(vals)
    
    if vals is None:
        vals = data('t')
    
    # Separate receivers and shots based on x-coordinate
    receivers = []
    shots = []
    for x in data.sensorPositions():
        if x[0] > 6:
            shots.append(x)
        else:
            receivers.append(x)
    
    shots = np.array(shots)
    receivers = np.array(receivers)
    
    # Calculate offset and apparent velocity
    offset = tt.utils.shotReceiverDistances(data, full=True)
    
    if min(vals) < 1e-10:
        print("Warning: zero traveltimes found.")
        print(vals)
    
    va = offset / vals
    
    if pseudosection:
        gx = np.asarray(pg.x(receivers))
        sx = np.asarray(pg.x(shots))
        midpoint = (gx + sx) / 2
        gci = pg.viewer.mpl.dataview.drawVecMatrix(
            ax, midpoint, offset, va, squeeze=True, label=pg.unit('as'))
    else:
        # Get y-positions for plotting
        datag_pos = [data.sensorPositions().array()[int(index)][1] for index in data['g']]
        datas_pos = [data.sensorPositions().array()[int(index)][1] for index in data['s']]
        
        gpos = np.array(datag_pos)
        spos = np.array(datas_pos)
        
        gci = pg.viewer.mpl.dataview.drawVecMatrix(
            ax, gpos, spos, va, squeeze=True, label=pg.unit('as'), **kwargs)
    
    if usePos:
        sy = np.asarray(pg.y(shots))
        gy = np.asarray(pg.y(receivers))
        
        # Set tick labels
        y_labels = np.linspace(np.min(sy), np.max(sy), 5)
        x_labels = np.linspace(np.min(gy), np.max(gy), 5)
        
        ax.set_xticks(x_labels)
        ax.set_yticks(y_labels)
        ax.set_xticklabels([f"{int(i)}" for i in x_labels])
        ax.set_yticklabels([f"{int(i)}" for i in y_labels])
    
    return gci




# %%============================================================================
# MAIN EXECUTION BLOCKS
# ============================================================================

# %% GEOLOGICAL MODEL CREATION
print("Creating geological model...")
geo_model, plc, map_data, truemodel_points = create_geological_model()

# %% PARAMETER GRID SETUP
print("\nSetting up parameter grid...")
# Grid parameters for heterogeneous modeling
np_x = 20  # Number of x-coordinates
np_y = 20  # Number of z-coordinates
grid, np_all = create_parameter_grid(plc, np_x, np_y)

print(f"Parameter grid created with {np_all} points ({np_x}x{np_y})")

# %% SENSOR SETUP
print("\nSetting up sensors...")
sensors = create_crosshole_sensors()

# Visualize sensors on geological model
ax, _ = pg.show(plc, label='paravec', alpha=0.3, hold=True)
ax.scatter(sensors[:, 0], sensors[:, 1], c='red', s=100)
plt.title('Cross-hole sensor array')
plt.show()

# %% ERT SETUP AND SIMULATION
print("\nSetting up ERT measurement scheme...")

# Create ERT measurement scheme
n = 30
xscheme = utils.create2Dxhconfs(n, ds=[4,2], bipole=True)
scheme = pg.DataContainerERT()

for i in sensors:
    scheme.createSensor(i)

for measurement in xscheme:
    scheme.addFourPointData(*measurement)

print(f"ERT scheme created with {scheme.size()} measurements")

# Create ERT mesh
for i in scheme.sensors():
    plc.createNode(i)
    plc.createNode(i - [0, 0.1])

mesh_ERT = mt.createMesh(plc, quality=31, area=100)

# Simulate heterogeneous ERT data
print("Simulating heterogeneous ERT data...")

# First generate and visualize the grid parameters (as in original)
para_cz_ert, para_br_ert = generate_heterogeneous_ert_parameters(np_all)
print("Showing ERT parameter distributions on grid...")
pg.show(grid, para_cz_ert, label='para_cz', cMap='Spectral_r')
pg.show(grid, para_br_ert, label='para_br', cMap='Spectral_r')

syndata, res_all_values, res_paramesh_cz, res_paramesh_br = simulate_ert_data_hetero(
    mesh_ERT, scheme, para_cz_ert, para_br_ert, grid)

print(f"ERT data simulated with {syndata.size()} valid measurements")

# %% TRAVEL TIME SETUP AND SIMULATION
print("\nSetting up travel time sensors and simulating data...")

# Setup travel time scheme
tt_scheme, sensors_tt = setup_tt_sensors_and_scheme()

# Separate shots and receivers for plotting
receivers = []
shots = []
for x in sensors_tt:
    if x[0] > 6:
        shots.append(x)
    else:
        receivers.append(x)
shots = np.array(shots)
receivers = np.array(receivers)

# Create travel time mesh
ttplc = plc
for i in sensors_tt:
    ttplc.createNode(i)

mesh_TT = mt.createMesh(ttplc, quality=31)

# Simulate heterogeneous travel time data
print("Simulating heterogeneous travel time data...")

# First generate and visualize the grid parameters (as in original)
para_cz_tt, para_br_tt = generate_heterogeneous_tt_parameters(np_all)
print("Showing parameter distributions on grid...")
pg.show(grid, para_cz_tt, label='para_cz', cMap='Spectral_r')
pg.show(grid, para_br_tt, label='para_br', cMap='Spectral_r')

data_tt_hetero, vel_all_values, vel_paramesh_cz, vel_paramesh_br = simulate_tt_data_hetero(
    mesh_TT, tt_scheme, para_cz_tt, para_br_tt, grid)

print(f"Travel time data simulated with {data_tt_hetero.size()} measurements")

# %% DATA VISUALIZATION AND SAVING
print("\nVisualizing and saving synthetic data...")

# Plot heterogeneous ERT model
print("Plotting heterogeneous ERT model...")
plot_heterogeneous_ert_model(mesh_ERT, res_paramesh_cz, res_paramesh_br, 
                             sensors, truemodel_points, synthetic_path)

# Plot ERT data
ax, _ = pg.show(syndata, label='Apparent Resistivities (ohm-m)')
ax.set_title('ERT data - heterogeneous')
fig = ax.get_figure()
fig.savefig(synthetic_path + 'ERT_hetero_data.png')
plt.show()

# Plot heterogeneous travel time model
print("Plotting heterogeneous travel time model...")
plot_heterogeneous_tt_model(mesh_TT, vel_paramesh_cz, vel_paramesh_br, 
                            shots, receivers, truemodel_points, synthetic_path)

# Plot travel time data
ax, cb = tt.showVA(data_tt_hetero, usePos=False, cMap='viridis')
ax.set_title('Travel time data - heterogeneous', fontsize=25)
ax.tick_params(labelsize=20)
cb.ax.tick_params(labelsize=20)
cb.ax.set_xlabel('Velocity (m/s)', fontsize=25)
ax.set_xlabel('Receivers', fontsize=20)
ax.set_ylabel('Sources', fontsize=20)
fig = ax.get_figure()
fig.savefig(synthetic_path + 'TT_hetero_data.svg')
plt.show()

# Save ERT data and model
print("Saving ERT data and model...")
resname_ert = 'XHOLE_HETERO_2D_ERT'
mesh_ERT.save(synthetic_path + f'{resname_ert}_MESH')
np.savez(synthetic_path + f'{resname_ert}_RESMAP', res=res_all_values)
syndata.save(synthetic_path + f'{resname_ert}_DATA')

# Save travel time data and model
print("Saving travel time data and model...")
resname_tt = 'XHOLE_HETERO_2D_TT'
mesh_TT.save(synthetic_path + f'{resname_tt}_MESH')
np.savez(synthetic_path + f'{resname_tt}_VELMAP', res=vel_all_values)
data_tt_hetero.save(synthetic_path + f'{resname_tt}_DATA.sgt')

# %% ADDITIONAL GEOLOGICAL MODEL PLOTS
print("\nCreating additional geological model plots...")

# Plot 2D section with GemPy
fig = gp.plot_2d(geo_model, figsize=(12,10))
fig.axes[0].tick_params(axis='both', which='major', labelsize=25)
fig.axes[0].set_xlabel('X(m)', fontsize=25)
fig.axes[0].set_ylabel('Z(m)', fontsize=25)
plt.show()

# Plot geological structure (PLC)
fig, ax = plt.subplots(figsize=(12,10))
pg.show(plc, ax=ax, markers=False)
ax.set_xlabel('X(m)', fontsize=25)
ax.set_ylabel('Z(m)', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)
plt.tight_layout()
plt.show()

# %% INVERSION EXAMPLES
print("\nRunning inversion examples...")

# ERT Inversion
print("Running ERT inversion...")
x, y, _ = syndata.sensorPositions().array().T
bound = 20
rect = mt.createRectangle([x[-1] - bound, y[30] - bound], 
                         [x[0] + bound, y[-1] + bound], marker=1)

for sen in syndata.sensors().array():
    rect.createNode(sen)

mesh_ert_inv = mt.createMesh(rect, quality=34)
mgr_ert = ert.ERTManager(data=syndata, verbose=True)
inv_ert = mgr_ert.invert(syndata, mesh_ert_inv, lam=1000, verbose=True, useGradient=False)

# Plot ERT inversion result
fig, ax = plt.subplots(1, 1, figsize=(7, 9))
mgr_ert.showResult(cMin=500, cMax=3000, ax=ax, showlogScale=True, cMap="Spectral_r", 
                   coverage=mgr_ert.standardizedCoverage())
ax.set_xlim(0, 12)
ax.set_ylim(0, 25)
ax.set_title('ERT Inversion Result - Heterogeneous', fontsize=20)
fig.savefig(synthetic_path + 'RegularInversion_HeteroERT.png')
plt.show()

# Travel Time Inversion
print("Running travel time inversion...")
x, y, _ = data_tt_hetero.sensorPositions().array().T
rect_tt = mt.createRectangle([x[-1] - bound, y[30] - bound], 
                            [x[0] + bound, y[-1] + bound], marker=0)

for sen in data_tt_hetero.sensors().array():
    rect_tt.createNode(sen)

mesh_tt_inv = mt.createMesh(rect_tt, quality=34, area=0.3)
mgr_tt = tt.TravelTimeManager(data=data_tt_hetero, verbose=True)
mgr_tt.setMesh(mesh_tt_inv)
mgr_tt.inv.setRegularization(background=False)
inv_tt = mgr_tt.invert(data_tt_hetero, lam=1000, verbose=True, useGradient=False, zWeight=1.0)

# Plot travel time inversion result
cov = pg.Vector(mgr_tt.standardizedCoverage())
fig, ax = plt.subplots(1, 1, figsize=(7, 9))
mgr_tt.showResult(cMin=1700, cMax=3200, ax=ax, showlogScale=True, cMap="viridis", coverage=cov)
ax.scatter(sensors_tt[:,0], sensors_tt[:,1], s=10, c='black', marker='D', label='Sensors')
ax.set_xlim(0, 12)
ax.set_ylim(4, 18)
ax.set_title('Travel Time Inversion Result - Heterogeneous', fontsize=20)
fig.savefig(synthetic_path + 'RegularInversion_HeteroTT.png')
plt.show()

print("Script completed successfully!")
