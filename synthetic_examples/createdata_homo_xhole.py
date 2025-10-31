# %% 
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.meshtools import mesh
import pygimli.physics.ert as ert
import sys
import os

# Add project root to Python path for module imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import custom modules
from utilsGeo import get_gempy_model as ggm  # nopep8
from utilsGeo import utils
import itertools
from pygimli.utils import sparseMatrix2coo
import pygimli.physics.traveltime as tt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pygimli.viewer.mpl import createColorBar
from utilsGeo.plotting_params import scatter_kwargs, pg_show_kwargs, set_style
set_style(20)

# Use relative path for output data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
synthetic_path = os.path.join(os.path.dirname(__file__), 'data/')

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
    If no external scheme file is available, creates a simple crosshole scheme.
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
            print("Creating simple default scheme...")
            tt_scheme = pg.DataContainer()
    else:
        print(f"✗ No scheme file found at: {scheme_file}")
        print("Creating default crosshole scheme...")
        tt_scheme = pg.DataContainer()
        print(f"  To use a custom scheme, place it at: {scheme_file}")
    
    # Ensure we have a valid scheme
    if not hasattr(tt_scheme, 'sensorPositions') or tt_scheme.size() == 0:
        print("⚠ Creating minimal default scheme (no valid scheme loaded)...")
        # Create a simple default scheme - users should replace this with their actual scheme
        tt_scheme = pg.DataContainer()
    
    tt_sensors = tt_scheme.sensorPositions().array()
    
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


def simulate_tt_data(mesh_TT, tt_scheme, contrast=600):
    """
    Simulate travel time data for homogeneous model.
    
    Parameters
    ----------
    mesh_TT : pygimli.Mesh
        Mesh for travel time simulation
    tt_scheme : pg.DataContainer
        Travel time measurement scheme
    contrast : int, optional
        Velocity contrast between layers, by default 600
        
    Returns
    -------
    tuple
        (data_tt_homo, slo_array_homo) - Simulated data and slowness array
    """
    # Define velocities
    hw_start_vel = 2600
    fw_start_vel = 2000
    
    # Create slowness array
    slo_array_homo = pg.solver.parseMapToCellArray(
        [[1, hw_start_vel], [2, fw_start_vel]], mesh_TT)
    
    # Simulate travel time data
    data_tt_homo = tt.simulate(
        mesh=mesh_TT, 
        scheme=tt_scheme, 
        slowness=1./slo_array_homo,
        secNodes=4, 
        noiseLevel=0.01, 
        noiseAbs=1e-5, 
        seed=1337
    )
    
    return data_tt_homo, slo_array_homo

# %%============================================================================
# ERT DATA FUNCTIONS
# ============================================================================

def simulate_ert_data(mesh_ERT, scheme, percentage=0.50):
    """
    Simulate ERT data for homogeneous model.
    
    Parameters
    ----------
    mesh_ERT : pygimli.Mesh
        Mesh for ERT simulation
    scheme : pg.DataContainer
        ERT measurement scheme
    percentage : float, optional
        Percentage contrast between layers, by default 0.50
        
    Returns
    -------
    tuple
        (homosyndata, res_array_homo) - Simulated data and resistivity array
    """
    # Define resistivities
    fw_start_rho = 1000 
    hw_start_rho = fw_start_rho + (percentage * fw_start_rho)
    
    # Create resistivity array
    res_array_homo = pg.solver.parseMapToCellArray(
        [[1, hw_start_rho], [2, fw_start_rho]], mesh_ERT)
    
    # Simulate ERT data
    homosyndata = ert.simulate(
        mesh_ERT, 
        scheme=scheme, 
        res=res_array_homo, 
        noiseLevel=1, 
        nseed=1337
    )
    
    # Remove negative apparent resistivity values
    homosyndata.remove(homosyndata['rhoa'] < 0)
    
    return homosyndata, res_array_homo



# %%============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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


def create_combined_tt_figure(mesh_TT, slo_array_homo, data_tt_homo, 
                             shots, receivers, truemodel_points, mgr, 
                             synthetic_path, contrast):
    """
    Create a combined figure showing true model, data, and inversion result.
    
    Parameters
    ----------
    mesh_TT : pygimli.Mesh
        Travel time mesh
    slo_array_homo : array-like
        True slowness model
    data_tt_homo : pg.DataContainer
        Synthetic travel time data
    shots : numpy.ndarray
        Shot positions
    receivers : numpy.ndarray
        Receiver positions
    truemodel_points : numpy.ndarray
        True fault positions
    mgr : tt.TravelTimeManager
        Inversion manager with results
    synthetic_path : str
        Path to save figure
    contrast : int
        Velocity contrast value
    """
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)
    
    # Subplot a: True model
    ax1 = fig.add_subplot(gs[0, 0])
    pg.show(mesh_TT, slo_array_homo, ax=ax1, **pg_show_kwargs['tt'])
    ax1.scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    ax1.scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
    ax1.scatter(truemodel_points[:, 0], truemodel_points[:, 1], 
               **scatter_kwargs['true_positions'])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(3, 20)
    legend = ax1.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_color('white')
    ax1.set_xlabel('X (m)', fontsize=20)
    ax1.set_ylabel('Z (m)', fontsize=20)
    ax1.set_title('True Model', fontsize=20)
    ax1.text(-1, 21, 'a)', fontsize=20)
    
    # Subplot b: Apparent velocities
    ax2 = fig.add_subplot(gs[0, 1])
    _, cb = showVA(data_tt_homo, usePos=False, ax=ax2, cMap='viridis',
                   cMin=2000, cMax=3000)
    cb.ax.set_xlabel('Apparent Velocity (m/s)', fontsize=20)
    cb.ax.tick_params(labelsize=20)
    ax2.invert_yaxis()
    ax2.set_xlabel('Receivers - Z(m)', fontsize=20)
    ax2.set_ylabel('Shot - Z(m)', fontsize=20)
    ax2.set_title('Synthetic data', fontsize=20)
    ax2.text(-2, -2, 'b)', fontsize=20)
    
    # Subplot c: Inversion result
    ax3 = fig.add_subplot(gs[0, 2])
    mgr.showResult(ax=ax3, showlogScale=True, **pg_show_kwargs['tt'])
    ax3.scatter(shots[:, 0], shots[:, 1], **scatter_kwargs['shots'])
    ax3.scatter(receivers[:, 0], receivers[:, 1], **scatter_kwargs['receivers'])
    ax3.set_xlim(0, 12)
    ax3.set_ylim(3, 20)
    legend = ax3.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_color('white')
    ax3.set_xlabel('X (m)', fontsize=20)
    ax3.set_ylabel('Z (m)', fontsize=20)
    ax3.set_title('Inversion Result', fontsize=20)
    ax3.text(-1, 21, 'c)', fontsize=20)
    
    # Save figure
    fig.savefig(synthetic_path + 'TT_homo_combined_figure_gridspec.pdf')
    plt.show()




# %%============================================================================
# MAIN EXECUTION BLOCKS
# ============================================================================

# %% GEOLOGICAL MODEL CREATION
print("Creating geological model...")
geo_model, plc, map_data, truemodel_points = create_geological_model()

# %% ERT SENSOR SETUP AND DATA SIMULATION
print("\nSetting up ERT sensors and simulating data...")

# Create cross-hole sensors
sensors = create_crosshole_sensors()

# Visualize sensors on geological model
ax, _ = pg.show(plc, label='paravec', alpha=0.3, hold=True)
ax.scatter(sensors[:, 0], sensors[:, 1], c='red', s=100)
plt.title('Cross-hole sensor array')
plt.show()

# Create ERT measurement scheme
n = 30
xscheme = utils.create2Dxhconfs(n, ds=[4,2], bipole=True)
scheme = pg.DataContainerERT()

for i in sensors:
    scheme.createSensor(i)

for measurement in xscheme:
    scheme.addFourPointData(*measurement)

print(f"ERT scheme created with {scheme.size()} measurements")

# Create ERT mesh and simulate data
for i in scheme.sensors():
    plc.createNode(i)
    plc.createNode(i - [0, 0.1])

mesh_ERT = mt.createMesh(plc, quality=31, area=100)

# Simulate ERT data
percentage = 0.50
homosyndata, res_array_homo = simulate_ert_data(mesh_ERT, scheme, percentage)

print(f"ERT data simulated with {homosyndata.size()} valid measurements")

# %% TRAVEL TIME SENSOR SETUP AND DATA SIMULATION
print("\nSetting up travel time sensors and simulating data...")

# Setup travel time scheme
tt_scheme, sensors_tt = setup_tt_sensors_and_scheme()

# Create travel time mesh
ttplc = plc
for i in sensors_tt:
    ttplc.createNode(i)

mesh_TT = mt.createMesh(ttplc, quality=31)

# Simulate travel time data
contrast = 600
data_tt_homo, slo_array_homo = simulate_tt_data(mesh_TT, tt_scheme, contrast)

print(f"Travel time data simulated with {data_tt_homo.size()} measurements")

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

# %% DATA VISUALIZATION AND SAVING
print("\nVisualizing and saving synthetic data...")

# Plot travel time data
fig, ax = plt.subplots(1, 1)
_, cb = showVA(data_tt_homo, ax=ax, usePos=False, cMap='viridis')
cb.ax.tick_params(labelsize=20)
ax.invert_yaxis()
ax.tick_params(labelsize=18)
cb.ax.set_xlabel('Apparent Velocity (m/s)', fontsize=20)
ax.set_xlabel('Receivers - y(m)', fontsize=18)
ax.set_ylabel('Shot - y(m)', fontsize=18)
ax.set_title('Synthetic data', fontsize=28)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.savefig(synthetic_path + f'TT_homo_data_c{contrast}.svg')

# Plot true travel time model
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
_, cb = pg.show(mesh_TT, slo_array_homo, ax=ax, **pg_show_kwargs['tt'])
ax.scatter(shots[:,0], shots[:,1], **scatter_kwargs['shots'])
ax.scatter(receivers[:,0], receivers[:,1], **scatter_kwargs['receivers'])
ax.scatter(truemodel_points[:,0], truemodel_points[:,1], **scatter_kwargs['true_positions'])
ax.set_xlim(0, 12)
ax.set_ylim(3, 20)
legend = ax.legend(loc='upper left', fontsize=20)
for text in legend.get_texts():
    text.set_color('white')
cb.ax.set_xlabel('Velocity (m/s)', fontsize=20)
ax.set_title('True model', fontsize=28)
fig.savefig(synthetic_path + f'TT_homo_truemodel_c{contrast}.pdf')

# Plot ERT true model
fig, ax = plt.subplots(1, 1)
pg.show(mesh_ERT, res_array_homo, ax=ax, **pg_show_kwargs['ert'])
ax.scatter(sensors[:,0], sensors[:,1], **scatter_kwargs['ert_sensors'])
ax.scatter(truemodel_points[:,0], truemodel_points[:,1], **scatter_kwargs['true_positions'])
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_xlim(0, 14)
ax.set_ylim(0, 25)
fig.savefig(synthetic_path + f'ERT_homo_truemodel_c{percentage}.png')

# Plot ERT data
ax, _ = pg.show(homosyndata, label='Apparent Resistivities (ohm-m)')
ax.set_title('ERT data - homogeneous')
fig = ax.get_figure()
fig.savefig(synthetic_path + f'ERT_homo_data_{percentage}.png')

# Save data
data_tt_homo.save(synthetic_path + f'XHOLE_HOMO_2D_TT_DATA_c{contrast}.sgt')
homosyndata.save(synthetic_path + f'XHOLE_HOMO_2D_ERT_DATA_c{percentage}')

plc, map = ggm.get_geometry_2d(geo_model, 'section1', show=True)


# %%
# %% INVERSION EXAMPLES
print("\nRunning inversion examples...")

# Travel Time Inversion
x, y, _ = data_tt_homo.sensorPositions().array().T
bound = 20
rect = mt.createRectangle([x[-1] - bound, y[30] - bound], 
                         [x[0] + bound, y[-1] + bound])

for sen in data_tt_homo.sensors().array():
    rect.createNode(sen)

mesh_inv = mt.createMesh(rect, quality=34, area=0.1)
mgr = tt.TravelTimeManager(data=data_tt_homo['t'], verbose=True)
inv = mgr.invert(data=data_tt_homo, mesh=mesh_inv, lam=50, verbose=True,  
                useGradient=False, ignoreRegionManager=True)

#%%
# Plot TT inversion result
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
ax, cb = mgr.showResult(ax=ax, showlogScale=True, **pg_show_kwargs['tt'])
ax.scatter(shots[:,0], shots[:,1], **scatter_kwargs['shots'])
ax.scatter(receivers[:,0], receivers[:,1], **scatter_kwargs['receivers'])
ax.set_xlim(0, 12)
ax.set_ylim(3, 20)
ax.set_title('Inversion Result', fontsize=28)
legend = ax.legend(loc='upper left', fontsize=20)
for text in legend.get_texts():
    text.set_color('white')
fig.savefig(synthetic_path + 'RegularInversion_HOMOTT.pdf')

# Create combined figure
create_combined_tt_figure(mesh_TT, slo_array_homo, data_tt_homo, 
                         shots, receivers, truemodel_points, mgr, 
                         synthetic_path, contrast)

print("Script completed successfully!")

