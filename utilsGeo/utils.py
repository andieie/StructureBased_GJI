"""
Utility Functions for Geometric Inversion and Geophysical Modeling

This module provides essential utility functions for geometric inversion workflows,
mesh operations, data interpolation, and configuration generation for geophysical
modeling using PyGIMLi framework.

Key Functionality:
------------------
- Data interpolation between different mesh structures
- Sensitivity matrix calculation for geophysical methods
- Mesh manipulation and parameter vector creation
- Crosshole measurement configuration generation
- Interface point filtering and geometric operations

Dependencies:
-------------
- PyGIMLi: Main geophysical modeling framework
- NumPy: Numerical operations and array handling
- PyGIMLi physics modules: ERT and travel-time modeling

Authors: Structure-Based Inversion Research Group
License: See LICENSE file for details
Version: 2.0 - Cleaned and documented
"""

from pygimli.utils import sparseMatrix2coo
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.utils import gmat2numpy as mat2np
import pygimli.physics.ert as ert
import numpy as np
import pygimli.physics.traveltime as tt


def interpol(inmesh, indata, outmesh):
    """
    Interpolate data from one mesh to another using PyGIMLi interpolation.
    
    This function performs spatial interpolation of data values from a source mesh
    to the cell centers of a destination mesh. Commonly used for transferring
    model parameters or results between different mesh discretizations.

    Parameters:
    -----------
    inmesh : pygimli.Mesh
        Source mesh containing the original data
    indata : array-like
        Data values associated with inmesh cells
    outmesh : pygimli.Mesh
        Destination mesh where data will be interpolated

    Returns:
    --------
    pygimli.Vector
        Interpolated data values at outmesh cell centers

    Example:
    --------
    >>> coarse_mesh = pg.createGrid(x=np.linspace(0, 10, 5))
    >>> fine_mesh = pg.createGrid(x=np.linspace(0, 10, 20))
    >>> coarse_data = [1, 2, 3, 4]
    >>> fine_data = interpol(coarse_mesh, coarse_data, fine_mesh)
    """
    outdata = pg.Vector()
    pg.interpolate(srcMesh=inmesh, inVec=indata,
                   destPos=outmesh.cellCenters(), outVec=outdata)
    return outdata


def message(txt):
    """
    Display a formatted message with decorative borders.
    
    Prints text within a box of dashes for emphasis, commonly used
    for status updates and progress indicators during long computations.

    Parameters:
    -----------
    txt : str
        Message text to display

    Example:
    --------
    >>> message("Starting inversion process")
    ----------------------------------
    ----------------------------------
    ----------------------------------
    ### Starting inversion process ###
    ----------------------------------
    ----------------------------------
    ----------------------------------
    """
    eline = "----------------------------------"
    print(eline)
    print(eline)
    print(eline)
    print("### " + txt + " ###")
    print(eline)
    print(eline)
    print(eline)


def interpolMat(inmesh, outmesh):
    """
    Create interpolation matrix between two meshes.
    
    Generates a dense interpolation matrix that can be used for repeated
    interpolation operations between the same mesh pair. More efficient
    than interpol() when multiple interpolations are needed.

    Parameters:
    -----------
    inmesh : pygimli.Mesh
        Source mesh
    outmesh : pygimli.Mesh
        Destination mesh

    Returns:
    --------
    numpy.ndarray
        Dense interpolation matrix (outmesh_cells × inmesh_cells)

    Notes:
    ------
    For large meshes, this matrix can consume significant memory.
    Consider using interpol() for single-use interpolations.
    """
    mat = inmesh.interpolationMatrix(outmesh)
    return sparseMatrix2coo(mat).toarray()


def nearest_neighbor_interpolation(indata, inmesh, outmesh, nan=99.9):
    """
    Perform nearest neighbor interpolation between meshes.
    
    For each position in the output mesh, finds the nearest cell in the 
    input mesh and assigns the corresponding data value. Useful when
    smooth interpolation is not desired or when preserving discrete values.

    Parameters:
    -----------
    indata : array-like
        Data values for each cell in inmesh
    inmesh : pygimli.Mesh
        Source mesh containing the data
    outmesh : pygimli.Mesh
        Destination mesh for interpolation
    nan : float, default=99.9
        Value to assign when no nearest cell is found

    Returns:
    --------
    list
        Interpolated data values at outmesh cell centers

    Notes:
    ------
    This method preserves sharp boundaries and discrete values better
    than smooth interpolation methods, making it suitable for geological
    unit assignments and material property mapping.

    Example:
    --------
    >>> data = [1, 2, 2, 3]  # Discrete material IDs
    >>> result = nearest_neighbor_interpolation(data, mesh1, mesh2)
    """
    outdata = []
    for pos in outmesh.cellCenters():
        cell = inmesh.findCell(pos)
        if cell:
            outdata.append(indata[cell.id()])
        else:
            outdata.append(nan)
    return outdata


def calc_sens(mesh, dta, model, method):
    """
    Calculate sensitivity matrix for geophysical forward modeling.
    
    Computes the Jacobian (sensitivity) matrix for the specified geophysical
    method. The mesh must have uniform cell markers for proper calculation.
    This matrix quantifies how changes in model parameters affect the data.

    Parameters:
    -----------
    mesh : pygimli.Mesh
        Computational mesh (cannot have heterogeneous cell markers)
    dta : pygimli.DataContainer
        Measurement data configuration (electrode positions, etc.)
    model : array-like
        Current model parameter values
    method : str
        Geophysical method ('TravelTime' or 'ERT')

    Returns:
    --------
    numpy.ndarray
        Dense sensitivity matrix (n_data × n_model_parameters)

    Notes:
    ------
    - All cells are set to marker=1 for uniform sensitivity calculation
    - Uses multi-threading (8 threads) for faster computation
    - The sensitivity matrix size can be very large for fine meshes
    - Computation time scales with mesh size and data configuration

    Raises:
    -------
    ValueError
        If method is not 'TravelTime' or 'ERT'

    Example:
    --------
    >>> mesh = create_mesh()
    >>> data = load_data_configuration()
    >>> model = np.ones(mesh.cellCount()) * 100  # Initial resistivity
    >>> sens = calc_sens(mesh, data, model, 'ERT')
    >>> print(f"Sensitivity matrix shape: {sens.shape}")
    """
    pg.tic("Sensitivity_Calculation")
    pg.setThreadCount(8)
    
    # Set uniform cell markers for sensitivity calculation
    for cell in mesh.cells():
        cell.setMarker(1)
    print('CELL MARKERS FOR ', np.unique(mesh.cellMarkers()))
    
    # Initialize appropriate forward operator
    if method == 'TravelTime':
        forwop = tt.TravelTimeDijkstraModelling()
    elif method == 'ERT':
        forwop = ert.ERTModelling()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'TravelTime' or 'ERT'")
    
    # Configure forward operator
    forwop.setData(dta)
    forwop.setMultiThreadJacobian(8)
    forwop.setMesh(mesh)
    
    # Calculate Jacobian matrix
    forwop.createJacobian(model)
    message(f"Jacobian shape: {forwop.jacobian().shape}")
    
    pg.toc(box=True)
    return pg.utils.sparseMatrix2Dense(forwop.jacobian())

def getPoints(geo_model, stack=False):
    """
    Extract surface points from a GemPy geological model.
    
    Retrieves the X, Y, Z coordinates of surface points that define
    geological interfaces in the model. These points are typically
    used to constrain the geological interpretation.

    Parameters:
    -----------
    geo_model : gempy.core.model.GeoModel
        GemPy geological model containing surface points
    stack : bool, default=False
        If True, return coordinates as single stacked array
        If False, return separate X, Y, Z arrays

    Returns:
    --------
    tuple or numpy.ndarray
        If stack=False: (x, y, z) as separate arrays
        If stack=True: (n_points, 3) stacked coordinate array

    Example:
    --------
    >>> x, y, z = getPoints(geo_model, stack=False)
    >>> coords = getPoints(geo_model, stack=True)
    >>> print(f"Found {len(coords)} surface points")
    """
    x = np.array(geo_model.surface_points.df['X'])
    y = np.array(geo_model.surface_points.df['Y'])
    z = np.array(geo_model.surface_points.df['Z'])
    
    if stack:
        stackv = np.stack([x, y, z], axis=1)
        return stackv
    else:
        return x, y, z
        
def filter_top_coverage_points(interface_points, mesh, coverage, min_points=6, fixed_points=None):
    """
    Filter interface points based on coverage values to select most informative points.
    
    This function selects the most informative interface points for geometric inversion
    by ranking them according to their coverage values (sensitivity). Points with higher
    coverage contribute more to the inversion and are prioritized for inclusion.

    Parameters:
    -----------
    interface_points : list of lists
        Interface points categorized by geological units, where each sublist
        contains points belonging to the same interface category
    mesh : pygimli.Mesh
        Computational mesh for finding nearest cells to points
    coverage : array-like
        Coverage/sensitivity values for each mesh cell
    min_points : int, default=6
        Minimum number of top points to retain
    fixed_points : list, optional
        Points that must be included regardless of coverage values

    Returns:
    --------
    list of lists
        Filtered interface points grouped by original categories,
        containing only the most informative points

    Notes:
    ------
    - Points are ranked by their coverage values in descending order
    - Fixed points are always included and assigned infinite coverage
    - The function preserves the original categorical structure
    - Useful for reducing computational cost while maintaining inversion quality

    Example:
    --------
    >>> filtered_points = filter_top_coverage_points(
    ...     interface_points, mesh, coverage_values, 
    ...     min_points=10, fixed_points=critical_points
    ... )
    >>> print(f"Selected {sum(len(cat) for cat in filtered_points)} points")
    """
    # Initialize fixed point handling
    if fixed_points is not None:
        fixed_point = []
    
    # Flatten interface points while preserving category information
    flat_interface_points = []
    for category_index, sublist in enumerate(interface_points):
        for point in sublist:
            flat_interface_points.append((point, category_index))
    
    # Calculate coverage for each point
    points_with_coverage = []
    for point, category_index in flat_interface_points:
        # Find nearest mesh cell for this point
        nearest_cell = mesh.findCell(point)
        
        if nearest_cell is not None:
            # Get coverage value for this cell
            cell_index = nearest_cell.id()
            coverage_value = coverage[cell_index]
            
            # Store point with its coverage and category
            points_with_coverage.append((point, coverage_value, category_index))
    
    # Sort points by coverage value (highest first)
    points_with_coverage.sort(key=lambda x: x[1], reverse=True)
    
    # Select top points with highest coverage
    top_points_with_coverage = points_with_coverage[:min_points]

    # Add fixed points with infinite coverage priority
    if fixed_points is not None:
        for fixed_point in fixed_points:
            # Check if fixed point is already in top points
            if fixed_point not in [p[0] for p in top_points_with_coverage]:
                # Find category index for the fixed point
                for point, category_index in flat_interface_points:
                    if np.allclose(point, fixed_point):
                        top_points_with_coverage.append((fixed_point, float('inf'), category_index))
                        break

    # Reconstruct categorical structure
    top_points_grouped = [[] for _ in interface_points]
    for point, _, category_index in top_points_with_coverage:
        top_points_grouped[category_index].append(point)
    
    return top_points_grouped


def createStartGeom(ipoints, dimshift, startshift=[0, 0]):
    """
    Create starting geometry coordinates for interface points.
    
    Generates initial coordinates for interface points in geometric inversion.
    This is typically used to create starting positions for interface points
    that will be optimized during the inversion process.

    Parameters:
    -----------
    ipoints : int
        Number of interface points to create
    dimshift : str
        Dimension(s) to shift the starting points ('X', 'Z', or 'XZ')
    startshift : list, default=[0, 0]
        Starting shift values [x_shift, z_shift] for the points

    Returns:
    --------
    numpy.ndarray
        Flattened array of starting coordinates

    Notes:
    ------
    - 'X': Only x-coordinates vary, z-coordinates are constant
    - 'Z': Only z-coordinates vary, x-coordinates are constant  
    - 'XZ': Both x and z coordinates are optimized (2D interface)

    Example:
    --------
    >>> # Create 5 points varying only in X direction
    >>> start_coords = createStartGeom(5, 'X', startshift=[100, -50])
    >>> # Create 3 points varying in both X and Z
    >>> start_coords_2d = createStartGeom(3, 'XZ', startshift=[0, 0])
    """
    sx = np.zeros(ipoints) + startshift[0]
    sz = np.zeros(ipoints) + startshift[1]
    
    if dimshift == 'X':
        ip = list(sx.T)
    elif dimshift == 'Z':
        ip = list(sz.T)
    elif dimshift == 'XZ':
        ip = list(np.stack((sx, sz)).T)
    else:
        raise ValueError(f"Unknown dimshift: {dimshift}. Use 'X', 'Z', or 'XZ'")
    
    sps = np.hstack(ip)
    return sps


def makeParavecGrid(mesh, paramesh, nx, ny, para_cz, para_br):
    """
    Interpolate parameter vectors from a regular grid to mesh cells.
    
    Creates parameter vectors for the mesh by interpolating from regular
    grids based on geological unit assignments. This is useful for
    initializing spatially varying parameters in geophysical modeling.

    Parameters:
    -----------
    mesh : pygimli.Mesh
        Target mesh for parameter interpolation
    paramesh : array-like
        Geological unit assignments for each mesh cell
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    para_cz : numpy.ndarray
        Parameter values for the first geological unit
    para_br : numpy.ndarray
        Parameter values for the second geological unit

    Returns:
    --------
    numpy.ndarray
        Parameter vector for mesh cells based on geological unit assignment

    Notes:
    ------
    - Creates a regular grid covering the mesh extent
    - Interpolates parameters from grid to mesh cell centers
    - Assigns parameters based on geological unit (paramesh values)
    - Currently supports two-unit models (units 1 and 2)

    Example:
    --------
    >>> # Create parameter grids for two geological units
    >>> cz_params = np.random.rand(nx * ny) * 100  # First unit parameters
    >>> br_params = np.random.rand(nx * ny) * 200  # Second unit parameters
    >>> params = makeParavecGrid(mesh, unit_ids, 20, 15, cz_params, br_params)
    """
    # Create regular grid covering mesh extent
    grid = mt.createGrid(x=np.linspace(mesh.xMin(), mesh.xMax(), nx),
                         y=np.linspace(mesh.yMin(), mesh.yMax(), ny))
    
    print(f"Parameter vector lengths: para_br={len(para_br)}, para_cz={len(para_cz)}")
    
    # Interpolate parameter grids to mesh
    data_cz = interpol(grid, para_cz, mesh)
    data_br = interpol(grid, para_br, mesh)
    
    # Assign parameters based on geological unit
    # Unit 2 gets para_cz, other units get para_br
    data = np.where(paramesh == 2, data_cz, data_br)
    
    return data


def makeParavec(mesh, initmap, paramap):
    """
    Create parameter vector for mesh cells based on geological mapping.
    
    Generates a parameter vector that assigns values to mesh cells according
    to their geological unit classification. Combines initial geological
    mapping with parameter assignments for each lithological unit.

    Parameters:
    -----------
    mesh : pygimli.Mesh
        Computational mesh for parameter assignment
    initmap : dict
        Dictionary mapping lithology names to initial cell identifiers
        Format: {lithology_name: [cell_ids]}
    paramap : dict
        Dictionary mapping lithology names to parameter values
        Format: {lithology_name: parameter_value}

    Returns:
    --------
    pygimli.Vector
        Parameter vector with values assigned to mesh cells

    Example:
    --------
    >>> initmap = {'sandstone': [1], 'shale': [2]}
    >>> paramap = {'sandstone': 100, 'shale': 20}  # Resistivity values
    >>> param_vector = makeParavec(mesh, initmap, paramap)
    
    Notes:
    ------
    This function is essential for creating heterogeneous parameter
    distributions based on geological interpretation. The parameter
    vector can represent any physical property (resistivity, velocity, etc.).
    """
    cells = []
    for lithology_name, param_value in paramap.items():
        cell_ids = initmap.get(lithology_name)
        if cell_ids is not None:
            # Create [cell_id, parameter_value] pair
            temp = [cell_ids[0], param_value]
            cells.append(temp)
    
    # Parse cell assignments into PyGIMLi vector
    param_vector = pg.solver.parseMapToCellArray(cells, mesh)
    return param_vector


def createMasterDict(initmap, paramap, sur_int):
    """
    Create a master dictionary combining geological modeling components.
    
    Consolidates geological unit information, parameter mappings, and surface 
    intersections into a single master dictionary. This provides a unified 
    data structure for geometric inversion workflows.

    Parameters:
    -----------
    initmap : dict
        Dictionary mapping lithology names to initial cell identifiers
    paramap : dict
        Dictionary mapping lithology names to parameter values
    sur_int : dict
        Dictionary containing surface intersection data for each lithology

    Returns:
    --------
    dict
        Master dictionary where each lithology maps to:
        [cell_id, parameter_values, surface_intersections]

    Structure:
    ----------
    {
        'lithology_name': [
            cell_identifier,     # From initmap
            parameter_values,    # From paramap
            surface_points       # From sur_int
        ]
    }

    Example:
    --------
    >>> initmap = {'sandstone': [1], 'shale': [2]}
    >>> paramap = {'sandstone': 100, 'shale': 20}
    >>> sur_int = {'sandstone': interface_points, 'shale': interface_points}
    >>> master = createMasterDict(initmap, paramap, sur_int)
    
    Notes:
    ------
    This master dictionary serves as the central data structure for organizing
    geological information needed in geometric inversion workflows. The simplified
    structure removes unused constraint parameters for better maintainability.
    """
    master_dict = {}
    for lithology_name, cell_id in initmap.items():
        parameter_value = paramap.get(lithology_name)
        surface_points = sur_int.get(lithology_name)
        
        master_dict[lithology_name] = [
            cell_id,
            parameter_value,
            surface_points
        ]
    
    return master_dict


def create2Dxhconfs(nel, ds=1, inhole=False, bipole=False):
    """
    Create 2D crosshole measurement configurations for geophysical surveys.
    
    Generates electrode configurations for crosshole geophysical measurements,
    such as electrical resistivity tomography (ERT) or induced polarization (IP).
    Supports various dipole configurations and optional in-hole measurements.

    Parameters:
    -----------
    nel : int
        Number of electrodes in each borehole (assumes equal numbers)
    ds : int or list, default=1
        Spacing of dipoles in electrode numbers. If int, single spacing used.
        If list, multiple spacings are combined
    inhole : bool, default=False
        If True, adds single-hole measurements (requires create2Dconfs function)
    bipole : bool, default=False
        If True, injects current across boreholes (A-B across holes)
        If False, uses conventional dipole-dipole (A-B, M-N per hole)

    Returns:
    --------
    numpy.ndarray
        Configuration array with shape (n_configs, 4) containing
        [A, B, M, N] electrode indices for each measurement

    Notes:
    ------
    - Electrode numbering: 0 to nel-1 for first borehole, nel to 2*nel-1 for second
    - Bipole configuration alternates current electrodes between boreholes
    - Standard configuration keeps current dipole within same borehole
    - In-hole measurements require additional function (currently disabled)

    Example:
    --------
    >>> # Simple crosshole configuration with 20 electrodes per hole
    >>> configs = create2Dxhconfs(20, ds=1, bipole=False)
    >>> print(f"Generated {len(configs)} configurations")
    
    >>> # Multiple dipole spacings
    >>> configs = create2Dxhconfs(20, ds=[1, 2, 3], bipole=True)
    
    Author: Florian Wagner (2012)
    """
    def createxhint(nel, ds):
        """Create crosshole interval configurations."""
        # Define electrode arrays for both boreholes
        bh1 = np.linspace(0, nel - 1, nel, dtype=int)
        bh2 = np.linspace(nel, 2 * nel - 1, nel, dtype=int)

        confs = []

        # Generate all possible configurations
        for i in range(nel - ds):
            for j in range(nel - ds):
                a = bh1[i]
                b = a + ds
                m = bh2[j]
                n = m + ds
                
                if not bipole:
                    # Standard dipole-dipole: A-B and M-N within same holes
                    confs.append((a, b, m, n))
                else:
                    # Bipole: current across boreholes
                    confs.append((a, m, b, n))
                    
        return np.asarray(confs)

    # Handle single or multiple dipole spacings
    if isinstance(ds, int):
        if ds > nel - 1:
            print(f"WARNING: dipole interval of {ds} is too large!")
            confs = np.array([]).reshape(0, 4)
        else:
            confs = createxhint(nel, ds)
    else:
        # Multiple spacings provided
        confs = []
        for spacing in ds:
            if spacing > nel - 1:
                print(f"WARNING: dipole interval of {spacing} is too large!")
                continue
            else:
                confs.append(createxhint(nel, spacing))
        
        if confs:
            confs = np.vstack(confs)
        else:
            confs = np.array([]).reshape(0, 4)
    
    # Add in-hole measurements if requested
    if inhole:
        print("Warning: create2Dconfs function not available - skipping in-hole measurements")
        # Note: Would require additional function for single-hole configurations
    
    print(f"{len(confs)} configurations generated.")
    return confs
