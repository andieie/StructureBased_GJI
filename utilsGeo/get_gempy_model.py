"""
GemPy-PyGIMLi Integration Module

This module provides utilities for converting geological models from GemPy 
to PyGIMLi-compatible formats for geophysical inversions. It handles the 
extraction of geological geometries, creation of computational meshes, and 
conversion between different mesh formats.

Key Functions:
- make_gempy_model: Create and initialize GemPy geological models
- get_geometry_2d: Extract 2D cross-sections as PyGIMLi PLCs
- get_geometry_3d: Convert 3D geological models to PyGIMLi meshes
- Polygon extraction and mesh generation utilities

Dependencies:
- numpy: Numerical operations
- pygimli: Geophysical inversion and mesh generation
- gempy: Geological modeling
- matplotlib: Plotting and visualization
- subsurface: Mesh data structures (for 3D operations)
"""

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import gempy as gp
from IPython.display import set_matplotlib_formats
from gempy.plot import visualization_2d as vv

# Configure matplotlib for better output
set_matplotlib_formats('svg')
plt.rcParams["figure.figsize"] = (10, 10)

def make_gempy_model(sidict, dimension, extent, resolution, section, plot=True):
    """
    Create and initialize a GemPy geological model with surfaces and orientations.
    
    This function creates a complete geological model by adding surface points,
    orientations, and configuring the interpolation settings. For 2D models,
    it automatically adds orientation points to ensure proper horizontal layer
    interpolation.

    Parameters:
    -----------
    sidict : dict
        Dictionary where keys are surface names and values are lists of 
        tuples representing surface points (x, y, z)
    dimension : int
        Model dimension (2 for 2D cross-sections, 3 for full 3D models)
    extent : list
        Spatial extent [xmin, xmax, ymin, ymax, zmin, zmax]
    resolution : list
        Grid resolution [nx, ny, nz]
    section : list
        2D section coordinates for cross-section extraction
    plot : bool, default=True
        Whether to display the model after creation

    Returns:
    --------
    gempy.core.model.GeoModel
        Configured geological model ready for computation

    Notes:
    ------
    - For 2D models, orientation points are automatically added at model
      edges and center to ensure proper horizontal layer interpolation
    - The last surface in the dictionary is treated as basement and may
      not receive orientation points
    - Uses horizontal orientations (90°, 0°, 1) for layered geology
    """
    geo_model = gp.create_model("geological_model")
    gp.init_data(geo_model, extent=extent, resolution=resolution)
    
    if dimension == 2:
        # Add surfaces and points for 2D model
        for surface, interface in sidict.items():
            geo_model.add_surfaces(surface)
            
            # Add surface points
            for (x, y, z) in interface:
                geo_model.add_surface_points(X=x, Y=y, Z=z, surface=surface)
            
            # Add orientation points for proper interpolation
            # Skip orientations for the last surface (typically basement)
            if not ((len(sidict) <= 3) and (surface == list(sidict.keys())[-1])):
                z_level = interface[0][2]  # Use Z coordinate from first point
                center_x = ((extent[1] - extent[0]) / 2) + extent[0]
                center_y = ((extent[3] - extent[2]) / 2) + extent[2]
                
                # Add orientation points at model boundaries and center
                orientation_points = [
                    (extent[0] + 1, extent[2] + 1),    # Bottom-left
                    (extent[1] - 1, extent[3] - 1),    # Top-right  
                    (center_x, extent[2] + 1),         # Bottom-center
                    (center_x, extent[3] - 1),         # Top-center
                ]
                
                for ox, oy in orientation_points:
                    geo_model.add_orientations(
                        X=ox, Y=oy, Z=z_level, surface=surface,
                        orientation=[90, 0, 1]  # Horizontal layers
                    )
                    
    else:
        # Add surfaces and points for 3D model
        for surface, interface in sidict.items():
            geo_model.add_surfaces(surface)
            
            for i, (x, y, z) in enumerate(interface):
                geo_model.add_surface_points(X=x, Y=y, Z=z, surface=surface)
                
                # Add orientations between consecutive points
                if i > 0:
                    # Calculate midpoint between current and previous point
                    prev_point = interface[i-1]
                    mid_x = ((x - prev_point[0]) / 2) + prev_point[0]
                    mid_y = ((y - prev_point[1]) / 2) + prev_point[1]
                    mid_z = ((z - prev_point[2]) / 2) + prev_point[2]
                    
                    # Add orientation at midpoint (only for even indices)
                    if i % 2 == 0:
                        geo_model.add_orientations(
                            X=mid_x, Y=mid_y, Z=mid_z, surface=surface,
                            orientation=[90, 0, 1]  # Horizontal layers
                        )

    # Configure model parameters
    geo_model.modify_kriging_parameters('range', 10)
    geo_model.set_section_grid(section)
    
    # Set up interpolator and compute model
    gp.set_interpolator(geo_model, compile_theano=True, theano_optimizer="fast_run")
    gp.compute_model(geo_model, to_subsurface=True)
    
    # Display model if requested
    if plot:
        if dimension == 2:
            gp.plot_2d(geo_model)
        else:
            gp.plot_3d(geo_model)
    
    return geo_model


def findExtrema(points):
    """
    Find the extreme coordinate values from a set of points.
    
    Parameters:
    -----------
    points : array-like
        Array of 2D points with shape (n, 2)
        
    Returns:
    --------
    dict
        Dictionary with keys 'top', 'bottom', 'left', 'right' containing
        the maximum and minimum coordinate values
    """
    return {
        'top': np.max(points[:, 1]),
        'bottom': np.min(points[:, 1]),
        'left': np.min(points[:, 0]),
        'right': np.max(points[:, 0])
    }


def placeMarker(points, shape):
    """
    Find a suitable position for placing a region marker inside a polygon.
    
    This function searches for a point inside the given shape that can serve
    as a marker position for mesh generation. It starts with the geometric
    center and then searches systematically across the bounding box.
    
    Parameters:
    -----------
    points : array-like
        Array of polygon boundary points
    shape : matplotlib.path.Path or similar
        Shape object that supports contains_points() method
        
    Returns:
    --------
    list
        [x, y] coordinates of a valid marker position inside the shape
        
    Notes:
    ------
    The algorithm first tries the geometric center, then performs a grid
    search across the bounding box to find an interior point.
    """
    bounds = findExtrema(points)
    bound_right = bounds['right']
    bound_left = bounds['left']
    bound_top = bounds['top']
    bound_bottom = bounds['bottom']
    
    # Calculate search grid resolution
    nsteps_vert = int((bound_top - bound_bottom) * 1.5)
    nsteps_hor = int((bound_right - bound_left) * 1.5)
    
    # Add small buffer to avoid boundary issues
    buffer_hor = (bound_right - bound_left) / 1000
    buffer_vert = (bound_top - bound_bottom) / 1000
    
    # Try geometric center first
    center_point = [(bound_right - bound_left) / 2, (bound_top - bound_bottom) / 2 * -1]
    if shape.contains_points([center_point]):
        return center_point
    
    # Grid search across bounding box
    for c_hor in np.linspace(bound_left + buffer_hor, bound_right - buffer_hor, nsteps_hor):
        for c_vert in np.linspace(bound_bottom + buffer_vert, bound_top - buffer_vert, 
                                 round(nsteps_vert)):
            test_point = [c_hor, c_vert]
            if shape.contains_points([test_point]):
                return test_point
    
    # Fallback to center if no interior point found
    return center_point


def clean_list_of_points(list_of_points):
    """
    Remove duplicate points from a list of point collections.
    
    Parameters:
    -----------
    list_of_points : list
        List of point arrays or single point array
        
    Returns:
    --------
    array-like
        Cleaned point array with duplicates removed
        
    Notes:
    ------
    If input contains only one point collection, returns it directly.
    Otherwise, removes duplicate points and returns the last collection.
    """
    if len(list_of_points) == 1:
        return list_of_points[0]
    
    seen = set()
    cleaned_list_of_lists = []

    for sublist in list_of_points:
        cleaned_sublist = []
        for point in sublist:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                cleaned_sublist.append(point)
        cleaned_list_of_lists.append(cleaned_sublist)
    
    return cleaned_list_of_lists[-1]


def _extract_boundaries(plot_obj, axes, section_name='topography'):
    """
    Extract geological boundaries from GemPy model for visualization.
    
    This is an internal function that extracts contour paths representing
    geological boundaries from a computed GemPy model. It handles both
    topographic surfaces and cross-sections, including fault surfaces.
    
    Parameters:
    -----------
    plot_obj : gempy.plot.visualization_2d.Plot2D
        GemPy plotting object containing model information
    axes : matplotlib.axes.Axes
        Matplotlib axes for plotting contours
    section_name : str, default='topography'
        Name of the section to extract ('topography' or section name)
        
    Returns:
    --------
    tuple
        (contour_sets, colors, extent) where:
        - contour_sets: List of matplotlib contour sets
        - colors: List of colors used for each geological unit
        - extent: Spatial extent of the section [xmin, xmax, ymin, ymax]
        
    Notes:
    ------
    This function is primarily used internally by get_polygon_dictionary()
    to extract boundary information for mesh generation.
    """
    cs = []
    
    # Get fault information
    faults = list(plot_obj.model._faults.df[plot_obj.model._faults.df['isFault'] == True].index)

    if section_name == 'topography':
        # Extract topographic surface data
        shape = plot_obj.model._grid.topography.resolution
        scalar_data = plot_obj.model.solutions.geological_map[1]
        extent = [
            plot_obj.model._grid.topography.extent[0],
            plot_obj.model._grid.topography.extent[1],
            plot_obj.model._grid.topography.extent[2],
            plot_obj.model._grid.topography.extent[3]
        ]
    else:
        # Extract cross-section data
        l0, l1 = plot_obj.model._grid.sections.get_section_args(section_name)
        section_index = np.where(plot_obj.model._grid.sections.names == section_name)[0][0]
        shape = [
            plot_obj.model._grid.sections.resolution[section_index][0],
            plot_obj.model._grid.sections.resolution[section_index][1]
        ]
        
        # Get scalar fields (excluding first for fault handling)
        scalar_fields = plot_obj.model.solutions.sections[1:]
        scalar_data = scalar_fields[:, l0:l1]
        fault_masks = plot_obj.model.solutions.fault_mask[:, l0:l1]
        
        extent = [
            0, 
            plot_obj.model._grid.sections.dist[section_index][0],
            plot_obj.model._grid.regular_grid.extent[4],
            plot_obj.model._grid.regular_grid.extent[5]
        ]

    # Process each geological unit
    zorder = 2
    counter = scalar_data.shape[0]
    fault_vertices = {}
    
    counters = np.arange(0, counter, 1)
    c_id = 0  # Color ID start
    colors = []
    
    for f_id in counters:
        block = scalar_data[f_id]
        level = plot_obj.model.solutions.scalar_field_at_surface_points[f_id][
            np.where(plot_obj.model.solutions.scalar_field_at_surface_points[f_id] != 0)
        ]

        levels = np.insert(level, 0, block.max())
        c_id2 = c_id + len(level)
        
        # Handle last unit
        if f_id == counters.max():
            levels = np.insert(levels, level.shape[0], block.min())
            c_id2 = c_id + len(levels)
            
        # Reshape data based on section type
        if section_name == 'topography':
            block = block.reshape(shape)
        else:
            block = block.reshape(shape).T
            
        zorder = zorder - (f_id + len(level))

        # Store fault vertices if this is a fault
        if f_id in faults:
            fault_vertices[f_id] = plot_obj.model.solutions.scalar_field_at_surface_points[f_id]

        # Create contours
        if f_id >= len(faults):
            # Regular geological unit
            color = plot_obj.cmap.colors[c_id:c_id2][::-1]
            plot = axes.contourf(
                block, 0, levels=np.sort(levels), colors=color,
                linestyles='solid', origin='lower',
                extent=extent, zorder=zorder
            )
        else:
            # Fault surface
            fault_mask = fault_masks[f_id].reshape(shape).T
            color = plot_obj.cmap.colors[c_id:c_id2][0]
            plot = axes.contour(
                fault_mask.astype(float), levels=[0.5], colors=color,
                linestyles='solid', origin='lower',
                extent=extent, zorder=zorder
            )
            
        c_id += len(level)
        cs.append(plot)
        
        # Store colors
        if isinstance(color, str):
            colors.append(color)
        else:
            colors.extend(color)
            
    return cs, colors, extent



def get_polygon_dictionary(geo_model, section_name):
    """
    Extract polygon paths and colors from a GemPy geological model section.
    
    This function extracts the boundary polygons for each geological unit
    from a computed GemPy model section. It returns the polygon paths and
    associated colors for visualization and mesh generation.

    Parameters:
    -----------
    geo_model : gempy.core.model.GeoModel
        Computed geological model with solutions
    section_name : str
        Name of the section to extract ('topography' or predefined section)

    Returns:
    --------
    tuple
        (pathdict, colordict, extent) where:
        - pathdict: Dictionary mapping surface names to matplotlib path objects
        - colordict: Dictionary mapping surface names to colors
        - extent: Spatial extent [xmin, xmax, ymin, ymax]
        
    Notes:
    ------
    The section_name must be either 'topography' or a predefined section
    from model.grid.sections. This function is typically used as a 
    preprocessing step for get_geometry_2d().
    """
    # Create plotting object and extract boundaries
    plot_obj = vv.Plot2D(geo_model)
    plot_obj.create_figure((13, 13))
    plot_obj.add_section(section_name, ax_pos=111)

    cs, colors, extent = _extract_boundaries(plot_obj, plot_obj.axes[0], section_name)
    
    # Extract paths from contour collections
    all_paths = []
    for contour in cs:
        for collection in contour.collections:
            all_paths.append(collection.get_paths())

    # Remove empty path lists
    all_paths = [path for path in all_paths if path != []]

    # Map colors to surface names
    surface_list = []
    for color in colors:
        matching_surfaces = geo_model._surfaces.df[
            geo_model._surfaces.df['color'] == color
        ]['surface'].values
        if len(matching_surfaces) > 0:
            surface_list.append(matching_surfaces[0])

    # Create dictionaries
    pathdict = dict(zip(surface_list, all_paths))
    colordict = dict(zip(surface_list, colors))

    return pathdict, colordict, extent



def get_geometry_2d(model, section, ignore=[], show=False, saveIMG=None, 
                   zshift=0, savePLC=None, simplify=True, resample=True):
    """
    Convert a GemPy geological model to a PyGIMLi-compatible 2D geometry.
    
    This function extracts 2D cross-sections from a 3D geological model and 
    converts them into PyGIMLi Piecewise Linear Complex (PLC) format for 
    geophysical mesh generation and inversion.

    Parameters:
    -----------
    model : gempy.core.model.GeoModel
        Computed GemPy geological model
    section : str
        Name of the predefined cross-section to extract
    ignore : list, default=[]
        List of surface names to exclude from the geometry
        (Note: Currently faults may cause issues and should be ignored)
    show : bool, default=False
        Display the resulting PyGIMLi geometry
    saveIMG : str, optional
        File path to save visualization image (only if show=True)
    zshift : float, default=0
        Vertical shift applied to coordinates (positive shifts model down)
    savePLC : str, optional
        File path to save the PLC geometry
    simplify : bool, default=True
        Remove redundant points to reduce mesh complexity
    resample : bool, default=True
        Resample polygon boundaries for better mesh quality

    Returns:
    --------
    tuple
        (world, mapping) where:
        - world: PyGIMLi PLC containing the 2D geological geometry
        - mapping: Dictionary mapping surface names to region markers

    Notes:
    ------
    - The function automatically sets boundary markers to 0 for proper
      boundary condition handling in PyGIMLi
    - Polygon simplification removes collinear points to reduce mesh size
    - The coordinate system is adjusted to match the section extent
    - Complex geological structures may require manual preprocessing
    
    Example:
    --------
    >>> plc, mapping = get_geometry_2d(geo_model, 'section1', 
    ...                               ignore=['fault1'], show=True)
    >>> mesh = mt.createMesh(plc, quality=30, area=0.5)
    """
    # Extract polygon paths and colors from the model
    verts, colors, extent = get_polygon_dictionary(model, section)
    plt.close('all')
    
    # Convert matplotlib paths to PyGIMLi geometry objects
    geoms = {surface: PathPatch(path[0]) for surface, path in verts.items()}

    # Find coordinate bounds for vertical shifting
    all_vertices = [geom.get_verts() for geom in geoms.values()]
    zmax = max([max(vertices[:, 1]) for vertices in all_vertices])
    zmin = min([min(vertices[:, 1]) for vertices in all_vertices])

    # Apply vertical shift
    if zshift:
        shift_amount = zmax
        zmin -= zmax
        zmax = 0
    else:
        shift_amount = zshift
        zmax -= shift_amount
        zmin -= shift_amount

    # Initialize geometry processing
    unit_count = 0
    units = []
    mapping = {}
    marker_positions = []
    
    # Process each geological unit
    for surface_name, unit_paths in verts.items():
        if surface_name.lower() not in [ignored.lower() for ignored in ignore]:
            unit_count += 1
            mapping[surface_name] = [unit_count]

            # Process each polygon within the unit
            for polygon_path in unit_paths:
                # Extract coordinate points from matplotlib path
                points = np.array([segment[0] for segment in polygon_path.iter_segments()])
                points = np.round(points, 4)

                # Apply vertical shift
                points[:, 1] -= shift_amount

                # Remove duplicate endpoints (closed polygon)
                if (points[0] == points[-1]).all():
                    points = np.delete(points, -1, axis=0)

                # Simplify polygon by removing collinear points
                if simplify:
                    # Create rolling matrix for curvature calculation
                    rolled_matrix = np.stack([
                        np.roll(points, -1, axis=0),  # Next point
                        points,                        # Current point  
                        np.roll(points, 1, axis=0)     # Previous point
                    ])

                    # Calculate second-order differences (curvature)
                    curvature = np.diff(rolled_matrix, n=2, axis=0)[0]
                    
                    # Remove points with zero curvature (collinear)
                    collinear_indices = np.where(curvature.any(axis=1) == 0)
                    points = np.delete(points, collinear_indices, axis=0)

                # Find suitable marker position inside polygon
                marker_pos = placeMarker(points, polygon_path)
                marker_positions.append(marker_pos)

                # Create PyGIMLi polygon
                # Note: Fault handling could be improved in future versions
                is_closed = True  # Always close polygons for geological units
                
                polygon = mt.createPolygon(
                    points, 
                    addNodes=1,
                    isClosed=is_closed, 
                    interpolate='linear', 
                    marker=unit_count,
                    markerPosition=marker_pos
                )
                
                units.append(polygon)

    # Merge all polygons into single PLC
    world = mt.mergePLC(units, tol=0.00001)
    
    # Align coordinate system with section extent
    section_x_start = model._grid.sections.points[0][0][0]
    x_offset = world.xmin() - section_x_start
    world.translate([-x_offset, 0])

    # Set boundary markers to 0 for proper boundary conditions
    for boundary in world.boundaries():
        if (boundary.center().x() == world.xmin() or 
            boundary.center().x() == world.xmax() or
            boundary.center().y() == world.ymin() or 
            boundary.center().y() == world.ymax()):
            boundary.setMarker(0)

    # Visualization and saving
    if show:
        fig, ax = plt.subplots(figsize=(12, 8))
        pg.show(world, ax=ax, showNodes=False, markers=True, colorBar=True)
        
        # Also show original GemPy model for comparison
        gp.plot_2d(model, show_data=True, direction='y')
        
        # Mark the marker positions
        for i, marker_pos in enumerate(marker_positions):
            ax.plot(marker_pos[0], marker_pos[1], 'o', 
                   label=f'Marker {i+1} at ({marker_pos[0]:.1f}, {marker_pos[1]:.1f})')
        
        ax.legend()
        ax.set_title(f'PyGIMLi PLC from GemPy model - Section: {section}')
        
        if saveIMG:
            fig.savefig(saveIMG, dpi=150, bbox_inches='tight')
        plt.show()
  
    if savePLC:
        mt.exportPLC(world, savePLC)

    return world, mapping

def fromSubsurface(obj, order='C', verbose=False):
    """
    Convert subsurface object to PyGIMLi mesh.
    
    This function converts various subsurface mesh objects into PyGIMLi-compatible
    mesh formats. It supports different data structures commonly used in 
    geological modeling and subsurface characterization.

    Parameters:
    -----------
    obj : subsurface object
        Input mesh object (TriSurf, UnstructuredData, or StructuredData)
    order : str, default='C'
        Memory layout for structured data ('C' for C-style, 'F' for Fortran-style)
        Use 'F' for GemPy meshes to maintain proper ordering
    verbose : bool, default=False
        Enable verbose output during conversion

    Returns:
    --------
    pygimli.Mesh
        Converted PyGIMLi mesh with preserved attributes and geometry

    Supported Objects:
    ------------------
    - TriSurf: Triangulated surface meshes
    - UnstructuredData: 3D boundary representations from triangulated surfaces  
    - StructuredData: 3D cell-centered voxel grids

    Notes:
    ------
    The order parameter is crucial for structured data - use 'F' (Fortran) 
    for GemPy meshes to preserve the correct spatial arrangement of properties.
    
    Example:
    --------
    >>> gempy_mesh = geo_model.solutions.s_regular_grid
    >>> pygimli_mesh = fromSubsurface(gempy_mesh, order='F')
    """
    # Import subsurface with error handling
    ss = pg.optImport('subsurface', 
                      'You need subsurface installed for mesh conversion')

    if isinstance(obj, ss.structs.unstructured_elements.TriSurf):
        # Handle triangulated surfaces by delegating to mesh conversion
        return fromSubsurface(obj.mesh)

    elif isinstance(obj, ss.structs.UnstructuredData):
        # Convert unstructured 3D data (typically boundary representations)
        mesh = pg.Mesh(3)
        
        # Add vertices
        for vertex in obj.vertex:
            mesh.createNode(vertex)

        # Add cells/boundaries
        for cell in obj.cells:
            mesh.createBoundary(cell)

        # Transfer attributes to cells
        for attribute_name, values in obj.attributes_to_dict.items():
            mesh[attribute_name] = np.array(values)

        # Transfer point attributes
        for attribute_name, values in obj.points_attributes_to_dict.items():
            mesh[attribute_name] = np.array(values)

    elif isinstance(obj, ss.structs.StructuredData):
        # Convert structured 3D grid data (voxel-based)
        
        def _voxelCenterToNodes(voxel_centers):
            """Convert voxel centers to node positions."""
            dv = pg.utils.diff(voxel_centers)
            # Calculate node positions at voxel boundaries
            nodes = np.append(voxel_centers[0] - dv[0]/2, 
                            voxel_centers[0] - dv[0]/2 + np.cumsum(dv))
            nodes = np.append(nodes, voxel_centers[-1] + dv[-1]/2)
            return nodes

        # Create structured grid mesh
        mesh = pg.meshtools.createGrid(
            x=_voxelCenterToNodes(obj.data.X.values),
            y=_voxelCenterToNodes(obj.data.Y.values),
            z=_voxelCenterToNodes(obj.data.Z.values)
        )

        # Transfer data variables
        for variable_name, data_array in obj.data.data_vars.items():
            # Flatten arrays with specified order and convert to float
            mesh[variable_name] = [
                np.array(vi.values.flatten(order=order), dtype=float)
                for vi in data_array
            ]
    else:
        print(f"Unsupported object type: {type(obj)}")
        pg.critical('Object type not yet implemented')

    return mesh