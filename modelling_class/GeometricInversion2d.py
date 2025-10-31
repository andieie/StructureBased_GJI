"""
Geometric Inversion 2D Forward Operator

This module contains a class for 2D geometric inversion that recovers geological 
interface positions from geophysical data (Travel Time or ERT).
"""

import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
import numpy as np
import pygimli.physics.ert as ert
import gempy as gp
from utilsGeo import utils, get_gempy_model
import pygimli.physics.traveltime as tt
from gempy.assets import topology as tp
from utilsGeo.plotting_params import scatter_kwargs, set_style, set_legend_white, pg_show_kwargs
set_style()
import os


class GeometricModelling2D(pg.frameworks.Modelling):
    """
    2D Geometric Inversion Forward Operator
    
    This class implements geometric inversion for recovering geological interface 
    positions from geophysical data. It supports both Travel Time and ERT methods
    and allows interface points to be moved in X, Z, or XZ directions.
    
    The forward operator:
    1. Takes interface point positions as model parameters
    2. Updates the geological model geometry
    3. Creates appropriate meshes for geophysical simulation
    4. Computes synthetic geophysical responses
    
    Parameters:
    -----------
    saving_path : str
        Path for saving results and intermediate files
    startgeomodel : gempy.Model
        Initial geological model from GemPy
    method : str
        Geophysical method ('TravelTime' or 'ERT')
    dirtype : str
        Direction of interface movement ('X', 'Z', or 'XZ')
    param_mode : str
        Parameter mode ('homogenous' or 'heterogenous')
    section : list
        Cross-section coordinates for 2D model extraction
    dta : pygimli.DataContainer
        Geophysical data container
    interpoints : array-like
        Starting interface point positions
    move_points : array-like
        Indices of interface points to be moved during inversion
    paramdict : dict
        Master dictionary mapping geological units to [cell_id, parameter_value, interface_points]
        Structure: {lithology: [unit_id, physical_property, surface_points]}
    cons : list
        Constraint values for regularization
    paracons : list, optional
        Parameter constraints for physical property inversion
    verbose : bool, default=True
        Enable verbose output
    showInter : bool, default=False
        Show intermediate results during inversion
    trueModel : dict, optional
        True model interface positions for comparison
    saveJacobian : bool, default=True
        Whether to save Jacobian matrices to disk
    saveMesh : bool, default=True
        Whether to save mesh files to disk
    """
    def __init__(self,
                 saving_path,
                 startgeomodel,
                 method,
                 dirtype,
                 param_mode,
                 section,
                 dta,
                 interpoints,
                 move_points,
                 paramdict,
                 cons,
                 paracons,
                 verbose=True,
                 showInter=False,
                 trueModel=None,
                 saveJacobian=True,
                 saveMesh=True):
        """Initialize the Geometric Modelling 2D forward operator."""
        super().__init__()
        
        # Initialize geological model and topology
        self.initplc, self.initmap = get_gempy_model.get_geometry_2d(
            startgeomodel, section, simplify=True, resample=False)
        self.edges_ori, self.centroids_ori = tp.compute_topology(startgeomodel)
        
        # Setup data and sensors based on method
        self._setup_data_and_sensors(method, dta)
        
        # Core model parameters
        self.move_points = move_points
        self.initgeomodel = startgeomodel
        self.geomodel = startgeomodel
        self.method = method
        self.start = interpoints
        self.section = section
        self.dirtype = dirtype
        
        # Inversion parameters
        self.param_mode = param_mode
        self.lambda_param = paracons[0] if paracons else 100
        self.verbose = verbose
        self.showInter = showInter
        self.trueModel = trueModel
        self.saveJacobian = saveJacobian
        self.saveMesh = saveMesh
        
        # Setup paths and directories
        self.path = saving_path
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + '/images', exist_ok=True)
        
        # Initialize tracking variables
        self._init_tracking_variables()
        
        # Setup parameter mapping
        self._setup_parameter_mapping(paramdict)
        
        # Setup model dimensions and constraints
        self._setup_model_dimensions(interpoints, dirtype, cons)
        
        # Initialize mesh and coverage
        self._initialize_mesh_and_coverage()
        
        # Initialize physical parameters with starting values
        self.physical_param[0] = self.paramap.copy()
        self.new_paramap = self.paramap.copy()

        print("Initialized Geometric Modelling 2D Forward Operator")    
        
    def _setup_data_and_sensors(self, method, dta):
        """Setup data and sensor configurations based on geophysical method."""
        if method == 'TravelTime':
            self.dta = dta['t']  # Travel time data
            # Separate receivers and shots based on position
            receivers = []
            shots = []
            for sensor in np.array(dta.sensors()):
                if sensor[0] > 6:
                    shots.append(sensor)
                else:
                    receivers.append(sensor)
            self.shots = np.array(shots)
            self.receivers = np.array(receivers)
        elif method == 'ERT':
            self.dta = dta['rhoa']  # Apparent resistivity data
            sensors = np.array(dta.sensors())
            dta.setSensors(sensors)
        
        self.scheme = dta  # Measurement scheme

    def _init_tracking_variables(self):
        """Initialize variables for tracking inversion progress."""
        self.rejected_models = 0
        self.testing = 0
        self.geo_models = {'start': self.initgeomodel}
        self.coverage_dict = {}
        self.meshes_dict = {}
        self.model_dict = {}
        self.response_count = 0
        self.iter_rejected = []
        self.param_dict = {}
        self.param_chi2_dict = {}
        self.new_paramap = {}
        self.physical_param = {}
        self.physical_param_mesh = {}
        self.jacobians = {}
        self.responses = {}
        self.jrun = 0
        self.irun = 0
        self.rrun = 0
        self.r0 = 0

    def _setup_parameter_mapping(self, paramdict):
        """Setup parameter mapping for geological units."""
        self.paramap = {}
        self.ipoints = {}
        self.paramdict = paramdict
        
        for keys, values in paramdict.items():
            self.paramap[values[0][0]] = values[1]  # Markers for lithology (parameter values)
            self.ipoints[keys] = values[2]          # Interface points
        
        self.all_points = np.array(sum(self.ipoints.values(), []))

    def _setup_model_dimensions(self, interpoints, dirtype, cons):
        """Setup model dimensions and constraints."""
        self.n_units = len(self.initmap.values())  # Number of geological units
        self.n_params = len(interpoints)
        
        if dirtype in ['X', 'Y', 'Z']:
            self.n_sps = len(interpoints)
        else:
            self.n_sps = int(len(interpoints) / 2)
        
        # Setup constraints
        if len(cons) == 2:
            self.conx = cons[0]
            self.conz = cons[1]
        else:
            self.cons = cons[0]

    def _initialize_mesh_and_coverage(self):
        """Initialize mesh and calculate coverage."""
        # Create initial mesh with sensors
        print('CREATING INITIAL MESH')
        for sensor in self.scheme.sensorPositions().array():
            sensor_exists = False
            for node in self.initplc.nodes():
                if np.allclose(sensor, node.pos().array()):
                    sensor_exists = True
                    break
            if not sensor_exists:
                self.initplc.createNode(sensor)
        
        self.mesh = mt.createMesh(self.initplc, quality=20, area=0.2)
        
        # Store initial point positions
        self.pX = np.array(self.initgeomodel.surface_points.df["X"])
        self.pY = np.array(self.initgeomodel.surface_points.df["Y"])
        self.pZ = np.array(self.initgeomodel.surface_points.df["Z"])
        
        # Store model bounds
        (self.initgeomodelxmin, self.initgeomodelxmax,
         self.initgeomodelymin, self.initgeomodelymax,
         self.initgeomodelzmin, self.initgeomodelzmax) = self.initgeomodel.grid.regular_grid.extent
        
        # Setup fixed points for visualization
        self.orientations = self.geomodel.orientations
        self.dfpoints = {'start': self.geomodel.surface_points.df}
        move_points_all = self.dfpoints['start'].iloc[self.move_points]
        self.fixed_points = self.dfpoints['start'][
            ~self.dfpoints['start'].index.isin(move_points_all.index) & 
            (self.dfpoints['start']['Y'] == 10)
        ][['X', 'Z']].to_numpy()
        
        # Calculate coverage
        self.coverage, self.coverage_mesh = self.getCoverage()
        plt.close('all')
        
        # Print storage options
        if self.verbose:
            print(f"Storage options: Jacobian saving={'ON' if self.saveJacobian else 'OFF'}, "
                  f"Mesh saving={'ON' if self.saveMesh else 'OFF'}")
            if not self.saveJacobian:
                print("  Note: Jacobian matrices will not be saved to disk (saves storage space)")
            if not self.saveMesh:
                print("  Note: Mesh files will not be saved to disk (saves storage space)")

    def createStartModel(self, dataVals):
        """
        Create the starting model for the geometric inversion.
        
        Returns:
        --------
        pg.Vector
            Starting model consisting of interface point positions
        """
        return pg.Vector(self.start)
    
    def getPoints(self, geo_model, stack=False):
        """
        Extract surface point coordinates from a geological model.
        
        Parameters:
        -----------
        geo_model : gempy.Model
            Geological model from GemPy
        stack : bool, default=False
            If True, return stacked coordinates as array
            
        Returns:
        --------
        tuple or array
            Point coordinates (x, y, z) or stacked array if stack=True
        """
        x = np.array(geo_model.surface_points.df['X'])
        y = np.array(geo_model.surface_points.df['Y'])
        z = np.array(geo_model.surface_points.df['Z'])
        
        if stack:
            return np.stack([x, y, z], axis=1)
        else:
            return x, y, z
        
    def check_high_value(self, array, threshold_factor=2.0, method='mean'):
        """
        Identify values in array that exceed threshold relative to central tendency.
        
        Parameters:
        -----------
        array : array-like
            Input array to analyze
        threshold_factor : float, default=2.0
            Factor to multiply central tendency for threshold
        method : str, default='mean'
            Method for central tendency ('mean' or 'median')
            
        Returns:
        --------
        array
            Indices of values exceeding threshold
        """
        if method == 'mean':
            compare_value = np.mean(array)
        elif method == 'median':
            compare_value = np.median(array)
        else:
            raise ValueError("Invalid method. Use 'mean' or 'median'.")

        threshold = threshold_factor * compare_value
        return np.where(array > threshold)[0]

    def calculateVerticalJumps(self):
        """
        Calculate vertical differences between consecutive interface points.
        
        Returns:
        --------
        list
            Vertical differences between consecutive points
        """
        diffs = []
        zmove = self.pZ[self.move_points]
        for i in range(len(zmove) - 1):
            diffs.append(zmove[i] - zmove[i+1])
        return diffs
    

    def constrainSPS(self, val, start, stop):
        """
        Create constraint matrix for surface point smoothness (SPS).
        
        Generates constraints to regularize interface point movements based on
        the specified constraint value and direction type.
        
        Parameters:
        -----------
        val : str or float
            Constraint value. If string starts with 'I', applies identity constraint.
            Otherwise applies smoothness constraint between adjacent points.
        start : int
            Starting parameter index
        stop : int
            Stopping parameter index
            
        Returns:
        --------
        list
            List of constraint rows for the constraint matrix
        """
        out = []
        
        if self.dirtype != 'XZ':
            # Single direction constraints (X, Y, or Z only)
            if str(val)[0] == 'I':
                # Identity constraints - tie to starting model
                for pm in range(int(start), int(stop)):
                    rnew = np.zeros(int(stop))
                    rnew[pm] = float(val[1:])
                    print(rnew, 'constraints for SPS geometric ')
                    out.append(rnew)
            else:
                # Smoothness constraints between adjacent points
                if len(np.unique(self.dfpoints.get('start')['surface'])) > 1:
                    print('DETECTED MULTIPLE SURFACES - APPLYING SURFACE-AWARE CONSTRAINTS')
                    list_of_surfaces_move_points = []
                    for move_point in self.move_points:
                        surface = self.dfpoints.get('start').loc[move_point, 'surface']
                        list_of_surfaces_move_points.append(surface)
                        
                    for i, pm in enumerate(range(start, stop-1, 1)):
                        row = np.zeros(int(stop))
                        # Only apply constraints within the same surface
                        if (i < len(list_of_surfaces_move_points) - 1 and 
                            list_of_surfaces_move_points[i] == list_of_surfaces_move_points[i+1]):
                            row[pm] = -int(val)
                            row[pm+1] = int(val)
                        else:
                            row[pm] = 0
                        out.append(row)
        else:
            # XZ direction constraints (both X and Z coordinates)
            if str(val)[0] == 'I':
                # Identity constraints for XZ pairs
                for pm in range(int(start), int(stop), 2):
                    rnew = np.zeros(int(stop))
                    rnew[pm] = float(val[1:])
                    out.append(rnew)
            else:
                # Smoothness constraints for XZ pairs
                if len(np.unique(self.dfpoints.get('start')['surface'])) > 1:
                    print('DETECTED MULTIPLE SURFACES - APPLYING XZ SURFACE-AWARE CONSTRAINTS')
                    list_of_surfaces_move_points = []
                    for move_point in self.move_points:
                        surface = self.dfpoints.get('start').loc[move_point, 'surface']
                        list_of_surfaces_move_points.extend([surface, surface])  # XZ pair
                        
                    for i, pm in enumerate(range(start, stop-1, 2)):
                        row = np.zeros(int(stop))
                        # Only apply constraints within the same surface
                        if (i*2 < len(list_of_surfaces_move_points) - 2 and 
                            list_of_surfaces_move_points[i*2] == list_of_surfaces_move_points[i*2+2]):
                            row[pm] = -int(val)
                            row[pm+2] = int(val)
                        out.append(row)
                else:
                    # Simple smoothness for single surface
                    for pm in range(start, stop-1, 2):
                        row = np.zeros(self.n_params)
                        row[pm] = -val
                        row[pm+2] = val
                        out.append(row)
                        
        return out

    def createConstraints(self):
        """
        Create constraint matrix for geometric inversion regularization.
        
        Sets up smoothness constraints for interface point movements based on
        the direction type (X, Z, or XZ) and constraint values provided during
        initialization.
        """
        self.C = pg.Matrix(0, self.n_params)

        if self.dirtype == 'XZ':        
            utils.message(f'Creating constraints for {self.dirtype}')
            
            # Create constraints for X and Z directions separately
            cx = self.constrainSPS(self.conx, 0, self.n_params)
            cz = self.constrainSPS(self.conz, 1, self.n_params)
            
            # Save constraint matrices for debugging
            np.save(self.path + '/cx', cx)
            np.save(self.path + '/cz', cz)
            
            # Add constraints to matrix alternating X and Z
            for i, j in zip(cx, cz):
                self.C.push_back(i)
                self.C.push_back(j)
        else:
            # Single direction constraints
            c = self.constrainSPS(self.cons, 0, self.n_params)
            for r in c:
                self.C.push_back(r)
        
        # Save and set constraints
        np.save(self.path + "/constraints", self.C)
        self.setConstraints(self.C)


    def shiftIP(self, shifts):
        """
        Shift interface points while ensuring they remain within model bounds.
        
        Parameters:
        -----------
        shifts : array-like
            Displacement values for interface points. For XZ mode, this should
            contain alternating X and Z displacements.
            
        Returns:
        --------
        array or tuple
            Shifted point positions. Returns tuple (sX, sZ) for XZ mode,
            single array for other modes.
        """
        if self.dirtype == 'XZ':
            shifts = self.splitvector(shifts)
            
        true_pointsz = self.pZ[self.move_points]
        true_pointsx = self.pX[self.move_points]
        
        if self.dirtype == 'XZ':
            sX = true_pointsx + np.array(shifts[0])
            sZ = true_pointsz + np.array(shifts[1])
            
            # Check bounds for X coordinates
            for i, sxi in enumerate(sX):
                if sxi <= self.initgeomodelxmin or sxi >= self.initgeomodelxmax:
                    print(f"X coordinate {sxi} out of bounds, keeping original")
                    sX[i] = true_pointsx[i]
                    
            # Check bounds for Z coordinates
            for i, szi in enumerate(sZ):
                if szi <= self.initgeomodelzmin or szi >= self.initgeomodelzmax:
                    print(f"Z coordinate {szi} out of bounds, keeping original")
                    sZ[i] = true_pointsz[i]
                    
        elif self.dirtype == 'X':
            s = true_pointsx + np.array(shifts)
            for i, sxi in enumerate(s):
                if sxi <= self.initgeomodelxmin or sxi >= self.initgeomodelxmax:
                    print(f"X coordinate {sxi} out of bounds, keeping original")
                    s[i] = true_pointsx[i]
                    
        elif self.dirtype == 'Y':
            s = self.pY[self.move_points] + np.array(shifts)
            for i, syi in enumerate(s):
                if syi <= self.initgeomodelymin or syi >= self.initgeomodelymax:
                    print(f"Y coordinate {syi} out of bounds, keeping original")
                    s[i] = self.pY[self.move_points][i]
                    
        elif self.dirtype == 'Z':
            s = true_pointsz + np.array(shifts)
            for i, szi in enumerate(s):
                if szi <= self.initgeomodelzmin or szi >= self.initgeomodelzmax:
                    print(f"Z coordinate {szi} out of bounds, keeping original")
                    s[i] = true_pointsz[i]

        # Print diagnostic information
        print("ORIGINAL INTERFACE POINTS:")
        print([[np.round(x, 1), np.round(z, 1)] 
               for x, z in zip(true_pointsx, true_pointsz)])
        
        print("SHIFTED INTERFACE POINTS:")
        if self.dirtype == 'XZ':
            print([[np.round(x, 1), np.round(z, 1)] for x, z in zip(sX, sZ)])
            return sX, sZ
        else:
            print([np.round(val, 1) for val in s])
            return s

    def get_other_points(self, same='X'):
        """
        Find other points in the same geological surface that should move together.
        
        Parameters:
        -----------
        same : str, default='X'
            Coordinate to group by ('X', 'Y', or 'Z')
            
        Returns:
        --------
        dict
            Dictionary mapping move point indices to lists of other point indices
            in the same group
        """
        # Group points by coordinate and surface
        grouped = self.geomodel.surface_points.df.groupby([same, 'surface'])
        
        other_points = {}
        for move_point in self.move_points:
            try:
                group_key = (self.geomodel.surface_points.df.loc[move_point, same], 
                           self.geomodel.surface_points.df.loc[move_point, 'surface'])
                group = grouped.get_group(group_key)
                other_points[move_point] = group.index.drop(move_point).tolist()
            except KeyError:
                # Handle case where group doesn't exist
                other_points[move_point] = []
                
        return other_points


    def makeMesh(self, model):
        """
        Create mesh from updated geological model with shifted interface points.
        
        Parameters:
        -----------
        model : array-like
            Updated interface point positions
            
        Returns:
        --------
        tuple
            (mesh, plc) - Generated mesh and piece-wise linear complex
        """
        sP = model
        
        # Update geological model with new interface positions
        if self.dirtype == 'XZ':
            # Handle both X and Z coordinates
            other_points = self.get_other_points(same='X')
            for i, n in enumerate(self.move_points):
                self.geomodel.modify_surface_points(n, Z=sP[1][i], X=sP[0][i])
                # Update other points in the same group
                if n in other_points.keys():
                    for other_n in other_points[n]:
                        self.geomodel.modify_surface_points(other_n, Z=sP[1][i], X=sP[0][i])
                        
        elif self.dirtype == 'X':
            # Handle X coordinate only
            other_points = self.get_other_points(same='X')
            for i, n in enumerate(self.move_points):
                self.geomodel.modify_surface_points(n, X=sP[i])
                if n in other_points.keys():
                    for other_n in other_points[n]:
                        self.geomodel.modify_surface_points(other_n, X=sP[i])
                        
        elif self.dirtype == 'Z':
            # Handle Z coordinate only
            other_points = self.get_other_points(same='X')
            for i, n in enumerate(self.move_points):
                self.geomodel.modify_surface_points(n, Z=sP[i])
                if n in other_points:
                    for other_n in other_points[n]:
                        self.geomodel.modify_surface_points(other_n, Z=sP[i])
        
        # Update geological model parameters and compute
        self.geomodel.modify_kriging_parameters('range', 10)
        self.geomodel.update_to_interpolator()
        gp.compute_model(self.geomodel)
        
        # Store updated points for tracking
        self.dfpoints[self.irun] = self.geomodel.surface_points.df
        
        # Check and potentially reject the model
        if self.RejectModels(self.geomodel):
            self.rrun += 1
        else:
            self.geo_models[self.irun] = [
                gp.get_data(self.geomodel, 'surface_points'), 
                gp.get_data(self.geomodel, 'orientations')
            ]
        
        # Generate mesh from updated model
        plc, paraid = get_gempy_model.get_geometry_2d(
            self.geomodel, self.section, simplify=True, resample=False)
        
        print('CREATING MESH FROM UPDATED MODEL')
        
        # Add sensors to mesh if not already present
        existing_node_positions = {tuple(node.pos().array()) for node in plc.nodes()}
        sensors_added = 0
        
        for sensor in self.scheme.sensorPositions().array():
            sensor_pos = tuple(sensor)
            if sensor_pos in existing_node_positions:
                print('Sensor already exists at position:', sensor_pos)
                pass
            else:
                i += 1
                plc.createNode(sensor)
                existing_node_positions.add(sensor_pos)
                sensors_added += 1
        
        if sensors_added > 0:
            print(f'Added {sensors_added} sensor nodes to mesh')
        
        mesh = mt.createMesh(plc, quality=20, area=0.2)
        print('MESH CREATED SUCCESSFULLY')
        
        # Print diagnostic information
        print(f'Original Z points: {np.round(self.pZ[self.move_points], 3)}')
        print(f'Modified Z points: {np.round(np.array(self.geomodel.surface_points.df["Z"])[self.move_points], 3)}')
        print(f'Original X points: {np.round(self.pX[self.move_points], 3)}')
        print(f'Modified X points: {np.round(np.array(self.geomodel.surface_points.df["X"])[self.move_points], 3)}')
        
        return mesh, plc
    
    def splitvector(self, model):
        """
        Split XZ parameter vector into separate X and Z components.
        
        Parameters:
        -----------
        model : array-like
            Combined XZ parameter vector (x1, z1, x2, z2, ...)
            
        Returns:
        --------
        tuple
            (x_components, z_components) - Separated X and Z coordinates
        """
        z = model[1::2]  # Every second element starting from index 1
        x = model[::2]   # Every second element starting from index 0
        return x, z
 
    def makeParavec(self, mesh, paramap):
        """
        Create parameter vector from mesh and parameter mapping.
        
        Parameters:
        -----------
        mesh : pygimli.Mesh
            Computational mesh
        paramap : dict
            Mapping from cell markers to parameter values

            
        Returns:
        --------
        array
            Parameter vector mapped to mesh cells
        """
        cells = []
        for ids, paramvalues in paramap.items():
            temp = [ids, paramvalues]
            cells.append(temp)
        map = pg.solver.parseMapToCellArray(cells, mesh)
        return map
    
    def create_array_from_mesh_and_dict(self, mesh, paramap):
        """
        Create parameter array from mesh cell markers and parameter dictionary.
        
        Parameters:
        -----------
        mesh : pygimli.Mesh
            Computational mesh
        paramap : dict
            Parameter mapping dictionary
            
        Returns:
        --------
        list
            Parameter values corresponding to mesh cells
        """
        cellMarkers = mesh.cellMarkers()
        array = [None] * len(cellMarkers)
        
        # Map parameters to cells based on markers
        for i, marker in enumerate(cellMarkers):
            if marker != 0 and marker in paramap:
                array[i] = paramap[marker]
                
        # Filter out None values
        return [value for value in array if value is not None]


    def RejectModels(self, geomodel):
        """
        Check if geological model should be rejected based on topology criteria.
        
        Models are rejected if:
        1. Geological topology changes (different layer ordering)
        2. Centroid positions change too drastically
        
        Parameters:
        -----------
        geomodel : gempy.Model
            Updated geological model to evaluate
            
        Returns:
        --------
        bool
            True if model should be rejected, False otherwise
        """
        centroid_cutoff = 75
        
        # Calculate topology differences
        edges, centroids = tp.compute_topology(geomodel)
        cent_diff = 0
        
        if len(centroids) == len(self.centroids_ori):
            for k in range(0,self.n_units-1):
                cent_diff += np.sqrt(np.sum((list(self.centroids_ori.values())[k] - list(centroids.values())[k])**2))
        else:
            cent_diff = 100  # Large penalty for different number of units
        
        print(f'Centroid difference: {cent_diff}')
        print(f'New edges: {edges}')
        
        # Calculate Jaccard distance for topology similarity
        j = tp.jaccard_index(self.edges_ori, edges)
        distance = 1 - j
        print(f'Topology distance: {distance}')
        
        # Reject if topology changed or centroids moved too much
        if (edges != self.edges_ori) or (cent_diff > centroid_cutoff):
            if edges != self.edges_ori:
                print('REJECTING: Geological layer ordering changed')
            elif cent_diff > centroid_cutoff:
                print('REJECTING: Centroid difference too high')
                
            pg.boxprint('MODEL REJECTED - REVERTING TO PREVIOUS STATE')
            
            # Revert to previous valid model
            all_index = np.array(self.geomodel.surface_points.df.index)
            
            if self.jrun == 0:
                # First run - revert to initial model
                self.geomodel.modify_surface_points(all_index, Z=self.pZ)
                self.geomodel.modify_kriging_parameters('range', 10)
                gp.compute_model(self.geomodel)
            else:
                # Later runs - revert to previous iteration
                if self.irun == 0:
                    dflast = self.dfpoints.get('start')
                else:
                    dflast = self.dfpoints.get(self.irun - 1)
                    
                dflastz = np.array(dflast['Z'])
                dflastx = np.array(dflast['X'])
                
                self.geomodel.modify_surface_points(all_index, Z=dflastz, X=dflastx)
                self.geomodel.modify_kriging_parameters('range', 10)
                gp.compute_model(self.geomodel)
            
            self.rejected_models += 1
            self.iter_rejected.append(self.irun)
            return True
        else:
            # Model accepted
            self.geomodel = geomodel
            return False

    def checkModel(self, r0, rN):
        """
        Compare model responses to assess improvement in data fit.
        
        Parameters:
        -----------
        r0 : array-like
            Original response
        rN : array-like
            New response
            
        Returns:
        --------
        float
            Difference in RMSE (positive means improvement)
        """
        og_misfit = np.sqrt(np.mean((r0 - self.dta)**2))
        new_misfit = np.sqrt(np.mean((rN - self.dta)**2))
        diff = (og_misfit - new_misfit) * 1e6
        print(f'RMSE difference: {np.round(diff, 2)}')
        return np.round(diff, 2)

    def getCoverage(self):
        """
        Calculate data coverage for the current sensor configuration.
        
        Performs a simple inversion to determine how well the data constrains
        different parts of the model domain.
        
        Returns:
        --------
        tuple
            (coverage, coverage_mesh) - Coverage values and corresponding mesh
        """
        # Setup manager based on method
        if self.method == 'ERT':
            mgr = ert.ERTManager(data=self.scheme)
            sensors = self.scheme.sensors()
        elif self.method == 'TravelTime':
            mgr = tt.TravelTimeManager(data=self.scheme)
            sensors = self.scheme.sensors()
        
        # Create rectangular domain for coverage calculation
        rect = mt.createRectangle(
            [self.initgeomodelxmin, self.initgeomodelzmin], 
            [self.initgeomodelxmax, self.initgeomodelzmax], 
            marker=0
        )
        
        # Add sensor positions to domain
        for sensor in sensors:
            rect.createNode(sensor)
            
        mesh = mt.createMesh(rect, quality=31)
        mgr.setMesh(mesh)
        
        # Perform simple inversion to get coverage
        mgr.invert(lam=1000, zWeight=1.0, useGradient=False)
        
        if self.method == 'TravelTime':
            return mgr.rayCoverage(), mesh
        else:
            return mgr.coverage(), mesh


    def invertForParaMap(self, mesh, plc, homo=False):
        """
        Invert for physical parameters on the current mesh geometry.
        
        This performs a geophysical inversion to determine physical parameters
        (velocity/resistivity) for the current geological model geometry.
        
        Parameters:
        -----------
        mesh : pygimli.Mesh
            Computational mesh for inversion
        plc : pygimli.PLC
            Piece-wise linear complex geometry
        homo : bool, default=False
            Whether to use homogeneous parameter mode
            
        Returns:
        --------
        dict or array
            Physical parameters for homogeneous mode (dict) or 
            heterogeneous parameters (array)
        """
        # Setup geophysical manager
        if self.method == 'ERT':
            mgr = ert.ERTManager(data=self.scheme)
            mgr.setMesh(mesh)
        elif self.method == 'TravelTime':
            mgr = tt.TravelTimeManager(data=self.scheme) 
            mgr.setMesh(mesh)
        
        if self.param_mode == 'homogenous':
            pg.boxprint('Inverting for homogeneous physical parameters')
            
            # Setup regularization for homogeneous inversion
            if self.method == 'TravelTime':
                mgr.inv.setRegularization(background=False, cType=1)
                for region, value in self.paramap.items():
                    mgr.inv.setRegularization(region, single=True)     
                resulting_paraid = mgr.invert(
                    lam=self.lambda_param, useGradient=False, maxIter=3)
            else:
                mgr.inv.setRegularization(background=False)
                for region, value in self.paramap.items():
                    mgr.inv.setRegularization(region, single=True)  
                resulting_paraid = mgr.invert(
                    lam=self.lambda_param, verbose=True, background=False, maxIter=3)
            
            # Extract unique values for each region
            new_paramap = {}
            for region in self.paramap.keys():
                if self.method == 'ERT':
                    new_values = np.unique(
                        resulting_paraid[mgr.paraDomain.cellMarkers() == int(region-1)])
                else:
                    new_values = np.unique(
                        resulting_paraid[mesh.cellMarkers() == int(region)])
                new_paramap[region] = new_values[0]
            
            # Store results
            self.physical_param[self.irun] = new_paramap
            self.physical_param_mesh[self.irun] = mesh
            
            if self.irun != 0 and (self.irun-1) in self.physical_param:
                print(f'Previous parameter values: {self.physical_param[self.irun-1]}')
                print(f'New parameter values: {new_paramap}')
            elif self.irun == 0:
                print(f'Initial parameter values: {new_paramap}')
                
            return new_paramap
            
        else: 
            pg.boxprint('Inverting for heterogeneous physical parameters')
            
            # Create extended mesh with boundary
            big_mesh = mt.appendBoundary(
                mesh, marker=0, xbound=2, ybound=2, isSubsurface=False)
            mgr.setMesh(big_mesh)
            mgr.inv.setRegularization(background=False)
            
            # Perform heterogeneous inversion
            resulting_paraid = mgr.invert(
                lam=self.lambda_param, verbose=True, robustData=True, maxIter=3)
            
            # Extract parameter domain (non-background cells)
            paraDomain = big_mesh.createSubMesh(
                big_mesh.cells(big_mesh.cellMarkers() != 0))
            resulting_paraid = resulting_paraid[big_mesh.cellMarkers() != 0]
            
            # Store results
            self.physical_param_mesh[self.irun] = paraDomain
            self.physical_param[self.irun] = resulting_paraid
            self.param_chi2_dict[self.irun] = mgr.inv.chi2()
            
            return resulting_paraid
    
    def joinVector(self, x_components, z_components):
        """
        Join separate X and Z component vectors into combined XZ vector.
        
        Parameters:
        -----------
        x_components : array-like
            X coordinate values
        z_components : array-like  
            Z coordinate values
            
        Returns:
        --------
        list
            Combined vector in format [x1, z1, x2, z2, ...]
        """
        assert len(x_components) == len(z_components), \
            "X and Z components must have the same length"
        
        combined_vector = []
        for x, z in zip(x_components, z_components):
            combined_vector.extend([x, z])
        return combined_vector

    def response(self, model, jacobianCalc=False):
        """
        Calculate geophysical response for given interface positions.
        
        This is the main forward modeling function that:
        1. Updates geological model with new interface positions
        2. Creates appropriate mesh
        3. Inverts for physical parameters (if needed)
        4. Simulates geophysical data
        
        Parameters:
        -----------
        model : array-like
            Interface point positions
        jacobianCalc : bool, default=False
            Whether this is for Jacobian calculation (affects parameter handling)
            
        Returns:
        --------
        pygimli.Vector
            Synthetic geophysical response
        """
        pg.tic("response start")
        
        # Create mesh with updated geometry
        if jacobianCalc:
            print('Creating response for Jacobian calculation')
            if self.dirtype == 'XZ':
                model = self.splitvector(model)
            mesh, plc = self.makeMesh(model)
        else:
            print('Creating response for inversion')
            sP = self.shiftIP(model)
            model = sP
            mesh, plc = self.makeMesh(model)
            # Save mesh if requested
            if self.saveMesh:
                mesh.save(self.path + f'/mesh_{self.irun}_{self.jrun}.bms')

        # Handle physical parameter assignment
        if jacobianCalc:
            print('Using existing physical parameters for Jacobian')
            if self.param_mode != 'homogenous':
                # Use interpolated heterogeneous parameters
                self.modelv = mt.interpolate(
                    mesh, self.physical_param_mesh[self.irun], 
                    self.physical_param[self.irun])
            else:
                # Use homogeneous parameters
                if self.irun == 0:
                    self.modelv = self.makeParavec(mesh, self.paramap) 
                else:
                    self.modelv = self.makeParavec(mesh, self.new_paramap)
        else:
            # Update physical parameters during inversion
            self.response_count += 1 
            if self.param_mode == 'homogenous':
                # Update homogeneous parameters periodically
                if self.response_count % 2 != 0:
                    self.new_paramap = self.invertForParaMap(mesh, plc, homo=True)
                self.modelv = self.makeParavec(mesh, self.new_paramap)
            else:
                # Update heterogeneous parameters
                modelv_param = self.invertForParaMap(mesh, plc, homo=False)
                self.modelv = modelv_param
                
            # Store results
            self.meshes_dict[self.irun] = mesh
            self.model_dict[self.irun] = model
            self.param_dict[self.irun] = self.modelv 
            
            # Validate mesh-parameter consistency
            if mesh.cellCount() != len(self.modelv):
                pg.boxprint('ERROR: Mesh cell count does not match parameter vector length')
        
        # Create visualization if requested
        if self.showInter and not jacobianCalc:
            self._create_iteration_plot(mesh, model)
        
        # Simulate geophysical response
        print(f'Parameter statistics: min={np.min(self.modelv):.2f}, '
              f'max={np.max(self.modelv):.2f}, unique_values={len(np.unique(self.modelv))}')
        
        if np.any(self.modelv == 0):
            print('WARNING: Zero values detected in parameter vector')
            responsev = np.zeros_like(self.r0) if hasattr(self, 'r0') else np.zeros(len(self.dta))
        else:
            responsev = self._simulate_geophysical_data(mesh)
            
        pg.toc("response done")
        return responsev

    def _create_iteration_plot(self, mesh, model):
        """Create visualization plot for current iteration."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot mesh with parameters
        if self.method == 'TravelTime':
            pg.show(mesh, self.modelv, ax=ax, **pg_show_kwargs['tt'])
            ax.scatter(self.shots[:,0], self.shots[:,1], **scatter_kwargs['shots'])
            ax.scatter(self.receivers[:,0], self.receivers[:,1], **scatter_kwargs['receivers'])
        else:
            pg.show(mesh, self.modelv, ax=ax)            
            ax.scatter(self.scheme.sensorPositions()[:,0], 
                      self.scheme.sensorPositions()[:,1], **scatter_kwargs['ert_sensors'])
        
        # Plot interface points
        actual_points_x = self.pX[self.move_points]
        actual_points_z = self.pZ[self.move_points]
        
        # Plot true model if available
        if self.trueModel is not None:
            true_model_points = np.vstack([points for points in self.trueModel.values()])
            ax.scatter(true_model_points[:, 0], true_model_points[:, 1], 
                      **scatter_kwargs['true_positions'])
        
        # Plot starting and current points
        ax.scatter(actual_points_x, actual_points_z, 
                  s=50, c='gray', edgecolor='black', label='Starting points')
        
        if self.dirtype == 'XZ':
            current_points = self.model_dict.get(self.irun)
            ax.scatter(current_points[0], current_points[1], **scatter_kwargs['moving_points'])
        else:
            ax.scatter(actual_points_x, self.model_dict.get(self.irun), 
                      **scatter_kwargs['moving_points'])
        
        # Plot fixed points
        ax.scatter(self.fixed_points[:,0], self.fixed_points[:,1], 
                  **scatter_kwargs['fixed_points'])
        
        # Set plot properties
        ax.set_xlim(0, 15)
        ax.set_ylim(-5, 25)
        ax.set_title(f'Iteration: {self.irun}')
        set_legend_white(ax)
        
        # Save plot
        fig.savefig(self.path + f'/images/{self.irun}_{self.jrun}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _simulate_geophysical_data(self, mesh):
        """Simulate geophysical data for current model."""
        if self.method == 'TravelTime':
            mgr = tt.TravelTimeManager()
            pg.tic('Simulating travel time data')
            response = mgr.simulate(
                mesh,
                slowness=1/self.modelv,
                scheme=self.scheme,
                noiseLevel=0,
                noiseAbs=0,
                seed=1887
            )
            pg.toc('Travel time simulation complete')
            return pg.Vector(response["t"])
            
        elif self.method == 'ERT':
            mgr = ert.ERTManager()
            pg.tic('Simulating ERT data')
            response = mgr.simulate(
                mesh,
                res=self.modelv,
                scheme=self.scheme,
                noiseLevel=0,
                noiseabs=0,
                seed=1887
            )
            pg.toc('ERT simulation complete')
            return pg.Vector(response["rhoa"])
    

    def check_and_adjust_points(self, model, min_distance_threshold=0.5):
        """
        Check if interface points are too close and adjust step size if needed.
        
        Parameters:
        -----------
        model : array-like
            Current model parameters (interface positions)
        min_distance_threshold : float, default=0.5
            Minimum allowed distance between interface points
            
        Returns:
        --------
        tuple
            (adjusted_factor, too_close_flag) - Updated step factor and flag
        """
        too_close = False
        distances = []
        
        # Get current move points dataframe
        move_points_df = self.dfpoints.get(self.irun).loc[self.move_points]
        surface_counts = move_points_df.groupby('surface').size()
        
        # Organize points by surface
        surface_points_dict = {}
        start_index = 0
        
        for surface_name, count in surface_counts.items():
            if self.dirtype == 'XZ':
                # Each point represented by 2 values (x, z)
                surface_points = [model[i:i+2] for i in range(start_index, start_index + count*2, 2)]
                start_index += count * 2
            else:
                # Each point represented by single value
                surface_points = model[start_index:start_index + count]
                start_index += count
            surface_points_dict[surface_name] = surface_points
        
        # Check distances between points on different surfaces
        surface_names = list(surface_points_dict.keys())
        for i in range(len(surface_names)):
            for j in range(i+1, len(surface_names)):
                surface_1_points = surface_points_dict[surface_names[i]]
                surface_2_points = surface_points_dict[surface_names[j]]
                
                for point1 in surface_1_points:
                    for point2 in surface_2_points:
                        if self.dirtype == 'XZ':
                            distance = np.linalg.norm(np.array(point1) - np.array(point2))
                        else:
                            distance = abs(point1 - point2)
                            
                        if np.round(distance, 2) <= min_distance_threshold:
                            too_close = True
                            break
                        distances.append(distance)
                    if too_close:
                        break
                if too_close:
                    break
        
        print(f"Inter-surface distances for iteration {self.jrun}: {np.round(distances, 3)}")
        
        # Adjust step factor if points are too close
        if too_close and (self.fak > 1):
            new_fak = np.round(1/self.fak, 2)
            print(f"Points too close, adjusting step factor: {self.fak} -> {new_fak}")
            return new_fak, True
        else:
            return self.fak, False

    def createJacobian(self, model):
        """
        Create Jacobian matrix using finite difference approximation.
        
        Computes sensitivity of geophysical response to interface point movements
        by perturbing each parameter and calculating response differences.
        
        Parameters:
        -----------
        model : array-like
            Interface point positions
        """
        pg.tic("Jacobian calculation start")
        
        # Initialize Jacobian matrix
        self.jac = pg.Matrix(rows=len(self.dta), cols=len(model))
        self.setJacobian(self.jac)
        
        # Set finite difference step factor
        if self.irun == 0:
            self.fak = 1.02
        elif self.rrun > 4:
            print('ERROR: Too many model rejections - regularization may be too low')
            exit()
        elif self.irun >= 1 and self.rrun > 0:
            self.fak = 1.01
            
        print(f'Finite difference factor: {self.fak}')
        
        # Prepare for finite difference calculation
        shifts = np.copy(model)
        print(f'Interface point shifts: {shifts}')
        
        sP = self.shiftIP(model)
        if self.dirtype == 'XZ':
            model = self.joinVector(sP[0], sP[1])
        else:
            model = sP
        
        # Initialize arrays for finite differences
        d_m = np.zeros((len(self.dta), len(model)))
        d_r = np.zeros(np.shape(d_m))
        fak_list = np.zeros(len(model))
        
        # Calculate reference response
        self.r0 = self.response(model, jacobianCalc=True)
        
        # Calculate finite differences for each parameter
        for n, p in enumerate(model):
            self.jrun += 1
            self.fak = 1.02
            print(f'Processing parameter {n}: {model[n]}')
            
            # Perturb parameter
            model[n] *= self.fak
            print(f'Shifted model: {np.round(model, 3)}')
            
            # Check if points are too close and adjust if needed
            new_fak, close = self.check_and_adjust_points(model, min_distance_threshold=0.1)
            if close:
                print(f'Points too close for parameter {n}, adjusting step size')
                model[n] /= self.fak  # Remove old perturbation
                self.fak = new_fak    # Use new step size
                model[n] *= self.fak  # Apply new perturbation
            
            # Calculate perturbed response
            r1 = self.response(model, jacobianCalc=True)
            
            # Reset parameter
            model[n] /= self.fak
            print(f'Reset parameter {n} to: {model[n]}')
            
            # Store differences
            d_r[:, n] = r1 - self.r0      
            print(f'Mean response difference: {np.mean(abs(d_r[:,n]))}')
            fak_list[n] = self.fak
        
        print(f'Step factors used: {fak_list}')
        
        # Calculate parameter differences
        for j in range(len(self.dta)):
            for i in range(len(model)):
                d_m[j, i] = (model[i] * fak_list[i]) - model[i]
        
        # Calculate Jacobian
        print(f'Response difference sums: {np.sum(abs(d_r), axis=0)}')
        print(f'Parameter difference sums: {np.sum(abs(d_m), axis=0)}')
        
        self.J_surface = d_r / d_m
        self.responses[self.irun] = self.r0
        
        print(f'Jacobian sensitivity sums: {np.sum(abs(self.J_surface), axis=0)}')
        
        # Set Jacobian in framework
        for nrow, entry in enumerate(self.J_surface):
            self.jac.setRow(nrow, entry)
        
        # Save Jacobian if requested
        if self.saveJacobian:
            pg.boxprint('Jacobian matrix created and saved - inversion ready')
            np.save(self.path + f"/Jacobian_{self.irun}", self.jac)
        else:
            pg.boxprint('Jacobian matrix created - inversion ready')
        
        self.setJacobian(self.jac)
        self.jacobians[self.irun] = self.jac
        
        # Reset counters
        self.jrun = 0
        self.irun += 1
        self.rrun = 0
        
        pg.toc("Jacobian calculation complete")
