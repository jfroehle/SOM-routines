# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:45:55 2024

@author: J. Fröhle

Last update: 07 September 2024 by J. Fröhle
"""
#%% Import modules.
from warnings import warn, formatwarning
try:
    from abc_analysis import abc_analysis
    use_abcanalysis = True
except:
    warn(
        "\nCould not import abc_analysis. This will impact the calculation of "
        "the P-matrix."
    )
    use_abcanalysis = False
try:
    from dask.distributed import Client
    set_cluster_to_None = False
except:
    warn(
        "\nCould not import distributed.client.Client. Computations will be "
        "performed without chunking."
    )
    set_cluster_to_None = True
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
import xarray as xr

#%% Customise warning formatter.
def _formatwarning(message, category, filename, lineno, line = None):
    return formatwarning(
        message = message,
        category = category,
        filename = filename,
        lineno = lineno,
        line = "" # line
    )
formatwarning = _formatwarning

#%% Class for data normalisation.
class Scaler:
    """
    Scale a dataset according to its variance or minimum and maximum along a
    given dimension. For the former method, the dataset is normalised, such
    that the variance is 1 along the given dimension. For the latter method,
    the dataset is scaled to the value range [0, 1]. Both operations are
    linear. If the dataset is normalised, the normalisation can be reverted.

    Attributes
    ----------
    norm_dim : str
        The dimension along which to normalise the data.
    norm_method : str
        The normalisation method used, either 'variance' or 'range'.
    norm_params : list
        The parameters used to normalise the data.

    Methods
    -------
    normalise():
        Normalise the data.
    denormalise():
        Revert the normalisation. Can only be called if the data is already
        normalised.
    """
    #=========================================================================#
    def __init__(
        self,
        f_dimension = None,
        f_method = "variance"
    ):
        """
        Initialise the scaler instance.

        Parameters
        ----------
        f_dimension : str or None
            The dimension along which to normalise the data. If None, applies
            the normalisation over all dimensions.
        f_method : str or xarray.DataArray
            If str, can be one of 'variance' or 'range'. If 'variance', the
            data will be variance-normed, i.e. its variance will be 1 along the
            given dimension. If 'range', the data will be scaled to the value
            range [0, 1] along the given dimension.
            If xarray.DataArray, must be a normalised dataset including the
            attributes 'norm_dim', 'norm_method' and 'norm_params'.
        """
        #---------------------------------------------------------------------#
        # Check for correct inputs.
        if type(f_method) is str and f_method not in ("variance", "range"):
            raise ValueError(
                f"Unrecognised method: {f_method}. f_method should be one of "
                "('variance', 'range')."
            )
        #---------------------------------------------------------------------#
        # Set attributes.
        if type(f_method) is str:
            self.norm_dim = f_dimension
            self.norm_method = f_method
            self.norm_params = None
        else:
            self.norm_dim = f_method.norm_dim
            self.norm_method = f_method.norm_method
            self.norm_params = f_method.norm_params
    #=========================================================================#
    def normalise(
        self,
        f_data = None
    ):
        """
        Normalise a dataset. If no parameters are provided during the
        initialisation of the scaler, the parameters are determined for the
        given dataset according to the chosen method and along the chosen
        dimension.
        Once a scaler instance has been initialised, it will use the parameters
        given or determined on the first call of normalise() on any other call
        of normalise(). To change the parameters, the scaler instance needs to
        be re-initialised.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray containing the data to normalise.

        Returns
        -------
        f_data : xarray.DataArray
            The normalised dataset.
        """
        #---------------------------------------------------------------------#
        # Determine normalisation parameters.
        if self.norm_params is None:
            self._norm_scale_init(f_data = f_data)
        #---------------------------------------------------------------------#
        # Normalise data.
        f_data = self._norm_scale_do(f_data = f_data)
        #---------------------------------------------------------------------#
        # Save parameters as attributes in dataset.
        f_data = f_data.assign_attrs({
            "norm_dim" : self.norm_dim,
            "norm_method" : self.norm_method,
            "norm_params" : self.norm_params
        })
        #---------------------------------------------------------------------#
        return f_data
    #=========================================================================#
    def denormalise(
        self,
        f_data = None
    ):
        """
        Revert the normalisation applied to a dataset. The respective
        parameters must be stored as attributes in the dataset.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray containing the normalised dataset.

        Returns
        -------
        f_data : xarray.DataArray
            The de-normalised dataset.
        """
        #---------------------------------------------------------------------#
        # Check if input data is normalised.
        if "norm_method" not in f_data.attrs:
            raise NotImplementedError(
                "Data must be normalised before calling 'denormalise'."
            )
        #---------------------------------------------------------------------#
        # Revert normalisation.
        f_data = self._norm_scale_undo(f_data = f_data)
        #---------------------------------------------------------------------#
        return f_data
    #=========================================================================#
    def _norm_scale_init(
        self,
        f_data = None
    ):
        """
        Determine the parameters according to which the data will be scaled.
        For all-NaN slices the mean / minimum will be set to 0 and the standard
        deviation / maximum will be set to 1.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray from which to estimate the parameters.
        """
        #---------------------------------------------------------------------#
        # Set all-NaN slices to 0.
        f_data = f_data.where(
            np.isfinite(f_data).any(dim = self.norm_dim),
            0.
        )
        #---------------------------------------------------------------------#
        # Determine normalisation parameters.
        if self.norm_method == "variance":
            self.norm_params = [
                f_data.mean(dim = self.norm_dim),
                f_data.std(dim = self.norm_dim)
            ]
            self.norm_params[1] = self.norm_params[1].where(
                self.norm_params[1] != 0,
                1
            )
        else:
            self.norm_params = [
                f_data.min(dim = self.norm_dim),
                f_data.max(dim = self.norm_dim)
            ]
            self.norm_params[1] = xr.where(
                self.norm_params[1] == self.norm_params[0],
                1,
                self.norm_params[1] - self.norm_params[0]
            )
        #---------------------------------------------------------------------#
        # Convert parameters to float if 1-dimensional.
        self.norm_params = [
            float(param) if param.size == 1
            else param
            for param in self.norm_params
        ]
    #=========================================================================#
    def _norm_scale_do(
        self,
        f_data = None
    ):
        """
        Normalise a dataset.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray containing the data to normalise.

        Returns
        -------
        f_data : xarray.DataArray
            The normalised dataset.
        """
        #---------------------------------------------------------------------#
        # Normalise data.
        f_data = (f_data - self.norm_params[0]) / self.norm_params[1]
        #---------------------------------------------------------------------#
        return f_data
    #=========================================================================#
    def _norm_scale_undo(
        self,
        f_data = None
    ):
        """
        Revert the normalisation applied to a dataset. The respective
        parameters must be stored as attributes in the dataset.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray containing the normalised dataset.

        Returns
        -------
        f_data : xarray.DataArray
            The de-normalised dataset.
        """
        #---------------------------------------------------------------------#
        # Revert normalisation.
        f_data = f_data * f_data.norm_params[1] + f_data.norm_params[0]
        #---------------------------------------------------------------------#
        return f_data

#%% Class for custom warning.
class DropFeaturesWarning(Warning):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return str(self.message)

#%% Class for self-organising maps.
class Self_Organising_Map:
    """
    Implementation of a self-organising map (SOM) machine learning algorithm
    utilising a batch algorithm. For the batch algorithm, the whole training
    dataset is gone through at once and only after this the SOM is updated
    with the net effect of all samples. Here, the SOM is updated by replacing
    the map with a weighted average over the samples, where the weighting
    factors are the neighbourhood function values. Different options are
    available for the shape of the SOM lattice. The lattice can be either
    rectangular or hexagonal. The type of lattice defines the neighbourhood of
    the different grid cells. For the rectangular grid a grid cell has 8
    direct neighbours, while for the hexagonal grid there are only 6 direct
    neighbours. The lattice can either be planar or toroidal. For the planar
    hexagonal lattice different shapes can be set. The lattice can be shaped as
    a rectangle, a rhombus or a regular hexagon. If the lattice is not shaped
    as a hexagon and the number of rows and columns is not specified, the
    number of rows and columns will be determined based on the ratio between
    the two largest eigenvalues of training dataset.

    Attributes
    ----------
    m : int
        The number of rows in the SOM.
    n : int
        The number of columns in the SOM.
    n_classes : int
        The size of the SOM, i.e. the number of SOM classes.
    n_features : int
        The number of features in the training dataset.
    n_samples : int or str
        The number of samples used to train the SOM.
    lattice : str
        The type of grid used for the SOM, either 'rect' or 'hexa'.
    shape : str
        The shape of the lattice; either 'rectangle' or 'toroid' for both types
        of lattice. If lattice is 'hexa', the shape can also be 'rhombus' or
        'hexagon'.
    som : xarray.Dataset
        An xarray.Dataset containing the weights of the SOM. If labels are
        provided alongside the features, the labels corresponding to the
        weights are also included.
    epochs : int
        The number of epochs for which the SOM has been trained.
    sigma1 : float
        The maximum neighbourhood radius during training.
    sigma0 : float
        The minimum neighbourhood radius during training.
    beta : float
        The decay rate of the neighbourhood radius during training.
    metric : str
        The metric to estimate the accuracy of the SOM if labels are provided
        alongside the features. Can be one of "mean_absolute_error",
        "root_mean_square_error", "median_absolute_deviation" or "accuracy".
    u_matrix : xarray.DataArray
        The U-matrix corresponding to the trained SOM.
    p_matrix : xarray.DataArray
        The P-matrix corresponding to the trained SOM.
    ustar_matrix : xarray.DataArray
        The U*-matrix corresponding to the trained SOM.
    attrs : dict
        A dictionary summarising all parameters that have been set.

    Methods
    -------
    init_from_file():
        Initialise the Self_Organising_Map instance using a trained SOM stored
        to a NetCDF-file, i.e. Self_Organising_Map() is replaced by
        Self_Organising_Map.init_from_file().
    compile():
        Initialise the SOM grid using random values. The values are scaled to
        match the input data's standard deviation and mean before training.
    train():
        Train the SOM for a given number of epochs. Repeated call of this
        method does not reset the SOM grid and the epochs. This is done in
        the compile method.
    predict()
        Predict the BMU for given data using the trained SOM.
    evaluate()
        Estimate the accuracy of the trained SOM if labels are available along-
        side the given features.
    Davies_Bouldin_Index()
        Estimate the Davies-Bouldin index for the trained SOM.
    compute_umatrix()
        Compute the U-matrix corresponding to the trained SOM.
    compute_pmatrix()
        Compute the P-matrix corresponding to the trained SOM.
    compute_ustarmatrix()
        Compute the U*-matrix corresponding to the trained SOM.
    SOM_to_netcdf()
        Save the trained SOM to a NetCDF-file, including all relevant
        parameters.
    visualise_SOM()
        Plot the weights of the SOM on the chosen lattice.
    """
    #=========================================================================#
    def __init__(
        self,
        f_chunk_size = "auto",
        f_dask_cluster = None,
        f_random_state = 12345,
        f_verbose = 1
    ):
        """
        Initialise the random number generator and estimate the chunk size in
        bytes, if auto-chunking is used. The dask cluster is used only if the
        size of the data used during computation exceeds the chunk size limit.
        For auto-chunking the limit is set to 1/20 of the memory limit per
        worker.

        Parameters
        ----------
        f_chunk_size : int, float or str
            If "auto", the chunk size is computed based on the memory limit of
            the dask cluster, i.e. the chunks size is set such that each chunk
            is approximately 1/20 of the worker memory limit. If an int,
            specifies the chunks size along the sample dimension of the data.
            If a float, determines the chunk size as the fraction of the worker
            memory limit.
        f_dask_cluster : distributed.deploy.local.LocalCluster or None
            A dask cluster to use for computation for large datasets. If None
            and the dataset is to big, will throw a memory error.
        f_random_state : int
            An integer defining the seed for the random number generator.
        f_verbose : int
            An integer defining whether output will be displayed in the
            console. If 0 no output will be shown, if 1 a progress bar and the
            number of cells (if it has to be calculated) will be displayed.
        """
        #---------------------------------------------------------------------#
        # Set attributes.
        self._verbose = f_verbose
        self._trained = False
        self._normed = False
        self.m = None
        self.n = None
        self.n_classes = None
        self._feature_names = None
        self.n_features = None
        self.n_samples = None
        self._training_data = None
        self.lattice = None
        self.shape = None
        self.som = None
        self.epochs = []
        self.sigma1 = []
        self.sigma0 = []
        self.beta = []
        self.u_matrix = None
        self.p_matrix = None
        self.ustar_matrix = None
        self._attributes()
        #---------------------------------------------------------------------#
        # Initialise random number genearator.
        self.rng = np.random.default_rng(
            np.random.SeedSequence(f_random_state)
        )
        #---------------------------------------------------------------------#
        # Initialise local dask cluster.
        if set_cluster_to_None is True or f_dask_cluster is None:
            self._cluster = None
            self._chunk_bytes = None
            self._chunk_size = None
        else:
            if type(f_chunk_size) in (str, int):
                frac = 1. / 20.
            elif type(f_chunk_size) is float:
                frac = f_chunk_size
            self._cluster = f_dask_cluster
            self._chunk_bytes = self._cluster._memory_per_worker() * frac
            self._chunk_size = f_chunk_size
    #=========================================================================#
    def init_from_file(
        f_init_file = None,
        f_chunk_size = "auto",
        f_dask_cluster = None,
        f_random_state = 12345,
        f_verbose = 1
    ):
        """
        Initialise a Self_Organising_Map class from an input file containing a
        trained SOM. All relevant information needs to be stored within the
        dataset; a respective file can be generated with the SOM_to_netcdf()
        method.
        The methods Davies_Bouldin_Index(), compute_umatrix(),
        compute_pmatrix(), and compute_ustarmatrix() can only be used if the
        respective training data is provided separately. Sensible results will
        only be achieved if the data corresponds exactly to the dataset that
        was used to train the SOM.

        Parameters
        ----------
        f_init_file : pathlib.PosixPath
            The absolute path pointing to the file containing the trained SOM.
        f_chunk_size : int, float or str
            If "auto", the chunk size is computed based on the memory limit of
            the dask cluster, i.e. the chunks size is set such that each chunk
            is approximately 1/20 of the worker memory limit. If an int,
            specifies the chunks size along the sample dimension of the data.
            If a float, determines the chunk size as the fraction of the worker
            memory limit.
        f_dask_cluster : distributed.deploy.local.LocalCluster or None
            A dask cluster to use for computation for large datasets. If None
            and the dataset is to big, will throw a memory error.
        f_random_state : int
            An integer defining the seed for the random number generator.
        f_verbose : int
            An integer defining whether output will be displayed in the
            console. If 0 no output will be shown, if 1 a progress bar and the
            number of cells (if it has to be calculated) will be displayed.

        Returns
        -------
        init_som_class : Self_Organising_Map
            A Self_Organising_Map instance initialised from a given file for
            further usage.
        """
        #---------------------------------------------------------------------#
        # Initialise Self-Organising Map class.
        init_som_class = Self_Organising_Map(
            f_chunk_size = f_chunk_size,
            f_dask_cluster = f_dask_cluster,
            f_random_state = f_random_state,
            f_verbose = f_verbose
        )
        #---------------------------------------------------------------------#
        # Load trained SOM.
        input_som_data = xr.open_dataset(f_init_file).load()
        #---------------------------------------------------------------------#
        # Set parameters.
        init_som_class._trained = True
        init_som_class.lattice = input_som_data.lattice
        init_som_class.shape = input_som_data.shape
        init_som_class.m = int(input_som_data.m)
        init_som_class.n = int(input_som_data.n)
        init_som_class.n_classes = input_som_data.grid_cell.size
        init_som_class._indices = input_som_data.grid_cell.values
        if "grid_mask" in input_som_data:
            init_som_class._grid_mask = input_som_data.grid_mask.values
        init_som_class.n_features = input_som_data.feature.size
        init_som_class._feature_names = input_som_data.feature.values
        try:
            init_som_class.epochs = [int(input_som_data.epochs)]
            init_som_class.sigma1 = [float(input_som_data.sigma1)]
            init_som_class.sigma0 = [float(input_som_data.sigma0)]
            init_som_class.beta = [float(input_som_data.beta)]
        except:
            init_som_class.epochs = list(map(int, input_som_data.epochs))
            init_som_class.sigma1 = list(map(float, input_som_data.sigma1))
            init_som_class.sigma0 = list(map(float, input_som_data.sigma0))
            init_som_class.beta = list(map(float, input_som_data.beta))
        if "norm_dim" in input_som_data.weights.attrs.keys():
            init_som_class._normed = True
            input_som_data["weights"] = input_som_data.weights.assign_attrs({
                "norm_params" : [
                    input_som_data.norm_params_0,
                    input_som_data.norm_params_1
                ]
            })
            input_som_data = input_som_data.drop_vars([
                "norm_params_0", "norm_params_1"
            ])
            init_som_class._norm_dim = input_som_data.weights.norm_dim
            init_som_class._norm_method = input_som_data.weights.norm_method
            init_som_class._norm_params = input_som_data.weights.norm_params
        input_som_data = input_som_data.assign_coords({
            "grid_cell" : input_som_data.gc_row + 1j * input_som_data.gc_col
        }).drop_vars([
            "gc_row", "gc_col"
        ])
        init_som_class.som = input_som_data.copy(deep = True)
        init_som_class._attributes()
        #---------------------------------------------------------------------#
        # Add offsets for toroidal maps.
        if init_som_class.shape == "toroid":
            init_som_class._offsets = np.array([
                [0, 0],
                [init_som_class.n, 0],
                [-init_som_class.n, 0],
                [0, init_som_class.m],
                [0, -init_som_class.m],
                [init_som_class.n, init_som_class.m],
                [init_som_class.n, -init_som_class.m],
                [-init_som_class.n, init_som_class.m],
                [-init_som_class.n, -init_som_class.m]
            ])
            init_som_class._offsets[:, 0] = (
                init_som_class._offsets[:, 0]
                - (
                    init_som_class._offsets[:, 1]
                    - (init_som_class._offsets[:, 1] & 1)
                ) / 2.
            )
            init_som_class._offsets = xr.DataArray(
                data = (
                    init_som_class._offsets[:, 1]
                    + 1j * init_som_class._offsets[:, 0]
                ),
                dims = "offsets"
            )
        #---------------------------------------------------------------------#
        return init_som_class
    #=========================================================================#
    def compile(
        self,
        f_features = None,
        f_SOM_cols = 10,
        f_SOM_rows = 10,
        f_SOM_size = None,
        f_lattice = "hexa",
        f_shape = "rectangle"
    ):
        """
        Set up the SOM grid depending on the type of lattice and its shape. The
        lattice can either be rectangular or hexagonal. The coordinates of the
        individual grid cells are stored as complex numbers in self._indices.
        The number of epochs is reset to 0.
        If the number of rows and columns is not specified, the SOM grid will
        be initialised during the first call to the train() method. The number
        of rows and columns is then determined based on the ratio of the two
        largest eigenvalues of the training dataset.

        Parameters
        ----------
        f_features : list
            A list containing the names of the features included in the
            training dataset.
        f_SOM_cols : int
            The number of columns in the SOM. Must be set to None if f_SOM_size
            is provided.
        f_SOM_rows : int
            The number of rows in the SOM. Must be set to None if f_SOM_size is
            is provided.
        f_SOM_size : int
            The approximate number of grid cells in the SOM. Must be provided
            instead of f_SOM_cols and f_SOM_rows if f_lattice is 'hexa' and
            f_shape is 'hexagon'. Must be set to None if f_SOM_cols and
            f_SOM_rows are provided.
        f_lattice : str
            Either 'rect' or 'hexa'. Defines the neighbourhood of the SOM's
            grid cells. For 'rect' there are 8 direct neighbours, while for
            'hexa' there are only 6 direct neighbours.
        f_shape : str
            Defines the shape of the lattice. Either 'rectangle' or 'toroid'
            for both types of lattice. If lattice is 'hexa', the shape can also
            be 'rhombus' or 'hexagon'.
        """
        #---------------------------------------------------------------------#
        # Check for correct inputs.
        if f_lattice not in ("rect", "hexa"):
            raise ValueError(
                "f_lattice should be one of 'rect' or 'hexa', "
                f"but is '{f_lattice}'."
            )
        if (
            f_lattice == "hexa"
            and f_shape not in ("toroid", "rectangle", "rhombus", "hexagon")
        ):
            raise ValueError(
                "f_shape should be one of 'toroid', 'rectangle', 'rhombus', "
                f"or 'hexagon' but is '{f_shape}'."
            )
        elif (
            f_lattice == "rect"
            and f_shape not in ("toroid", "rectangle")
        ):
            raise ValueError(
                "f_shape should be one of 'toroid' or 'rectangle' but is "
                f"'{f_shape}'."
            )
        if (
            f_lattice == "hexa" and f_shape == "toroid"
            and f_SOM_rows is not None and f_SOM_rows % 2 != 0
        ):
            raise ValueError(
                "f_SOM_rows must be even if f_lattice is 'hexa' and f_shape "
                "is 'toroid'."
            )
        if (
            f_lattice == "hexa" and f_shape == "hexagon"
            and f_SOM_size is None
        ):
            raise ValueError(
                "f_SOM_size must be provided if f_lattice is 'hexa' and "
                "f_shape is 'hexagon'."
            )
        if (
            f_SOM_size is not None
            and (f_SOM_cols is not None or f_SOM_rows is not None)
        ):
            raise ValueError(
                "f_SOM_size and f_SOM_cols / f_SOM_rows cannot be set at the "
                "same time. Either set f_SOM_size or f_SOM_cols / f_SOM_rows "
                "to None."
            )
        if f_SOM_rows is not None and f_SOM_rows < 1:
            raise ValueError(
                f"f_SOM_rows should be > 1, but is {f_SOM_rows}."
            )
        if f_SOM_cols is not None and f_SOM_cols < 1:
            raise ValueError(
                f"f_SOM_cols should be > 1, but is {f_SOM_cols}."
            )
        if f_SOM_size is not None and f_SOM_size < 1:
            raise ValueError(
                f"f_SOM_size should be > 1, but is {f_SOM_size}."
            )
        #---------------------------------------------------------------------#
        # Set SOM grid parameters.
        self._trained = False
        self._normed = False
        self.n_samples = None
        self._feature_names = f_features
        self.n_features = len(self._feature_names)
        self.m = f_SOM_rows
        self.n = f_SOM_cols
        self.n_classes = self.m * self.n if f_SOM_size is None else f_SOM_size
        self.lattice = f_lattice
        self.shape = (
            f_shape if self.lattice == "hexa" or f_shape == "toroid"
            else "rectangle"
        )
        self.epochs = []
        self.sigma1 = []
        self.sigma0 = []
        self.beta = []
        self.u_matrix = None
        self.p_matrix = None
        self.ustar_matrix = None
        self._attributes()
        #---------------------------------------------------------------------#
        # Define grid coordinates.
        if (
            not (self.lattice == "hexa" and self.shape == "hexagon")
            and self.n is None
        ):
            return
        elif (
            self.lattice == "rect"
            or (self.lattice == "hexa" and self.shape == "rhombus")
        ):
            self._cols, self._rows = np.meshgrid(
                np.arange(self.n, dtype = "float32"),
                np.arange(self.m, dtype = "float32")
            )
        elif self.lattice == "hexa" and self.shape == "rectangle":
            self._cols, self._rows = np.meshgrid(
                np.arange(self.n, dtype = "float32"),
                np.arange(self.m, dtype = "float32")
            )
            self._cols = (
                self._cols
                - (self._rows - (self._rows.astype("int32") & 1)) / 2.
            )
        elif self.lattice == "hexa" and self.shape == "toroid":
            self._cols, self._rows = np.meshgrid(
                np.arange(self.n, dtype = "float32"),
                np.arange(self.m, dtype = "float32")
            )
            self._cols = (
                self._cols
                - (self._rows - (self._rows.astype("int32") & 1)) / 2.
            )
            self._offsets = np.array([
                [0, 0],
                [self.n, 0],
                [-self.n, 0],
                [0, self.m],
                [0, -self.m],
                [self.n, self.m],
                [self.n, -self.m],
                [-self.n, self.m],
                [-self.n, -self.m]
            ])
            self._offsets[:, 0] = (
                self._offsets[:, 0]
                - (self._offsets[:, 1] - (self._offsets[:, 1] & 1)) / 2.
            )
            self._offsets = xr.DataArray(
                data = self._offsets[:, 1] + 1j * self._offsets[:, 0],
                dims = "offsets"
            )
        else:
            num_rings = int(np.round(
                -0.5 + np.sqrt(0.25 + (self.n_classes - 1.) / 3.)
            ))
            self.m = 2 * num_rings + 1
            self.n = 2 * num_rings + 1
            pad = int(np.ceil(self.m / 2. - 1.))
            self._cols, self._rows = np.meshgrid(
                np.arange(self.n, dtype = "float32"),
                np.arange(self.m, dtype = "float32")
            )
            self._grid_mask = np.array([
                (
                    [1] * max(pad - i, 0)
                    + [0] * (self.n - abs(pad - i))
                    + [1] * abs(min(pad - i, 0))
                )
                for i in range(self.m)
            ]).astype("bool")
            self._rows[self._grid_mask] = np.nan
            self._cols[self._grid_mask] = np.nan
            self.n_classes = (~self._grid_mask).sum()
            if self._verbose == 1:
                print(f"SOM size is set to {self.n_classes} classes.")
        self._indices = (self._rows + 1j * self._cols).flatten()
        self._indices = self._indices[np.isfinite(self._indices)]
        #---------------------------------------------------------------------#
        # Initialise SOM grid.
        self.som = xr.Dataset(
            data_vars = {
                "weights" : (
                    ("grid_cell", "feature"),
                    self.rng.normal(
                        loc = 0.,
                        scale = 1.,
                        size = (self.n_classes, self.n_features)
                    )
                ),
            },
            coords = {
                "grid_cell" : (("grid_cell", ), self._indices),
                "feature" : (("feature", ), self._feature_names)
            },
            attrs = {
                "n" : self.n,
                "m" : self.m,
                "lattice" : self.lattice,
                "shape" : self.shape
            }
        )
        #---------------------------------------------------------------------#
        # Update attributes.
        self._attributes()
    #=========================================================================#
    def train(
        self,
        f_data_features = None,
        f_data_labels = None,
        f_epochs = 10,
        f_neighbourhood_radius = {"s1" : 1., "s0" : None, "b" : 0.1}
    ):
        """
        Train the SOM for a given number of epochs. If labels are available
        alongside the features, labels are assigned to the trained SOM.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The training dataset. An xarray.DataArray of shape (k, l)
            containing k samples of the l features. The dimensions must be
            named 'sample' and 'feature'.
        f_data_labels : xarray.DataArray or None
            An xarray.DataArray of shape (k, ) containing the labels
            corresponding to f_data_features. The dimension must be named
            'sample'.
        f_epochs : int
            The number of epochs for which to train the SOM.
        f_neighbourhood_radius : dict
            A dictionary containing the maximum neighbourhood radius ("s1"), as
            well as either the minimum neighbourhood radius ("s0") or the decay
            rate ("b"). The respective other parameter has to be set to None.
            The neighbourhood radius follows an exponential decy during
            training.

        Returns
        -------
        self.som : xarray.Dataset
            An xarray.Dataset containing the weights of the SOM after the
            training process. If f_data_labels is given, the labels
            corresponding to the trained SOM grid are included as well.
        """
        #---------------------------------------------------------------------#
        # Check for correct inputs.
        f_data_features = f_data_features.transpose("sample", "feature", ...)
        if (
            np.setdiff1d(
                self._feature_names,
                f_data_features.feature.values
            ).size != 0
        ):
            raise ValueError(
                "f_data_features has features "
                f"{f_data_features.feature.values}, but the "
                "Self_Organising_Map object has been compiled with "
                f"{self._feature_names}."
            )
        elif (
            np.setdiff1d(
                self._feature_names,
                f_data_features.feature.values
            ).size == 0
            and self.n_features < f_data_features.feature.size
        ):
            drop_features = list(np.setdiff1d(
                f_data_features.feature.values,
                self._feature_names
            ))
            f_data_features = f_data_features.sel(
                feature = self._feature_names
            )
            if self._verbose == 1:
                warn(
                    "\nThe following features will be removed from "
                    "f_data_features, as the Self_Organising_Map object has "
                    f"been compiled without them:\n{drop_features}\n",
                    category = DropFeaturesWarning,
                    stacklevel = 2
                )
        if (
            f_data_labels is not None
            and len(f_data_labels.shape) > 1
        ):
            raise ValueError(
                "f_data_labels is expected to be 1-dimensional but has "
                f"{len(f_data_labels.shape)}."
            )
        if (
            f_data_labels is not None
            and f_data_labels.sample.size != f_data_features.sample.size
        ):
            raise ValueError(
                "f_data_features and f_data_labels must have identical number "
                f"of samples but have {f_data_features.sample.size} and "
                f"{f_data_labels.sample.size}."
            )
        if (
            self._trained is True
            and "labels" in self.som
            and f_data_labels is None
        ):
            raise ValueError(
                "f_data_labels is None, but SOM has been trained with labels. "
                "To continue training the SOM provide f_data_labels, else "
                "re-compile the SOM."
            )
        if f_epochs < 1:
            raise ValueError(f"f_epochs should be > 1, but is {f_epochs}.")
        if all([it is not None for it in f_neighbourhood_radius.values()]):
            raise ValueError(
                "Only two parameters can be provided in "
                "f_neighbourhood_radius. Either 's0' or 'b' must be set to "
                "None."
            )
        if f_neighbourhood_radius["s1"] is None:
            raise ValueError("f_neighbourhood_radius['s1'] cannot be None.")
        if (
            (self._trained is True and self._normed is False)
            and "norm_method" in f_data_features.attrs
        ):
            raise ValueError(
                "f_data_features is normalised, but the SOM has been trained "
                "on un-normalised data. To continue training the SOM use "
                "un-normalised data instead, else re-compile the SOM."
            )
        if (
            (self._trained is True and self._normed is True)
            and "norm_method" not in f_data_features.attrs
        ):
            raise ValueError(
                "f_data_features is not normalised, but the SOM has been "
                "trained on normalised data. To continue training the SOM use "
                "normalised data instead, else re-compile the SOM."
            )
        if (
            (self._trained is True and self._normed is True)
            and (
                not isinstance(
                    self._norm_params[0],
                    type(f_data_features.norm_params[0])
                )
                or not np.allclose(
                    self._norm_params,
                    f_data_features.norm_params
                )
            )
        ):
            raise ValueError(
                "f_data_features has been normalised using different "
                "parameters than the dataset used to train the SOM. To "
                "continue training the SOM use data that is normalised using "
                "the same parameters, else re-compile the SOM."
            )
        #---------------------------------------------------------------------#
        # Assign coordinate to sample dimension if not available.
        if "sample" not in f_data_features.coords:
            f_data_features = f_data_features.assign_coords({
                "sample" : np.arange(f_data_features.sample.size)
            })
        if f_data_labels is not None and "sample" not in f_data_labels.coords:
            f_data_labels = f_data_labels.assign({
                "sample" : f_data_features.sample
            })
        #---------------------------------------------------------------------#
        # Check if all-NaN features are present in the data.
        f_data_features = f_data_features.dropna(
            dim = "feature",
            how = "all"
        )
        drop_features = list(np.setdiff1d(
            self._feature_names,
            f_data_features.feature.values
        ))
        if len(drop_features) > 0:
            self._feature_names = [
                feat for feat in self._feature_names
                if feat not in drop_features
            ]
            self.n_features = len(self._feature_names)
            if self._verbose == 1:
                warn(
                    "\nThe following features will be removed from"
                    "f_data_features, as they do not contain any valid values:"
                    f"\n{drop_features}\n",
                    category = DropFeaturesWarning,
                    stacklevel = 2
                )
        #---------------------------------------------------------------------#
        # Initialise SOM grid if dependent on training data.
        self.n_samples = f_data_features.sample.size
        if self.n is None:
            autocorr = self._compute_auto_correlation(f_data = f_data_features)
            eigenvalues = np.sort(np.linalg.eig(autocorr).eigenvalues).real
            sidelength_ratio = np.sqrt(eigenvalues[-1] / eigenvalues[-2])
            if self.lattice == "hexa":
                self.m = int(np.round(np.sqrt(
                    self.n_classes / sidelength_ratio * np.sqrt(0.75)
                )))
                if self.shape == "toroid" and self.m % 2 != 0:
                    self.m += 1
            else:
                self.m = int(np.round(np.sqrt(
                    self.n_classes / sidelength_ratio
                )))
            self.n = int(np.round(self.n_classes / self.m))
            self.compile(
                f_features = self._feature_names,
                f_SOM_cols = self.n,
                f_SOM_rows = self.m,
                f_SOM_size = None,
                f_lattice = self.lattice,
                f_shape = self.shape
            )
            if self._verbose == 1:
                print(f"SOM grid is set to {self.m} x {self.n} grid cells.")
        else:
            self.som = self.som.sel(feature = self._feature_names)
        #---------------------------------------------------------------------#
        # Scale SOM by input data standard deviation and mean after
        # initialisation with random values.
        if self._trained is False:
            self.som = self.som.assign({
                "weights" : (
                    self.som.weights
                    * f_data_features.std(dim = "sample")
                    + f_data_features.mean(dim = "sample")
                )
            })
        #---------------------------------------------------------------------#
        # Define radii for each epoch.
        self.epochs.append(int(f_epochs))
        self.sigma1.append(float(f_neighbourhood_radius["s1"]))
        if f_neighbourhood_radius["s0"] is None:
            self.beta.append(float(f_neighbourhood_radius["b"]))
            neighbourhood_radii = self.sigma1[-1] * np.exp(
                -np.arange(0., self.epochs[-1], 1.)
                * self.beta[-1]
            )
            self.sigma0.append(float(neighbourhood_radii.min()))
        else:
            self.sigma0.append(float(f_neighbourhood_radius["s0"]))
            self.beta.append(float(
                -np.log(self.sigma0[-1] / self.sigma1[-1])
                / self.epochs[-1]
            ))
            neighbourhood_radii = self.sigma1[-1] * np.exp(
                -np.arange(0., self.epochs[-1], 1.)
                * self.beta[-1]
            )
        #---------------------------------------------------------------------#
        # Update weights of the SOM grid.
        for nr in tqdm(
            neighbourhood_radii,
            ncols = 85,
            desc = "Update SOM classes",
            disable = False if self._verbose == 1 else True,
            position = 0,
            leave = True
        ):
            self._update_weights(
                f_data = f_data_features,
                f_radius = nr
            )
        #---------------------------------------------------------------------#
        # Assign labels to SOM grid.
        if f_data_labels is not None:
            self._assign_labels(
                f_data_features = f_data_features,
                f_data_labels = f_data_labels
            )
        #---------------------------------------------------------------------#
        # Add grid mask for hexagon-shaped lattice.
        if self.lattice == "hexa" and self.shape == "hexagon":
            self.som = self.som.assign({
                "grid_mask" : (("cols", "rows"), self._grid_mask)
            })
        #---------------------------------------------------------------------#
        # Update attributes.
        self._attributes()
        self._trained = True
        self._training_data = f_data_features
        if "norm_method" in f_data_features.attrs.keys():
            self._normed = True
            self._norm_dim = f_data_features.norm_dim
            self._norm_method = f_data_features.norm_method
            self._norm_params = f_data_features.norm_params
            self.som = self.som.assign({
                "weights" : self.som.weights.assign_attrs({
                    "norm_dim" : self._norm_dim,
                    "norm_method" : self._norm_method,
                    "norm_params" : self._norm_params
                })
            })
        #---------------------------------------------------------------------#
        return self.som.copy(deep = True)
    #=========================================================================#
    def predict(
        self,
        f_data_features = None,
        f_mask = None
    ):
        """
        Determine the best matching units for the given dataset. If f_mask is
        provided, the distance calculation will be weighted according to
        f_mask. If f_data_features contains missing values, f_mask can be set
        to "auto". In this case, the distance calculation will be weighted by
        the correlation between the features of the data used to train the SOM.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The dataset for which to estimate the best matching units. An
            xarray.DataArray of shape (j, l) containing j samples of the l
            features. The dimensions must be named 'sample' and 'feature'.
        f_mask : xarray.DataArray, "auto" or None
            An xarray.DataArray of shape (l, ) or (j, l) to be used to weight
            the BMU search. If "auto", determines the correlation between
            missing and available values in the dataset used to train the SOM
            and weights the BMU search based on the correlations. If None, all
            weights are set to None, i.e. equal weights for all inputs.

        Returns
        -------
        best_matching_units : xarray.DataArray
            An xarray.DataArray of shape (j, ) containing the best matching
            units and their coordinates. If labels are available, the labels
            corresponding to best matching units are included as well.
        """
        #---------------------------------------------------------------------#
        # Check if SOM is trained and for correct inputs.
        if not self._trained:
            raise NotImplementedError(
                "Self_Organising_Map object has no predict() method "
                "until after calling train()."
            )
        input_feature_names = f_data_features.feature.values
        if (
            np.setdiff1d(
                self._feature_names,
                input_feature_names
            ).size != 0
        ):
            raise ValueError(
                "f_data_features has features "
                f"{input_feature_names}, but the Self_Organising_Map object "
                f"has been compiled with {self._feature_names}."
            )
        elif (
            np.setdiff1d(
                self._feature_names,
                input_feature_names
            ).size == 0
            and self.n_features < input_feature_names.size
        ):
            drop_features = list(np.setdiff1d(
                input_feature_names,
                self._feature_names
            ))
            f_data_features = f_data_features.sel(
                feature = self._feature_names
            )
            if self._verbose == 1:
                warn(
                    "\nThe following features will be removed from "
                    "f_data_features, as the Self_Organising_Map object has "
                    f"been compiled without them:\n{drop_features}\n",
                    category = DropFeaturesWarning,
                    stacklevel = 2
                )
        if (
            "norm_method" in f_data_features.attrs
            and self._normed is False
        ):
            raise ValueError(
                "f_data_features is normalised, but the SOM has been trained "
                "on un-normalised data. To make predictions use un-normalised "
                "data instead."
            )
        if (
            "norm_method" not in f_data_features.attrs
            and self._normed is True
        ):
            raise ValueError(
                "f_data_features is not normalised, but the SOM has been "
                "trained on normalised data. To make predictions use "
                "normalised data instead."
            )
        if (
            (self._trained is True and self._normed is True)
            and (
                not isinstance(
                    self._norm_params[0],
                    type(f_data_features.norm_params[0])
                )
                or not np.allclose(
                    self._norm_params,
                    f_data_features.norm_params
                )
            )
        ):
            raise ValueError(
                "f_data_features has been normalised using different "
                "parameters than the dataset used to train the SOM. To make "
                "predictions use data that is normalised using the same "
                "parameters, else re-compile the SOM."
            )
        #---------------------------------------------------------------------#
        # Assign coordinate to sample dimension if not available.
        if "sample" not in f_data_features.coords:
            f_data_features = f_data_features.assign_coords({
                "sample" : np.arange(f_data_features.sample.size)
            })
        #---------------------------------------------------------------------#
        # Determine BMU coordinates.
        best_matching_units = self._find_bmu(
            f_data = f_data_features,
            f_mask = f_mask,
            f_num_values = 1
        )
        #---------------------------------------------------------------------#
        # Predict weights and labels if applicable.
        best_matching_units = self.som.sel(
            grid_cell = best_matching_units.dropna(dim = "sample")
        ).reindex(
            sample = best_matching_units.sample,
            feature = input_feature_names
        ).assign_coords(f_data_features.coords)
        #---------------------------------------------------------------------#
        return best_matching_units
    #=========================================================================#
    def evaluate(
        self,
        f_data_features = None,
        f_data_labels = None,
        f_mask = None,
        f_metric = "accuracy"
    ):
        """
        Estimate the accuracy of the trained SOM grid. Only possible if labels
        are provided during the training process.  If f_mask is provided, the
        distance calculation during the BMU search will be weighted according
        to f_mask. If f_data_features contains missing values, f_mask can be
        set to "auto". In this case, the distance calculation will be weighted
        by the correlation between the features of the data used to train the
        SOM.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The testing dataset. An xarray.DataArray of shape (j, l) containing
            j samples of the l features.
        f_data_labels : xarray.DataArray
            An xarray.DataArray of shape (l, ) containing the labels
            corresponding to f_data_features.
        f_mask : xarray.DataArray, "auto" or None
            An xarray.DataArray of shape (l, ) or (j, l) to be used to weight
            the BMU search. If "auto", determines the correlation between
            missing and available values in the dataset used to train the SOM
            and weights the BMU search based on the correlations. If None, all
            weights are set to None, i.e. equal weights for all inputs.
        f_metric : str
            The metric to use for estimating the accuracy. Can be one of
            "root_mean_square_error", "mean_absolute_error",
            "median_absolute_deviation" or "accuracy". If "accuracy", estimates
            the fraction of correctly assigned labels.

        Returns
        -------
        metric_value : float
            The value of the specified metric.
        """
        #---------------------------------------------------------------------#
        # Check if SOM is trained with labels and for correct inputs.
        if "labels" not in self.som:
            raise NotImplementedError(
                "Self_Organising_Map object has no evaluate() method "
                "if train() was called without f_data_labels."
            )
        if len(f_data_labels.shape) > 1:
            raise ValueError(
                "f_data_labels is expected to be 1-dimensional but has "
                f"{len(f_data_labels.shape)}."
            )
        if f_data_labels.sample.size != f_data_features.sample.size:
            raise ValueError(
                "f_data_features and f_data_labels must have identical number "
                f"of samples but have {f_data_features.sample.size} and "
                f"{f_data_labels.sample.size}."
            )
        #---------------------------------------------------------------------#
        # Predict labels.
        f_data_features = f_data_features.dropna(
            dim = "sample",
            how = "all"
        )
        predicted_labels = self.predict(
            f_data_features = f_data_features,
            f_mask = f_mask
        ).labels
        f_data_labels = f_data_labels.sel(sample = predicted_labels.sample)
        #---------------------------------------------------------------------#
        # Estimate accuracy.
        if f_metric == "root_mean_square_error":
            metric_value = float(np.sqrt(
                (f_data_labels - predicted_labels) ** 2.
            ).mean(dim = "sample"))
        elif f_metric == "mean_absolute_error":
            metric_value = float(np.abs(
                f_data_labels - predicted_labels
            ).mean(dim = "sample"))
        elif f_metric == "median_absolute_deviation":
            metric_value = float(np.abs(
                f_data_labels - predicted_labels
            ).median(dim = "sample"))
        elif f_metric == "accuracy":
            metric_value = float(
                (f_data_labels == predicted_labels).sum(dim = "sample")
                / f_data_labels.size
            )
        #---------------------------------------------------------------------#
        # Update attributes.
        self._attributes()
        #---------------------------------------------------------------------#
        return metric_value
    #=========================================================================#
    def Davies_Bouldin_Index(
        self,
        f_data_training = None
    ):
        """
        Estimate the Davies-Bouldin index. The index is defined as the average
        similarity measure of each cluster with its most similar cluster, where
        similarity is the ratio of within-cluster distances to between-cluster
        distances. Thus, clusters which are farther apart and less dispersed
        will result in a better score.
        The minimum score is 0, with lower values indicating better clustering.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.

        Returns
        -------
        db_index : float
            The Davies-Bouldin index.
        """
        #---------------------------------------------------------------------#
        # Check if training data is available.
        if (
            f_data_training is None
            and self._training_data is None
        ):
            raise NotImplementedError(
                "If the Self_Organising_Map object is initialised from file, "
                "f_data_training must be provided to compute the "
                "Davies-Bouldin index."
            )
        if f_data_training is None:
            f_data_training = self._training_data.copy(deep = True)
        #---------------------------------------------------------------------#
        # Check input data for missing values.
        if np.isnan(f_data_training).any():
            raise ValueError("Data cannot contain missing values.")
        #---------------------------------------------------------------------#
        # Predict the classes for the input data.
        predictions = self.predict(
            f_data_features = f_data_training,
            f_mask = None
        ).grid_cell
        #---------------------------------------------------------------------#
        # Define look-up table for SOM grid cells.
        lookup_table = {
            gc : cnum for cnum, gc in enumerate(np.unique(predictions))
        }
        #---------------------------------------------------------------------#
        # Compute Davies-Bouldin score.
        db_index = davies_bouldin_score(
            f_data_training,
            [lookup_table[gc] for gc in predictions.values]
        )
        #---------------------------------------------------------------------#
        return float(db_index)
    #=========================================================================#
    def compute_umatrix(self):
        """
        Compute the U-matrix corresponding to the SOM grid. The U-matrix gives
        the local distance structure at each neuron. It is calculated as the
        sum of the distances between a SOM class and all of its immediate
        neighbours. The U-matrix is normalised by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Returns
        -------
        self.u_matrix : xarray.DataArray
            An xarray.DataArray containing the U-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        # Check if SOM is trained.
        if not self._trained:
            raise NotImplementedError(
                "Self_Organising_Map object has no compute_umatrix() method "
                "until after calling train()."
            )
        #---------------------------------------------------------------------#
        # Split data into chunks if necessary.
        if (
            self._chunk_bytes is not None
            and self.som.weights.size * self.n_classes * 8. > self._chunk_bytes
        ):
            compute_with_dask = True
            chunk_size = int(np.ceil(
                self._chunk_bytes / (self.som.weights.size * 8.)
            ))
            som_grid_cells = xr.DataArray(
                data = self.som.grid_cell.values,
                coords = {"gc_tmp" : (("gc_tmp", ), self.som.grid_cell.values)}
            ).chunk({
                "gc_tmp" : chunk_size
            })
        else:
            compute_with_dask = False
            som_grid_cells = self.som.grid_cell.rename({
                "grid_cell" : "gc_tmp"
            })
        #---------------------------------------------------------------------#
        # Determine neighbourhood of each grid cell on the SOM grid.
        if self.lattice == "rect":
            squared_distances = self._rect_grid_distance_squared(
                f_bmu_coordinates = som_grid_cells
            )
        elif self.lattice == "hexa":
            squared_distances = self._hex_grid_distance_squared(
                f_bmu_coordinates = som_grid_cells
            )
        neighbourhood_weights = self.som.weights.where(squared_distances == 1.)
        #---------------------------------------------------------------------#
        # Compute U-matrix.
        self.u_matrix = np.sqrt(
            ((
                self.som.weights.rename({"grid_cell" : "gc_tmp"})
                - neighbourhood_weights
            ) ** 2.).sum(
                dim = "feature"
            )
        ).sum(
            dim = "gc_tmp"
        )
        self.u_matrix = self.u_matrix / self.u_matrix.max()
        if compute_with_dask is True:
            with Client(self._cluster) as client:
                self.u_matrix = self.u_matrix.compute()
        #---------------------------------------------------------------------#
        return self.u_matrix
    #=========================================================================#
    def compute_pmatrix(
        self,
        f_data_training = None
    ):
        """
        Compute the P-matrix corresponding to the SOM grid. The P-matrix gives
        the local data density at each neuron. It is calculated as the number
        of data points within a defined hypersphere around each neuron. The
        P-matrix is normalised by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.

        Returns
        -------
        self.p_matrix : xarray.DataArray
            An xarray.DataArray containing the P-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        # Check if training data is available.
        if (
            f_data_training is None
            and self._training_data is None
        ):
            raise NotImplementedError(
                "If the Self_Organising_Map object is initialised from file, "
                "f_data_training must be provided to compute the P-matrix."
            )
        if f_data_training is None:
            f_data_training = self._training_data.copy(deep = True)
        #---------------------------------------------------------------------#
        # Check for missing values.
        if np.isnan(f_data_training).any():
            raise ValueError("Data cannot contain missing values.")
        #---------------------------------------------------------------------#
        # Determine distances between samples.
        sample_distances = squareform(pdist(
            X = f_data_training,
            metric = "euclidean"
        ))
        sample_distances = sample_distances[np.tril_indices_from(
            arr = sample_distances,
            k = -1
        )]
        #---------------------------------------------------------------------#
        # Determine cutoff radius.
        if use_abcanalysis is True:
            assumed_pareto = np.quantile(
                sample_distances,
                q = 0.2
            )
            abc_results = abc_analysis(
                psData = sample_distances,
                boolPlotResult = False
            )
            cutoff_radius = 1. / (
                min(sample_distances[abc_results["Aind"]])
                / max(sample_distances[abc_results["Cind"]])
            ) * assumed_pareto
        else:
            cutoff_radius = np.quantile(
                sample_distances,
                q = 0.4
            )
        #---------------------------------------------------------------------#
        # Split data into chunks if necessary.
        if (
            self._chunk_bytes is not None
            and f_data_training.size * self.n_classes * 8. > self._chunk_bytes
        ):
            compute_with_dask = True
            if type(self._chunk_size) in (str, float):
                chunk_size = int(np.ceil(
                    self._chunk_bytes / (self.n_classes * self.n_features * 8.)
                ))
            else:
                chunk_size = self._chunk_size
            f_data_training = f_data_training.chunk({"sample" : chunk_size})
        else:
            compute_with_dask = False
        #---------------------------------------------------------------------#
        # Compute P-matrix.
        self.p_matrix = (
            np.sqrt(
                ((f_data_training - self.som.weights) ** 2.).sum(
                    dim = "feature"
                )
            ) <= cutoff_radius
        ).sum(
            dim = "sample"
        )
        self.p_matrix = self.p_matrix / self.p_matrix.max()
        if compute_with_dask is True:
            with Client(self._cluster) as client:
                self.p_matrix = self.p_matrix.compute()
        #---------------------------------------------------------------------#
        return self.p_matrix
    #=========================================================================#
    def compute_ustarmatrix(
        self,
        f_data_training = None
    ):
        """
        Compute the U*-matrix corresponding to the SOM grid. The U*-matrix
        is a scaled version of the U-matrix, allowing for a more clear
        depiction of cluster boundaries. The values of the U-matrix are
        dampened in highly dense regions, unaltered in regions of average
        density and emphasised in sparse regions. The U*-matrix is normalised
        by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.

        Returns
        -------
        self.ustar_matrix : xarray.DataArray
            An xarray.DataArray containing the U*-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        # Check if training data is available.
        if (
            f_data_training is None
            and self._training_data is None
        ):
            raise NotImplementedError(
                "If the Self_Organising_Map object is initialised from file, "
                "f_data_training must be provided to compute the U*-matrix."
            )
        #---------------------------------------------------------------------#
        # Compute U-matrix and P-matrix if necessary.
        if self.u_matrix is None:
            _ = self.compute_umatrix()
        if self.p_matrix is None:
            _ = self.compute_pmatrix(f_data_training = f_data_training)
        #---------------------------------------------------------------------#
        # Determine U-matrix scaling factors.
        Pmedian = float(self.p_matrix.median())
        P95percentile = float(self.p_matrix.quantile(0.95))
        denominator = P95percentile - Pmedian
        if denominator < 0.01:
            denominator = 1.
        factor_A = -1. / denominator
        factor_B = -factor_A * P95percentile
        scaling_factor = factor_A * self.p_matrix + factor_B
        scaling_factor = np.maximum(scaling_factor, 0.)
        scaling_factor = np.minimum(scaling_factor, 1.)
        #---------------------------------------------------------------------#
        # Compute U*-matrix.
        self.ustar_matrix = self.u_matrix * scaling_factor
        high_val_idx = self.u_matrix.grid_cell.where(
            self.u_matrix > float(self.ustar_matrix.mean()),
            drop = True
        )
        self.ustar_matrix.loc[high_val_idx] = (
            0.5 * self.u_matrix.sel(grid_cell = high_val_idx)
            + 0.5 * self.ustar_matrix.sel(grid_cell = high_val_idx)
        )
        self.ustar_matrix = np.minimum(
            self.ustar_matrix,
            float(self.u_matrix.max())
        )
        self.ustar_matrix = self.ustar_matrix / self.ustar_matrix.max()
        #---------------------------------------------------------------------#
        return self.ustar_matrix
    #=========================================================================#
    def SOM_to_netcdf(
        self,
        f_file_save = None
    ):
        """
        Save a trained SOM to a NetCDF-file.

        Parameters
        ----------
        f_file_save : pathlib.PosixPath
            The absolute path pointing to the file to which to save the SOM.
        """
        #---------------------------------------------------------------------#
        if not self._trained:
            raise NotImplementedError(
                "Self_Organising_Map object has no SOM_to_netcdf() method "
                "until after calling train()."
            )
        #---------------------------------------------------------------------#
        # Prepare xarray.Dataset for saving.
        som_data_save = self.som.copy(deep = True)
        if self._normed is True:
            som_data_save = som_data_save.assign({
                "norm_params_0":som_data_save.weights.attrs["norm_params"][0],
                "norm_params_1":som_data_save.weights.attrs["norm_params"][1]
            })
            del som_data_save.weights.attrs["norm_params"]
        som_data_save = som_data_save.assign_coords({
            "gc_row" : som_data_save.grid_cell.real,
            "gc_col" : som_data_save.grid_cell.imag
        }).drop_vars("grid_cell")
        #---------------------------------------------------------------------#
        # Add relevant parameters to attributes.
        som_data_save = som_data_save.assign_attrs({
            "epochs" : self.epochs,
            "sigma1" : self.sigma1,
            "sigma0" : self.sigma0,
            "beta" : self.beta
        })
        #---------------------------------------------------------------------#
        # Save SOM to NetCDF file.
        som_data_save.to_netcdf(
            f_file_save,
            compute = True
        )
    #=========================================================================#
    def visualise_SOM(
        self,
        f_figure_width = 10,
        f_feature_dict = {0 : "Spectral_r"}
    ):
        """
        Visualise the weights of the SOM on its native lattice (either
        rectangular or hexagonal).

        Parameters
        ----------
        f_figure_width : float
            The width of the figure in inches. The height is set automatically.
        f_feature_dict : dict
            A dictionary whose keys define the features to plot and the
            corresponding values define the colourmap to use. The key can
            either be a single value or a tuple of three features. If the key
            is a tuple of three features the colourmap will be ignored and the
            weights corresponding to the three features are converted to RGB
            triplets. The length of f_feature_dict defines the total number of
            subplots in the figure.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The matplotlib.figure.Figure instance of the plot.
        axes : dict
            A dictionary containing the matplotlib.axes._axes.Axes instances of
            the individual subplots. The keys are the same as in
            f_feature_dict.
        """
        #---------------------------------------------------------------------#
        # Set parameters for plotting.
        polygon_params = self._visualisation_grid()
        rgba_values = self._map_values_to_colours(
            f_feature_dict = f_feature_dict
        )
        figure_specs = self._figure_params(f_feature_dict = f_feature_dict)
        #---------------------------------------------------------------------#
        # Generate figure.
        figure = plt.figure(
            "SOM grid",
            figsize = (f_figure_width, f_figure_width * figure_specs["ratio"]),
            clear = True
        )
        figure.subplots_adjust(
            left = 0.05,
            right = 0.95,
            bottom = 0.05,
            top = 0.95,
            hspace = 0.1,
            wspace = figure_specs["wspace"]
        )
        axis_grid = figure.add_gridspec(
            nrows = figure_specs["nrows"],
            ncols = figure_specs["ncols"]
        )
        axes = {}
        for i, (feat, vals) in enumerate(rgba_values.items()):
            axes.update({
                feat : figure.add_subplot(axis_grid[
                    i // figure_specs["ncols"],
                    i % figure_specs["ncols"]
                ])
            })
            polygons = [
                RegularPolygon(
                    (x, y),
                    numVertices = polygon_params["n_vertices"],
                    radius = polygon_params["radius"],
                    orientation = polygon_params["orientation"]
                )
                for x, y in zip(polygon_params["x"], polygon_params["y"])
            ]
            axes[feat].add_artist(
                PatchCollection(
                    patches = polygons,
                    facecolors = vals,
                    edgecolors = "black",
                    linewidth = 1
                )
            )
            axes[feat].set_aspect("equal")
            axes[feat].set_xlim(
                polygon_params["x"].min() - polygon_params["ax_pad"],
                polygon_params["x"].max() + polygon_params["ax_pad"]
            )
            axes[feat].set_ylim(
                polygon_params["y"].max() + polygon_params["ax_pad"],
                polygon_params["y"].min() - polygon_params["ax_pad"]
            )
            axes[feat].axis("off")
            axes[feat] = self._set_title(
                f_axis = axes[feat],
                f_feature = feat
            )
        #---------------------------------------------------------------------#
        return figure, axes
    #=========================================================================#
    def _attributes(self):
        """
        Add all attributes that are not None to a dictionary.
        """
        #---------------------------------------------------------------------#
        # Print attributes.
        self.attrs = {
            key : val
            for key, val in zip(
                (
                    "m", "n", "n_classes", "n_features",
                    "lattice", "shape", "epochs", "n_samples",
                    "sigma1", "sigma0", "beta"
                ),
                (
                    self.m, self.n, self.n_classes, self.n_features,
                    self.lattice, self.shape, self.epochs, self.n_samples,
                    self.sigma1, self.sigma0, self.beta
                )
            )
            if val is not None
        }
        if len(self.epochs) == 0:
            del (
                self.attrs["epochs"], self.attrs["sigma1"],
                self.attrs["sigma0"], self.attrs["beta"]
            )
        elif len(self.epochs) == 1:
            self.attrs.update({
                key : it[0] for key, it in self.attrs.items()
                if key in ("epochs", "sigma1", "sigma0", "beta")
            })
    #=========================================================================#
    def _compute_auto_correlation(
        self,
        f_data = None
    ):
        """
        Compute the auto-correlation matrix of a dataset.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray of shape (j, l) containing j samples of the l
            features.

        Returns
        -------
        auto_correlation : xarray.DataArray
            An xarray.DataArray of shape (l, l) containing the auto-correlation
            matrix.
        """
        #---------------------------------------------------------------------#
        # Compute auto correlation matrix.
        f_data = f_data - f_data.mean(dim = "sample")
        if (
            self._chunk_bytes is not None
            and f_data.size * self.n_features * 8. > self._chunk_bytes
        ):
            if type(self._chunk_size) in (str, float):
                chunk_size = int(np.ceil(
                    self._chunk_bytes / (self.n_features ** 2. * 8.)
                ))
            else:
                chunk_size = self._chunk_size
            f_data = f_data.chunk({"sample" : chunk_size})
            auto_correlation = (
                f_data * f_data.rename({"feature" : "feature1"})
            ).sum(
                dim = "sample"
            ) / self.n_samples
            with Client(self._cluster) as client:
                if self._verbose == 1:
                    print("Compute auto-correlation matrix.")
                auto_correlation = auto_correlation.compute()
        else:
            auto_correlation = (
                f_data * f_data.rename({"feature" : "feature1"})
            ).sum(
                dim = "sample"
            ) / self.n_samples
        #---------------------------------------------------------------------#
        return auto_correlation
    #=========================================================================#
    def _multiple_idxmin(
        self,
        f_data = None,
        f_indices = None,
        f_num_values = 1
    ):
        """
        Determine the indices of the k smallest entries in an 1-dimensional
        array.

        Parameters
        ----------
        f_data : numpy.array
            An 1-dimensional numpy.array containing the values to sort.
        f_indices : numpy.array
            An 1-dimensional numpy.array containing the indices corresponding
            to f_data.
        f_num_values : int
            The number of values to determine (k).

        Returns
        -------
        indices_smallest : numpy.array
            An 1-dimensional numpy.array containing the indices of the k
            smallest values in f_data. The indices are sorted by increasing
            order, i.e. indices_smallest[0] corresponds to the smallest value
            in f_data, indices_smallest[1] corresponds to the second smallest
            value in f_data, etc.
        """
        #---------------------------------------------------------------------#
        # Determine k smallest values.
        idx_k_smallest = np.argpartition(
            a = f_data,
            kth = f_num_values
        )[:f_num_values]
        #---------------------------------------------------------------------#
        # Sort values.
        values_k_smallest = f_data[idx_k_smallest]
        idx_sort =  np.argsort(values_k_smallest)
        #---------------------------------------------------------------------#
        # Determine data indices of k smallest values.
        indices_smallest = f_indices[idx_k_smallest][idx_sort]
        #---------------------------------------------------------------------#
        return indices_smallest
    #=========================================================================#
    def _find_bmu(
        self,
        f_data = None,
        f_mask = None,
        f_num_values = 1
    ):
        """
        Determine the coordinates of the best matching units on the SOM grid
        for given input vectors. If f_mask is provided, the distance
        calculation will be weighted according to f_mask.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray of shape (j, l) containing j samples of the l
            features for which to find the best matching units.
        f_mask : xarray.DataArray, "auto" or None
            An xarray.DataArray of shape (l, ) or (j, l) to be used to weight
            the BMU search. If "auto", determines the correlation between
            missing and available values in the dataset used to train the SOM
            and weights the BMU search based on the correlations. If None, all
            weights are set to None, i.e. equal weights for all inputs.
        f_num_values : int
            The number of values to determine (k).

        Returns
        -------
        bmu_coordinates : xarray.DataArray
            An xarray.DataArray containing the coordinates of the best matching
            units on the SOM grid. If f_num_values is 1, bmu_coordinates will
            have shape (j, ), else (j, f_num_values).
        """
        #---------------------------------------------------------------------#
        # Determine coordinates of BMU.
        if f_mask is None:
            f_mask = xr.ones_like(
                f_data.isel(sample = 0)
            ).drop_vars(
                "sample",
                errors = "ignore"
            )
        elif type(f_mask) is str and f_mask == "auto":
            f_mask = self._weight_by_correlation(f_data = None)
        sample_indices = f_data.sample
        f_data = f_data.dropna(
            dim = "sample",
            how = "all"
        )
        if (
            self._chunk_bytes is not None
            and f_data.size * self.n_classes * 8. > self._chunk_bytes
        ):
            if type(self._chunk_size) in (str, float):
                chunk_size = int(np.ceil(
                    self._chunk_bytes / (self.n_classes * self.n_features * 8.)
                ))
            else:
                chunk_size = self._chunk_size
            f_data = f_data.chunk({"sample" : chunk_size})
            distance_squared = (
                (self.som.weights - f_data) ** 2. * f_mask
            ).sum(
                dim = "feature"
            )
            with Client(self._cluster) as client:
                distance_squared = distance_squared.compute()
        else:
            distance_squared = (
                (self.som.weights - f_data) ** 2. * f_mask
            ).sum(
                dim = "feature"
            )
        bmu_coordinates = xr.apply_ufunc(
            self._multiple_idxmin,
            distance_squared,
            distance_squared.grid_cell,
            kwargs = {"f_num_values" : f_num_values},
            input_core_dims = [["grid_cell"], ["grid_cell"]],
            output_core_dims = [["n"]],
            exclude_dims = set(("grid_cell", )),
            vectorize = True,
            keep_attrs = True
        ).reindex(
            sample = sample_indices
        )
        try:
            bmu_coordinates = bmu_coordinates.squeeze(
                dim = "n",
                drop = True
            )
        except:
            pass
        #---------------------------------------------------------------------#
        return bmu_coordinates
    #=========================================================================#
    def _weight_by_correlation(
        self,
        f_data = None
    ):
        """
        Calculate the Spearman correlation coefficients between all pairs of
        features in the dataset used to train the SOM. Based on the resulting
        correlation matrix, weights are estimated to put on the respective
        available features in a sample when searching the BMU. I.e. if features
        are missing in a sample, those features that generally correlate more
        strongly with the missing feature are more relevant in the BMU search
        compared to features that generally correlate less strong.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray of shape (j, l) containing j samples of the l
            features for which estimate the weights.

        Returns
        -------
        feature_weights : xarray.DataArray
            An xarray.DataArray of shape (j, l) determining the weight to put
            on each feature of each sample during the BMU search.
        """
        #---------------------------------------------------------------------#
        # Rank input data.
        if f_data is None:
            f_data = self._training_data
        data_rank = f_data.rank(dim = "sample").to_pandas()
        #---------------------------------------------------------------------#
        # Compute squared correlation matrix.
        sq_corr_matrix = xr.DataArray(
            data = data_rank.corr(method = "pearson"),
            coords = {
                "feature" : (("feature", ), f_data.feature.values),
                "f_miss" : (("f_miss", ), f_data.feature.values)
            }
        ).fillna(0.) ** 2.
        #---------------------------------------------------------------------#
        # Determine the correlation weights between missing and available
        # features.
        if (
            self._chunk_bytes is not None
            and sq_corr_matrix.size*f_data.sample.size*8. > self._chunk_bytes
        ):
            if type(self._chunk_size) in (str, float):
                chunk_size = int(np.ceil(
                    self._chunk_bytes / (sq_corr_matrix.size * 8.)
                ))
            else:
                chunk_size = self._chunk_size
            valid_mask = np.isfinite(f_data).chunk({"sample" : chunk_size})
            feature_weights = sq_corr_matrix.where(
                ~valid_mask.rename({"feature" : "f_miss"})
                & valid_mask,
                0.
            ).sum(
                dim = "f_miss"
            ) + 1.
            with Client(self._cluster) as client:
                if self._verbose == 1:
                    print("Estimate correlation weights.")
                feature_weights = feature_weights.compute()
        else:
            feature_weights = sq_corr_matrix.where(
                np.isnan(f_data).rename({"feature" : "f_miss"})
                & np.isfinite(f_data),
                0.
            ).sum(
                dim = "f_miss"
            ) + 1.
        #---------------------------------------------------------------------#
        return feature_weights
    #=========================================================================#
    def _rect_grid_distance_squared(
        self,
        f_bmu_coordinates = None
    ):
        """
        Calculate the squared Euclidean distance between the BMU and all grid
        cells of the SOM on a rectangular lattice.

        Parameters
        ----------
        f_bmu_coordinates : xarray.DataArray
            An xarray.DataArray containing the coordinates of the BMUs.

        Returns
        -------
        distances_squared : xarray.DataArray
            An xarray.DataArray containing the squared Euclidean distance
            between the BMUs and all grid cells of the SOM.
        """
        #---------------------------------------------------------------------#
        # Compute Euclidean distance.
        if self.shape == "toroid":
            complex_abs = lambda cpx : np.abs(cpx.real) + 1j * np.abs(cpx.imag)
            distances = xr.concat(
                [
                    complex_abs(f_bmu_coordinates - self.som.grid_cell),
                    (
                        (self.m + 1j * self.n)
                        - complex_abs(f_bmu_coordinates-self.som.grid_cell)
                    )
                ],
                dim = "tmp_dim"
            )
            distances_squared = (
                distances.real.min(dim = "tmp_dim") ** 2.
                + distances.imag.min(dim = "tmp_dim") ** 2.
            )
        else:
            distances = np.abs(f_bmu_coordinates - self.som.grid_cell)
            distances_squared = distances ** 2.
        #---------------------------------------------------------------------#
        return distances_squared
    #=========================================================================#
    def _hex_grid_distance_squared(
        self,
        f_bmu_coordinates = None
    ):
        """
        Calculate the squared Manhattan distance between the BMU and all grid
        cells of the SOM on a hexagonal lattice.

        Parameters
        ----------
        f_bmu_coordinates : xarray.DataArray
            An xarray.DataArray containing the coordinates of the BMUs.

        Returns
        -------
        distances_squared : xarray.DataArray
            An xarray.DataArray containing the squared Manhattan distance
            between the BMUs and all grid cells of the SOM.
        """
        #---------------------------------------------------------------------#
        # Compute Manhattan distance.
        if self.shape == "toroid":
            f_bmu_coordinates = f_bmu_coordinates + self._offsets
            distances = f_bmu_coordinates - self.som.grid_cell
            distances = 0.5 * (
                np.abs(distances.real)
                + np.abs(distances.imag)
                + np.abs(distances.real + distances.imag)
            ).min(
                dim = "offsets"
            )
            distances_squared = distances ** 2.
        else:
            distances = f_bmu_coordinates - self.som.grid_cell
            distances = 0.5 * (
                np.abs(distances.real)
                + np.abs(distances.imag)
                + np.abs(distances.real + distances.imag)
            )
            distances_squared = distances ** 2.
        #---------------------------------------------------------------------#
        return distances_squared
    #=========================================================================#
    def _update_weights(
        self,
        f_data = None,
        f_radius = 1.
    ):
        """
        Update the weights of the SOM grid for given input vectors.

        Parameters
        ----------
        f_data : xarray.DataArray
            An xarray.DataArray of shape (j, l) containing j samples of the l
            features for which to update the SOM grid.
        f_radius : float
            The neighbourhood radius at the current epoch.
        """
        #---------------------------------------------------------------------#
        # Determine BMU coordinates.
        bmu_coordinates = self._find_bmu(
            f_data = f_data,
            f_mask = None,
            f_num_values = 1
        )
        #---------------------------------------------------------------------#
        # Update weights.
        if self.lattice == "hexa":
            distance_squared = self._hex_grid_distance_squared(
                f_bmu_coordinates = bmu_coordinates
            ) 
        else:
            distance_squared = self._rect_grid_distance_squared(
                f_bmu_coordinates = bmu_coordinates
            )
        distance_func = np.exp(-distance_squared / (2. * f_radius ** 2.))
        if (
            self._chunk_bytes is not None
            and f_data.size * self.n_classes * 8. > self._chunk_bytes
        ):
            if type(self._chunk_size) in (str, float):
                chunk_size = int(np.ceil(
                    self._chunk_bytes / (self.n_classes * self.n_features * 8.)
                ))
            else:
                chunk_size = self._chunk_size
            f_data = f_data.chunk({"sample" : chunk_size})
            distance_func = distance_func.where(np.isfinite(f_data))
            distance_func_summed = distance_func.sum(dim = "sample")
            distance_func_summed = distance_func_summed.where(
                distance_func_summed != 0.
            )
            self.som = self.som.assign({
                "weights" : self.som.weights.where(
                    ~np.isfinite(distance_func_summed),
                    (
                        (distance_func * f_data).sum(dim = "sample")
                        / distance_func_summed
                    )
                )
            })
            with Client(self._cluster) as client:
                self.som = self.som.compute()
        else:
            distance_func = distance_func.where(np.isfinite(f_data))
            distance_func_summed = distance_func.sum(dim = "sample")
            distance_func_summed = distance_func_summed.where(
                distance_func_summed != 0.
            )
            self.som = self.som.assign({
                "weights" : self.som.weights.where(
                    ~np.isfinite(distance_func_summed),
                    (
                        (distance_func * f_data).sum(dim = "sample")
                        / distance_func_summed
                    )
                )
            })
    #=========================================================================#
    def _assign_labels(
        self,
        f_data_features = None,
        f_data_labels = None,
        f_fill_value = -1
    ):
        """
        Assign labels to the trained SOM grid. For each vector in the input
        dataset the corresponding best matching unit is estimated. The label of
        a cell on the SOM grid is determined as the most common label at that
        cell corresponding to the given input data. For SOM grids with more
        classes than input samples, the labels are attributed based on their
        neighbourhood.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The training dataset. An xarray.DataArray of shape (j, l)
            containing j samples of the l features.
        f_data_labels : xarray.DataArray
            An xarray.DataArray of shape (l, ) containing the labels
            corresponding to f_data_features.
        f_fill_value : int, float or str
            The fill value to use if the attribution of a label to a grid cell
            is ambiguous, i.e. if different labels are found to be equally
            attributable to a grid cell.
        """
        #---------------------------------------------------------------------#
        # Define function to assign label to a single grid cell.
        def label_of_grid_cell(
            f_distances = None,
            f_sample_IDs = None,
            f_labels = None
        ):
            #-----------------------------------------------------------------#
            f_labels = f_labels[f_sample_IDs[f_distances == f_distances.min()]]
            unique_labels, counts = np.unique(
                f_labels,
                return_counts = True
            )
            grid_cell_label = unique_labels[counts == counts.max()]
            if grid_cell_label.size == 1:
                grid_cell_label = grid_cell_label[0]
            else:
                grid_cell_label = None
            #-----------------------------------------------------------------#
            return grid_cell_label
        #---------------------------------------------------------------------#
        # Determine BMU coordinates.
        bmu_coordinates = self._find_bmu(
            f_data = f_data_features,
            f_mask = None,
            f_num_values = 1
        )
        #---------------------------------------------------------------------#
        # Compute distances between BMUs and all grid cells.
        if self.lattice == "rect":
            distances_squared = self._rect_grid_distance_squared(
                f_bmu_coordinates = bmu_coordinates
            )
        else:
            distances_squared = self._hex_grid_distance_squared(
                f_bmu_coordinates = bmu_coordinates
            )
        #---------------------------------------------------------------------#
        # Determine labels per grid cell.
        grid_cell_labels = xr.apply_ufunc(
            label_of_grid_cell,
            distances_squared,
            f_data_features.sample,
            f_data_labels,
            input_core_dims = [["sample"], ["sample"], ["sample"]],
            exclude_dims = set(("sample", )),
            output_dtypes = ["object"],
            vectorize = True
        )
        #---------------------------------------------------------------------#
        # Replace ambiguous grid cell labels with fill value.
        grid_cell_labels = grid_cell_labels.where(
            grid_cell_labels != None,
            f_fill_value
        )
        #---------------------------------------------------------------------#
        # Convert to original data type.
        try:
            grid_cell_labels = grid_cell_labels.astype(f_data_labels.dtype)
        except:
            pass
        finally:
            self.som = self.som.assign({"labels" : grid_cell_labels})
    #=========================================================================#
    def _visualisation_grid(self):
        """
        Set the parameters of the polygons and the grid to plot the SOM
        weights.

        Returns
        -------
        param_dict : dict
            A dictionary containing the relevant parameters for the
            visualisation. The included values are:
                x : The x-coordinates of the grid.
                y : The y-coordinates of the grid.
                radius : The radius of the polygons to draw.
                orienation : The orientation of the polygons to draw.
                ax_pad : The padding between the polygons and the axes.
        """
        #---------------------------------------------------------------------#
        # Define grid cell coordinates.
        coord_x, coord_y = np.meshgrid(
            np.arange(self.n, dtype = "float32"),
            np.arange(self.m, dtype = "float32")
        )
        if self.lattice == "hexa":
            if self.shape in ("rectangle", "toroid"):
                coord_x[1::2, :] += 0.5
            elif self.shape == "rhombus":
                coord_x += np.arange(0., self.m * 0.5, 0.5).reshape(-1, 1)
            else:
                coord_x += np.arange(0., self.m * 0.5, 0.5).reshape(-1, 1)
                coord_x[self._grid_mask] = np.nan
                coord_y[self._grid_mask] = np.nan
            ratio = np.sqrt(3.) / 2.
            coord_y *= ratio
            radius = 1. / np.sqrt(3.)
            num_vertices = 6.
            orientation = np.radians(120.)
            ax_pad = radius * 1.1
        else:
            radius = 1. / np.sqrt(2.)
            num_vertices = 4.
            orientation = np.radians(45.)
            ax_pad = radius
        coord_x = coord_x.reshape(-1, 1)
        coord_x = coord_x[np.isfinite(coord_x)]
        coord_y = coord_y.reshape(-1, 1)
        coord_y = coord_y[np.isfinite(coord_y)]
        #---------------------------------------------------------------------#
        # Store grid parameters in dictionary.
        param_dict = {
            "x" : coord_x,
            "y" : coord_y,
            "radius" : radius,
            "n_vertices" : num_vertices,
            "orientation" : orientation,
            "ax_pad" : ax_pad
        }
        #---------------------------------------------------------------------#
        return param_dict
    #=========================================================================#
    def _map_values_to_colours(
        self,
        f_feature_dict = None
    ):
        """
        Map the values of the SOM weights to colours for each feature in
        f_feature_dict. Either maps one feature to a given colourmap or
        converts three features to RGB triplets.

        Parameters
        ----------
        f_feature_dict : dict
            A dictionary whose keys define the features to plot and the
            corresponding values define the colourmap to use. The key can
            either be a single value or a tuple of three features. If the key
            is a tuple of three features the value will be ignored and the
            weights corresponding to the three features are converted to RGB
            triplets.

        Returns
        -------
        rgba_values : dict
            A dictionary containing the RGB triplets corresponding to each
            feature / group of features in f_feature_dict.
        """
        #---------------------------------------------------------------------#
        # Map values to colours.
        rgba_values = {}
        if self._normed is True:
            weights_scaler = Scaler(
                f_dimension = None,
                f_method = self.som.weights
            )
            som_weights = weights_scaler.denormalise(
                f_data = self.som.weights
            )
        else:
            som_weights = self.som.weights
        for feat, cmap in f_feature_dict.items():
            if type(feat) is not tuple:
                norm = mcolors.Normalize(
                    vmin = som_weights.sel(feature = feat).min(),
                    vmax = som_weights.sel(feature = feat).max()
                )
                mapper = plt.cm.ScalarMappable(
                    norm = norm,
                    cmap = cmap
                )
                rgba_values.update({
                    feat : mapper.to_rgba(
                        som_weights.sel(feature = feat).values
                    )
                })
            else:
                if len(feat) not in (3, 4):
                    raise ValueError(
                        "Trying to plot SOM features as RGB triplet requires "
                        f"3 features, but {len(feat)} are given."
                    )
                rgba_scaler = Scaler(
                    f_dimension = "grid_cell",
                    f_method = "range"
                )
                rgba_values.update({
                    feat : rgba_scaler.normalise(
                        f_data = som_weights.sel(feature = list(feat))
                    ).values
                })
        #---------------------------------------------------------------------#
        return rgba_values
    #=========================================================================#
    def _figure_params(
        self,
        f_feature_dict = None
    ):
        """
        Set relevant figure parameters for the visualisation of the SOM grid.

        Parameters
        ----------
        f_feature_dict : dict
            A dictionary whose keys define the features to plot and the
            corresponding values define the colourmap to use.

        Returns
        -------
        param_dict : dict
            A dictionary containing the relevant figure parameters for the
            visualisation. The included values are
                ratio : The ratio between the height and width of the figure.
                nrows : The number of rows in the figure.
                ncols : The number of cols in the figure.
                wspace : The horizontal space between the individual subplots.
        """
        #---------------------------------------------------------------------#
        # Determine figure shape.
        if len(f_feature_dict) < 4:
            ncols = len(f_feature_dict)
            nrows = 1
        else:
            ncols = int(np.ceil(np.sqrt(len(f_feature_dict))))
            nrows = int(np.ceil(len(f_feature_dict) / ncols))
        #---------------------------------------------------------------------#
        # Set figure parameters.
        if self.lattice == "rect":
            figure_ratio = (self.m / (self.n * ncols)) * nrows
            wspace = 0.05
        elif self.lattice == "hexa" and self.shape in ("rectangle", "toroid"):
            figure_ratio = (
                (2. * np.ceil(0.5 * self.m) + np.floor(0.5 * self.m))
                / (np.sqrt(3.) * (self.n * ncols + 0.5))
            ) * nrows
            wspace = -0.12
        elif self.lattice == "hexa" and self.shape == "rhombus":
            figure_ratio = (
                (2. * np.ceil(0.5 * self.m) + np.floor(0.5 * self.m))
                / (np.sqrt(3.) * (self.n * ncols + 0.5 * (self.m - 1.)))
            ) * nrows
            wspace = -(1. - (self.n + 0.2) / (self.n + 0.5 * (self.m - 1.)))
        elif self.lattice == "hexa" and self.shape == "hexagon":
            figure_ratio = (
                (2. * np.ceil(0.5 * self.m) + np.floor(0.5 * self.m))
                / (np.sqrt(3.) * self.n * ncols)
            ) * nrows
            wspace = 0.05
        #---------------------------------------------------------------------#
        # Store figure parameters in dictionary.
        figure_dict = {
            "ratio" : figure_ratio,
            "nrows" : nrows,
            "ncols" : ncols,
            "wspace" : wspace
        }
        #---------------------------------------------------------------------#
        return figure_dict
    #=========================================================================#
    def _set_title(
        self,
        f_axis = None,
        f_feature = None
    ):
        """
        Set the title of a subplot according to whether a single feature or a
        group of features is plotted. If a group of features is plotted, the
        name of the feature in the title is coloured according to the colour it
        represents in the RGB triplet.

        Parameters
        ----------
        f_axis : matplotlib.axes._axes.Axes
            The matplotlib.axes._axes.Axes instance of the subplot for which to
            set the title.
        f_feature : int, str, or tuple
            The name(s) of the feature(s) that are plotted in the respective
            subplot.

        Returns
        -------
        f_axis : matplotlib.axes._axes.Axes
            The matplotlib.axes._axes.Axes instance of the subplot for which to
            set the title.
        """
        #---------------------------------------------------------------------#
        # Set axis title.
        if type(f_feature) is tuple:
            title = f_axis.set_title(
                "Feature: ",
                fontsize = 14,
                loc = "left",
                pad = 0
            )
            for j, (ft, col) in enumerate(
                zip(f_feature, ("red", "green", "blue", "grey")),
                start = 1
            ):
                title = f_axis.annotate(
                    ft,
                    xycoords = title,
                    xy = (1, 0),
                    fontsize = 14,
                    color = col,
                    horizontalalignment = "left",
                    verticalalignment = "bottom"
                )
                if j < len(f_feature):
                    title = f_axis.annotate(
                        " / ",
                        xycoords = title,
                        xy = (1, 0),
                        fontsize = 14,
                        horizontalalignment = "left",
                        verticalalignment = "bottom"
                    )
        else:
            f_axis.set_title(
                f"Feature: {f_feature}",
                fontsize = 14,
                loc = "left",
                pad = 0
            )
        #---------------------------------------------------------------------#
        return f_axis

#%% Class for iterative completion SOM.
class ITCOMPSOM:
    """
    Implementation of an Iterative Completion Self-Organising Map (ITCOMPSOM)
    algorithm. The ITCOMPSOM uses Self-Organising Maps (SOMs) to fill gaps in a
    given dataset. It is an iterative process with progressively larger
    topological maps build from progressively larger subsamples of the input
    data. At each iteration the topological maps combine previously completed
    data and new data with missing values. The missing data are filled after
    each iteration using the latest topological map, i.e. missing data is
    filled several times before the whole dataset is filled completely. The
    missing values are filled using the best matching units (BMUs) of the SOM.
    The BMUs are determined using the weighted Euclidian distance between the
    respective sample and the weights of the SOM. The Euclidian distance is
    weighted by the correlation between the missing and the available features
    in the sample. The Spearman correlation coefficients used are calculated
    using the available data of all pairs of features in the input dataset.

    Attributes
    ----------
    n_it : int
        The number of iterations performed.
    i_nc : int
        The initial number of SOM classes during the first iteration.
    f_nc : int
        The final number of SOM classes during the final iteration.
    n_classes : list
        A list of the number of SOM classes per iteration.

    Methods
    -------
    preprocessing()
        Prepare the input data for the gap imputation process and determine the
        weights for the BMU search.
    run()
        Perform the imputation process using the ITCOMPSOM method.
    restart_from_file()
        Restart an ITCOMPSOM run from an iteration stored in a NetCDF-file.
    assign_labels_to_SOM()
        Assign labels to the different SOM classes.
    predict()
        Predict the BMU for given data using the trained SOM.
    evaluate()
        Estimate the accuracy of the trained SOM if labels are available along-
        side the given features.
    Davies_Bouldin_Index()
        Estimate the Davies-Bouldin index for the trained SOM.
    compute_umatrix()
        Compute the U-matrix corresponding to the trained SOM.
    compute_pmatrix()
        Compute the P-matrix corresponding to the trained SOM.
    compute_ustarmatrix()
        Compute the U*-matrix corresponding to the trained SOM.
    visualise_SOM()
        Plot the weights of the SOM on the chosen lattice.
    """
    #=========================================================================#
    def __init__(
        self,
        f_chunk_size = "auto",
        f_dask_cluster = None,
        f_temporary_output = {
            "path" : None,
            "delete" : True
        },
        f_random_state = 12345,
        f_verbose = 1
    ):
        """
        Initialise the SOM instance, including initialising the random number
        generator and estimating the chunk size in bytes, if auto-chunking is
        used. The dask cluster is used only if the size of the data used during
        computation exceeds the chunk size limit. For auto-chunking the limit
        is set to 1/20 of the memory limit per worker.

        Parameters
        ----------
        f_chunk_size : str or int
            If "auto", the chunk size is computed based on the memory limit of
            the dask cluster, i.e. the chunks size is set such that each chunk
            is approximately 1/20 of the worker memory limit. If an int,
            specifies the chunks size along the sample dimension of the data.
            If a float, determines the chunk size as the fraction of the worker
            memory limit.
        f_dask_cluster : distributed.deploy.local.LocalCluster or None
            A dask cluster to use for computation for large datasets. If None
            and the dataset is to big, will throw a memory error.
        f_temporary_output : dict
            A dictionary containing the following variables:
            path : pathlib.PosixPath
                The absolute path pointing to the directory where to store the
                temporary data.
            delete : bool
                If True, results of the previous iteration are deleted once the
                results from the current iteration are stored to disc.
        f_random_state : int
            An integer defining the seed for the random number generator.
        f_verbose : int
            An integer defining whether output will be displayed in the
            console. If 0, no output will be shown. If 1, the progress bar for
            for the iterative completion SOM will be shown. If 2, progress
            bars and the number of cells of the SOM will be displayed for each
            iteration.
        """
        #---------------------------------------------------------------------#
        # Check for correct inputs.
        if f_verbose not in range(3):
            raise ValueError(
                f"f_verbose is {f_verbose}, but should be on of (0, 1, 2)."
            )
        #---------------------------------------------------------------------#
        # Set attributes.
        self._verbose = f_verbose
        self._feature_names = None
        self.n_it = 0
        self.i_nc = None
        self.f_nc = None
        self._nneurons = None
        self._subset_indices = None
        self.n_classes = None
        self._start_iteration = 0
        #---------------------------------------------------------------------#
        # Set path for temporary outputs.
        if f_temporary_output["path"] is None:
            self._path_output = Path().cwd().joinpath("temp_itcompsom")
        else:
            self._path_output = f_temporary_output["path"]
        self._path_output.mkdir(
            parents = True,
            exist_ok = True
        )
        self._delete_output = f_temporary_output["delete"]
        #---------------------------------------------------------------------#
        # Initialise Self Organising Map.
        self.som = Self_Organising_Map(
            f_chunk_size = f_chunk_size,
            f_dask_cluster = f_dask_cluster,
            f_random_state = f_random_state,
            f_verbose = self._verbose // 2
        )
    #=========================================================================#
    def catch_keyboard_interrupt(func):
        """
        A decorator to catch KeyboardInterrupt exceptions during the iterative
        process.
        """
        #---------------------------------------------------------------------#
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                print(
                    "Computation interrupted by user on iteration "
                    f"{self._iteration}.\nTo continue the computation, restart"
                    " it using the following parameters:\n"
                    f"    'path' : {self._path_output.absolute()}\n"
                    f"    'iteration' : {self._iteration - 1}"
                )
                return self.data
        #---------------------------------------------------------------------#
        return wrapper
    #=========================================================================#
    def preprocessing(
        self,
        f_data = None,
        f_norm_dim = "sample",
        f_norm_method = "variance"
    ):
        """
        Prepare the data and compute the squared correlation matrix between all
        pairs of features in the input dataset. Features that do not contain
        any valid data for any sample will be removed. The data is normalised
        and sorted by the number of missing features per sample. The squared
        correlation matrix is determined by calculating the Spearman
        correlation coefficient between the available data of all pairs of
        features in the input dataset. If no overlapping available data are
        available for any pair of features, the correlation coefficient is set
        to 0, i.e. the features are declared as being uncorrelated.

        Parameters
        ----------
        f_data : xarray.DataArray
            The training dataset. An xarray.DataArray of shape (k, l)
            containing k samples of the l features. The dimensions must be
            named 'sample' and 'feature'.
        f_norm_dim : str
            The dimension along which to normalise the data.
        f_norm_method : str
            The normalisation method used, either 'variance' or 'range'.
        """
        #---------------------------------------------------------------------#
        # Assign coordinate to dimensions if not available.
        if "feature" not in f_data.coords:
            f_data = f_data.assign_coords({
                "feature" : np.arange(f_data.feature.size)
            })
        if "sample" not in f_data.coords:
            f_data = f_data.assign_coords({
                "sample" : np.arange(f_data.sample.size)
            })
        #---------------------------------------------------------------------#
        # Check if all-NaN features are present in the data.
        self.data = f_data.dropna(
            dim = "feature",
            how = "all"
        )
        self._feature_names = self.data.feature.values
        self._all_features = f_data.feature.values
        drop_features = list(np.setdiff1d(
            self._all_features,
            self._feature_names
        ))
        if len(drop_features) > 0:
            warn(
                "\nThe following features will be removed from f_data, "
                f"as they do not contain any valid values:\n{drop_features}\n",
                category = DropFeaturesWarning,
                stacklevel = 2
            )
        #---------------------------------------------------------------------#
        # Normalise input data.
        self._data_scaler = Scaler(
            f_dimension = f_norm_dim,
            f_method = f_norm_method
        )
        self._data_norm = self._data_scaler.normalise(
            f_data = self.data
        )
        #---------------------------------------------------------------------#
        # Determine the correlation weights between missing and available
        # features.
        self._feature_weights = self.som._weight_by_correlation(
            f_data = self._data_norm
        )
        #---------------------------------------------------------------------#
        # Sort by number of missing values.
        sort_index = np.argsort(
            np.isnan(self._data_norm).sum(dim = "feature")
        ).values
        self._data_norm = self._data_norm.assign_coords({
            "sample_org" : self.data.sample
        })
        self._data_norm = self._data_norm.isel(
            sample = sort_index
        ).assign_coords({
            "sample" : (("sample", ), np.arange(self._data_norm.sample.size))
        })
        self._feature_weights = self._feature_weights.isel(
            sample = sort_index
        ).assign_coords({
            "sample" : (("sample", ), np.arange(self._data_norm.sample.size))
        })
        self._valid_mask = np.isfinite(self._data_norm)
        self._valid_mask.attrs.clear()
    #=========================================================================#
    @catch_keyboard_interrupt
    def run(
        self,
        f_lattice = "hexa",
        f_shape = "rectangle",
        f_SOM_size = (20, 60),
        f_num_iterations = 5,
        f_training_params = None
    ):
        """
        Build a Self-Organising Map (SOM) iteratively with progressively larger
        topological maps. After each iteration, missing values in the input
        data are filled by the best matching units of the current SOM, i.e. the
        missing values are filled several times. At each iteration a larger
        dataset is used, as well as an increasing number of SOM classes. The
        best matching units are determined by weighting the Euclidian distance
        between the weights of the SOM and the data vectors by the correlation
        between missing and available features for each sample.
        All relevant parameters are stored to the given output directory within
        a NetCDF-file after each iteration. These files can be used to restart
        the computation from the last stored iteration if the computation is
        interrupted. Additionally, the trained SOM is stored to a NetCDF-file
        after each iteration in the same directory. These are, however, not
        required for a restart.

        Parameters
        ----------
        f_lattice : str
            Either 'rect' or 'hexa'. Defines the neighbourhood of the SOM's
            grid cells. For 'rect' there are 8 direct neighbours, while for
            'hexa' there are only 6 direct neighbours.
        f_shape : str
            Defines the shape of the lattice. Either 'rectangle' or 'toroid'
            for both types of lattice. If lattice is 'hexa', the shape can also
            be 'rhombus' or 'hexagon'.
        f_SOM_size : tuple
            A tuple of int defining the approximate initial (during the first
            iteration) and the final (during the last iteration) sizes of the
            SOM grid.
        f_num_iterations : int
            The number of iterations.
        f_training_params : list
            A list of dict containing the parameters how the SOM is trained.
            For each dict in f_training_params, the train() method of
            Self_Organising_Map is called. Each dict must contain the following
            two parameters:
                epochs : float
                    The number of epochs for which to train the SOM.
                radius : dict
                    A dictionary containing the maximum neighbourhood radius
                    ("s1"), as well as either the minimum neighbourhood radius
                    ("s0") or the decay rate ("b"). The respective other
                    parameter has to be set to None.
            For more details refer to the train() method in
            Self_Organising_Map.

        Returns
        -------
        self.data : xarray.DataArray
            The completed input data.
        """
        #---------------------------------------------------------------------#
        # Check for correct inputs.
        if any(sz <  1 for sz in f_SOM_size):
            raise ValueError(
                "All entries in f_SOM_size should be > 1 but are "
                f"{f_SOM_size}."
            )
        if f_SOM_size[1] < f_SOM_size[0]:
            raise ValueError(
                "The final SOM size should be larger than the initial SOM "
                "size."
            )
        if f_num_iterations < 1:
            raise ValueError(
                f"f_num_iterations should be > 1 but is {f_num_iterations}."
            )
        if type(f_training_params) is not list:
            raise ValueError(
                "f_training_params should be a list, but is "
                f"{type(f_training_params)}."
            )
        if not all([
            "epochs" in dct and "radius" in dct
            for dct in f_training_params
        ]):
            raise ValueError(
                "Each dict in f_training_params should have 'epochs' and "
                "'radius' as keys."
            )
        if any([
            all([it is None for it in dct["radius"].values()])
            for dct in f_training_params
        ]):
            raise ValueError(
                "Only two parameters can be provided in "
                "f_training_params['radius']. Either 's0' or 'b' must be set "
                "to None."
            )
        if any([dct["radius"]["s1"] is None for dct in f_training_params]):
            raise ValueError(
                "f_training_params['radius']['s1'] cannot be None."
            )
        #---------------------------------------------------------------------#
        # Define function to adjust data before saving.
        def prepare_data_to_save(
            f_dataarray = None,
            f_new_name = "input_data"
        ):
            """
            Prepare xarray.DataArrays to be saved as temporary output. The
            original name and coordinates are saved as attributes. The
            xarray.DataArray and non-indexed coordinates are renamed.

            Parameters
            ----------
            f_dataarray : xarray.DataArray
                The xarray.DataArray to adjust.
            f_new_name : str
                The new name of the xarray.DataArray.

            Returns
            -------
            f_dataarray : xarray.DataArray
                The adjusted xarray.DataArray.
            """
            #-----------------------------------------------------------------#
            f_dataarray = f_dataarray.assign_attrs({
                "original_name" : str(f_dataarray.name),
                "original_coords" : [coord for coord in f_dataarray.coords],
                "rename_dict" : [
                    f"{coord}:{f_new_name}_coord_{i}"
                    for i, coord in enumerate(f_dataarray.coords)
                    if coord not in f_dataarray.dims
                ]
            })
            f_dataarray = f_dataarray.rename(f_new_name).rename({
                c.split(":")[0] : c.split(":")[1]
                for c in f_dataarray.rename_dict
            })
            #-----------------------------------------------------------------#
            return f_dataarray
        #---------------------------------------------------------------------#
        # Set the training parameters.
        if self._start_iteration == 0:
            # Set parameters.
            self.n_it = f_num_iterations
            self.i_nc = f_SOM_size[0]
            self.f_nc = f_SOM_size[1]
            self.n_classes = np.array([])
            # Determine the number of cells in the SOM for each iteration.
            #self._nneurons = np.round([
            #    max(self.f_nc * (it + 1) / self.n_it, self.i_nc)
            #    for it in range(self.n_it)
            #]).astype("int32")
            self._nneurons = np.round([
                self.i_nc + (self.f_nc - self.i_nc) * it / (self.n_it - 1)
                for it in range(self.n_it)
            ]).astype("int32")
            # Determine the end indices of the subsets for each iteration.
            self._subset_indices = np.ceil([
                self.data.sample.size * (it + 1) / self.n_it
                for it in range(self.n_it)
            ]).astype("int32")
        #---------------------------------------------------------------------#
        # Save relevant data to NetCDF.
        if self._start_iteration == 0:
            data_save = xr.merge([
                prepare_data_to_save(
                    f_dataarray = self.data,
                    f_new_name = "input_data"
                ),
                prepare_data_to_save(
                    f_dataarray = self._feature_weights,
                    f_new_name = "feature_weights"
                ),
                prepare_data_to_save(
                    f_dataarray = self._data_norm.norm_params[0],
                    f_new_name = "norm_params_0"
                ),
                prepare_data_to_save(
                    f_dataarray = self._data_norm.norm_params[1],
                    f_new_name = "norm_params_1"
                ),
                prepare_data_to_save(
                    f_dataarray = self._valid_mask,
                    f_new_name = "valid_mask"
                )
            ]).assign({
                "feature_names" : (("feature_names", ), self._feature_names),
                "all_features" : (("all_features", ), self._all_features),
                "nneurons" : (("epoch", ), self._nneurons),
                "subset_indices" : (("epoch", ), self._subset_indices)
            }).assign_attrs({
                "norm_dim" : self._data_norm.norm_dim,
                "norm_method" : self._data_norm.norm_method,
                "n_it" : self.n_it,
                "i_nc" : self.i_nc,
                "f_nc" : self.f_nc,
                "lattice" : f_lattice,
                "shape" : f_shape,
                "epochs" : [dct["epochs"] for dct in f_training_params]+[-1],
                "sigma1" : [
                    str(dct["radius"]["s1"]) for dct in f_training_params
                ] + [-1],
                "sigma0" : [
                    str(dct["radius"]["s0"]) for dct in f_training_params
                ] + [-1],
                "beta" : [
                    str(dct["radius"]["b"]) for dct in f_training_params
                ] + [-1]
            })
            data_save.to_netcdf(
                self._path_output.joinpath("parameters_general.nc"),
                compute = True
            )
        #---------------------------------------------------------------------#
        # Iterative gap imputation.
        for it, (sidx, nns) in tqdm(
            enumerate(
                zip(self._subset_indices, self._nneurons),
                start = 1
            ),
            total = self._nneurons.size,
            ncols = 85,
            desc = "ITCOMPSOM",
            disable = False if self._verbose == 1 else True,
            position = 0,
            leave = True
        ):
            # Skip iterations if re-started from file.
            self._iteration = it
            if it <= self._start_iteration:
                continue
            # Select the subset from the dataset.
            subset_data = self._data_norm.isel(sample = slice(0, sidx))
            # Compile the SOM.
            self.som.compile(
                f_features = self._feature_names,
                f_SOM_cols = None,
                f_SOM_rows = None,
                f_SOM_size = nns,
                f_lattice = f_lattice,
                f_shape = f_shape
            )
            # Train the SOM.
            for train_dict in f_training_params:
                _ = self.som.train(
                    f_data_features = subset_data,
                    f_data_labels = None,
                    f_epochs = train_dict["epochs"],
                    f_neighbourhood_radius = train_dict["radius"]
                )
            self.n_classes = np.append(self.n_classes, self.som.n_classes)
            # Determine the best matching units, weighted by the estimated
            # correlations.
            best_matching_units = self.som.predict(
                f_data_features = subset_data,
                f_mask = self._feature_weights.sel(
                    feature = subset_data.feature,
                    sample = subset_data.sample
                )
            ).weights.drop_vars("grid_cell")
            # Complete the missing observations with the corresponding
            # components from the BMUs.
            self._data_norm.loc[
                subset_data.sample,
                subset_data.feature
            ] = subset_data.where(
                self._valid_mask.sel(
                    feature = subset_data.feature,
                    sample = subset_data.sample
                ),
                best_matching_units
            )
            # Save interim results to NetCDF.
            data_save = self._data_norm.copy(deep = True)
            data_save.attrs.clear()
            data_save = data_save.assign_attrs({
                "n_classes" : self.n_classes
            })
            data_save.to_netcdf(
                self._path_output.joinpath(
                    f"parameters_iteration_{it:02d}.nc"
                ),
                compute = True
            )
            if (
                self._delete_output is True
                and self._path_output.joinpath(
                    f"parameters_iteration_{it - 1:02d}.nc"
                ).exists()
            ):
                self._path_output.joinpath(
                    f"parameters_iteration_{it - 1:02d}.nc"
                ).unlink()
            # Save SOM to NetCDF.
            self.som.SOM_to_netcdf(
                f_file_save = self._path_output.joinpath(
                    f"som_iteration_{it:02d}.nc"
                )
            )
            if (
                self._delete_output is True
                and self._path_output.joinpath(
                    f"som_iteration_{it - 1:02d}.nc"
                ).exists()
            ):
                self._path_output.joinpath(
                    f"som_iteration_{it - 1:02d}.nc"
                ).unlink()
        #---------------------------------------------------------------------#
        # Finalise results.
        self._data_norm = self._data_norm.sortby(
            variables = "sample_org"
        ).swap_dims({
            "sample" : "sample_org"
        }).drop_vars(
            "sample"
        ).rename({
            "sample_org" : "sample"
        })
        self.data = self._data_scaler.denormalise(
            f_data = self._data_norm
        ).reindex(
            feature = self._all_features
        )
        self.i_nc = self.n_classes[0]
        self.f_nc = self.n_classes[-1]
        self.som._training_data = self._data_norm
        #---------------------------------------------------------------------#
        return self.data.copy(deep = True)
    #=========================================================================#
    def restart_from_file(
        self,
        f_temporary_output = {
            "path" : None,
            "iteration" : 1,
            "delete" : True
        }
    ):
        """
        Restart the ITCOMPSOM from an iteration stored to a NetCDF-file. The
        relevant parameters are extracted and the computation is continued from
        the defined iteration.

        Parameters
        ----------
        f_temporary_output : dict
            A dictionary containing the following variables:
            f_temporary_output : dict
            A dictionary containing the following variables:
            path : pathlib.PosixPath
                The absolute path pointing to the directory where to store the
                temporary data.
            iteration : int
                The iteration from which to restart the computation.
            delete : bool
                If True, results of the previous iteration are deleted once the
                results from the current iteration are stored to disc.

        Returns
        -------
        self.data : xarray.DataArray
            The completed input data.
        """
        #---------------------------------------------------------------------#
        def recover_original_data(
            f_dataset = None,
            f_variable = "input_data"
        ):
            """
            Recover an xarray.DataArray, as it was originally supplied to the
            algorithm, from the temporary output stored as xarray.Dataset. The
            respective xarray.DataArray is extracted, renamed and the original
            coordinates are restored.

            Parameters
            ----------
            f_dataset : xarray.Dataset
                The xarray.Dataset from which to extract the xarray.DataArray.
            f_varialbe : str
                The name of the xarray.DataArray to extract.

            Returns
            -------
            dataarray : xarray.DataArray
                The extracted xarray.DataArray.
            """
            #-----------------------------------------------------------------#
            attribute = lambda attr : list(
                attr if isinstance(attr, (list, np.ndarray))
                else [attr]
            )
            #-----------------------------------------------------------------#
            dataarray = f_dataset[f_variable]
            dataarray = dataarray.rename(
                dataarray.original_name.replace("None", "")
            ).rename({
                c.split(":")[1] : c.split(":")[0]
                for c in attribute(dataarray.rename_dict)
            })
            dataarray = dataarray.drop_vars([
                coord for coord in dataarray.coords
                if coord not in attribute(dataarray.original_coords)
            ])
            #-----------------------------------------------------------------#
            del (
                dataarray.attrs["original_name"],
                dataarray.attrs["original_coords"],
                dataarray.attrs["rename_dict"]
            )
            #-----------------------------------------------------------------#
            return dataarray.load()
        #---------------------------------------------------------------------#
        # Set attributes.
        self._path_output = f_temporary_output["path"]
        self._delete_output = f_temporary_output["delete"]
        self._start_iteration = f_temporary_output["iteration"]
        #---------------------------------------------------------------------#
        # Load general parameters.
        with xr.open_dataset(
            self._path_output.joinpath("parameters_general.nc")
        ) as data_general:
            data_general.load()
        self.data = recover_original_data(
            f_dataset = data_general,
            f_variable = "input_data"
        )
        self._feature_weights = recover_original_data(
            f_dataset = data_general,
            f_variable = "feature_weights"
        )
        self._valid_mask = recover_original_data(
            f_dataset = data_general,
            f_variable = "valid_mask"
        )
        self._feature_names = data_general.feature_names.values
        self._all_features = data_general.all_features.values
        self._nneurons = data_general.nneurons.values
        self._subset_indices = data_general.subset_indices.values
        self.n_it = data_general.n_it
        self.i_nc = data_general.i_nc
        self.f_nc = data_general.f_nc
        #---------------------------------------------------------------------#
        # Load parameters of last iteration.
        with xr.open_dataarray(self._path_output.joinpath(
            f"parameters_iteration_{f_temporary_output['iteration']:02d}.nc"
        )) as self._data_norm:
            self._data_norm.load()
        self._data_norm = self._data_norm.assign_attrs({
            "norm_dim" : data_general.norm_dim,
            "norm_method" : data_general.norm_method,
            "norm_params" : [
                recover_original_data(data_general, "norm_params_0"),
                recover_original_data(data_general, "norm_params_1")
            ]
        })
        self.n_classes = self._data_norm.attrs.pop("n_classes")
        #---------------------------------------------------------------------#
        # Initialise scaler.
        self._data_scaler = Scaler(
            f_dimension = self._data_norm.norm_dim,
            f_method = self._data_norm.norm_method
        )
        #---------------------------------------------------------------------#
        # Run ITCOMPSOM.
        self.data = self.run(
            f_lattice = data_general.lattice,
            f_shape = data_general.shape,
            f_SOM_size = (self.i_nc, self.f_nc),
            f_num_iterations = self.n_it,
            f_training_params = [
                {
                    "epochs" : epochs,
                    "radius" : {
                        "s1" : eval(s1), "s0" : eval(s0), "b" : eval(b)
                    }
                }
                for epochs, s1, s0, b in zip(
                    data_general.epochs[:-1],
                    data_general.sigma1[:-1],
                    data_general.sigma0[:-1],
                    data_general.beta[:-1]
                )
            ]
        )
        #---------------------------------------------------------------------#
        return self.data.copy(deep = True)
    #=========================================================================#
    def assign_labels_to_SOM(
        self,
        f_data_features = None,
        f_data_labels = None,
        f_fill_value = -1
    ):
        """
        Assign labels to the trained SOM grid. For each vector in the input
        dataset the corresponding best matching unit is estimated. The label of
        a cell on the SOM grid is determined as the most common label at that
        cell corresponding to the given input data. For SOM grids with more
        classes than input samples, the labels are attributed based on their
        neighbourhood.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The training dataset. An xarray.DataArray of shape (j, l)
            containing j samples of the l features.
        f_data_labels : xarray.DataArray
            An xarray.DataArray of shape (l, ) containing the labels
            corresponding to f_data_features.
        f_fill_value : int, float or str
            The fill value to use if the attribution of a label to a grid cell
            is ambiguous, i.e. if different labels are found to be equally
            attributable to a grid cell.
        """
        #---------------------------------------------------------------------#
        return self.som._assign_labels(
            f_data_features = f_data_features,
            f_data_labels = f_data_labels,
            f_fill_value = f_fill_value
        )
    #=========================================================================#
    def predict(
        self,
        f_data_features = None,
        f_mask = None
    ):
        """
        Determine the best matching units for the given dataset. If f_mask is
        provided, the distance calculation will be weighted according to
        f_mask. If f_data_features contains missing values, f_mask can be set
        to "auto". In this case, the distance calculation will be weighted by
        the correlation between the features of the data used to train the SOM.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The dataset for which to estimate the best matching units. An
            xarray.DataArray of shape (j, l) containing j samples of the l
            features. The dimensions must be named 'sample' and 'feature'.
        f_mask : xarray.DataArray, "auto" or None
            An xarray.DataArray of shape (l, ) or (j, l) to be used to weight
            the BMU search. If "auto", determines the correlation between
            missing and available values in the dataset used to train the SOM
            and weights the BMU search based on the correlations. If None, all
            weights are set to None, i.e. equal weights for all inputs.

        Returns
        -------
        best_matching_units : xarray.DataArray
            An xarray.DataArray of shape (j, ) containing the best matching
            units and their coordinates. If labels are available, the labels
            corresponding to best matching units are included as well.
        """
        #---------------------------------------------------------------------#
        return self.som.predict(
            f_data_features = f_data_features,
            f_mask = f_mask
        )
    #=========================================================================#
    def evaluate(
        self,
        f_data_features = None,
        f_data_labels = None,
        f_mask = None,
        f_metric = "accuracy"
    ):
        """
        Estimate the accuracy of the trained SOM grid. Only possible if labels
        are provided during the training process.  If f_mask is provided, the
        distance calculation during the BMU search will be weighted according
        to f_mask. If f_data_features contains missing values, f_mask can be
        set to "auto". In this case, the distance calculation will be weighted
        by the correlation between the features of the data used to train the
        SOM.

        Parameters
        ----------
        f_data_features : xarray.DataArray
            The testing dataset. An xarray.DataArray of shape (j, l) containing
            j samples of the l features.
        f_data_labels : xarray.DataArray
            An xarray.DataArray of shape (l, ) containing the labels
            corresponding to f_data_features.
        f_mask : xarray.DataArray, "auto" or None
            An xarray.DataArray of shape (l, ) or (j, l) to be used to weight
            the BMU search. If "auto", determines the correlation between
            missing and available values in the dataset used to train the SOM
            and weights the BMU search based on the correlations. If None, all
            weights are set to None, i.e. equal weights for all inputs.
        f_metric : str
            The metric to use for estimating the accuracy. Can be one of
            "root_mean_square_error", "mean_absolute_error",
            "median_absolute_deviation" or "accuracy". If "accuracy", estimates
            the fraction of correctly assigned labels.

        Returns
        -------
        metric_value : float
            The value of the specified metric.
        """
        #---------------------------------------------------------------------#
        if "labels" not in self.som.som:
            raise NotImplementedError(
                "ITCOMPSOM object has no evaluate() method "
                "until after calling assign_labels_to_SOM()."
            )
        #---------------------------------------------------------------------#
        return self.som.evaluate(
            f_data_features = f_data_features,
            f_data_labels = f_data_labels,
            f_mask = f_mask,
            f_metric = f_metric
        )
    #=========================================================================#
    def Davies_Bouldin_Index(self):
        """
        Estimate the Davies-Bouldin index. The index is defined as the average
        similarity measure of each cluster with its most similar cluster, where
        similarity is the ratio of within-cluster distances to between-cluster
        distances. Thus, clusters which are farther apart and less dispersed
        will result in a better score.
        The minimum score is 0, with lower values indicating better clustering.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.
        
        Returns
        -------
        db_index : float
            The Davies-Bouldin index.
        """
        #---------------------------------------------------------------------#
        return self.som.Davies_Bouldin_Index(f_data_training = f_data_training)
    #=========================================================================#
    def compute_umatrix(self):
        """
        Compute the U-matrix corresponding to the SOM grid. The U-matrix gives
        the local distance structure at each neuron. It is calculated as the
        sum of the distances between a SOM class and all of its immediate
        neighbours. The U-matrix is normalised by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Returns
        -------
        self.u_matrix : xarray.DataArray
            An xarray.DataArray containing the U-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        return self.som.compute_umatrix()
    #=========================================================================#
    def compute_pmatrix(
        self,
        f_data_training = None
    ):
        """
        Compute the P-matrix corresponding to the SOM grid. The P-matrix gives
        the local data density at each neuron. It is calculated as the number
        of data points within a defined hypersphere around each neuron. The
        P-matrix is normalised by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.

        Returns
        -------
        self.p_matrix : xarray.DataArray
            An xarray.DataArray containing the P-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        return self.som.compute_pmatrix(f_data_training = f_data_training)
    #=========================================================================#
    def compute_ustarmatrix(
        self,
        f_data_training = None
    ):
        """
        Compute the P-matrix corresponding to the SOM grid. The P-matrix gives
        the local data density at each neuron. It is calculated as the number
        of data points within a defined hypersphere around each neuron. The
        P-matrix is normalised by its maximum value.
        See Ultsch & Mörchen (2005) for more details.

        Parameters
        ----------
        f_data_training : None or xr.DataArray
            If not None, an xarray.DataArray containing the data with which the
            SOM was trained.

        Returns
        -------
        self.p_matrix : xarray.DataArray
            An xarray.DataArray containing the P-matrix corresponding to the
            trained SOM grid.
        """
        #---------------------------------------------------------------------#
        return self.som.compute_ustarmatrix(f_data_training = f_data_training)
    #=========================================================================#
    def visualise_SOM(
        self,
        f_figure_width = 10,
        f_feature_dict = {0: 'Spectral_r'}
    ):
        """
        Visualise the weights of the SOM on its native lattice (either
        rectangular or hexagonal).

        Parameters
        ----------
        f_figure_width : float
            The width of the figure in inches. The height is set automatically.
        f_feature_dict : dict
            A dictionary whose keys define the features to plot and the
            corresponding values define the colourmap to use. The key can
            either be a single value or a tuple of three features. If the key
            is a tuple of three features the colourmap will be ignored and the
            weights corresponding to the three features are converted to RGB
            triplets. The length of f_feature_dict defines the total number of
            subplots in the figure.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The matplotlib.figure.Figure instance of the plot.
        axes : dict
            A dictionary containing the matplotlib.axes._axes.Axes instances of
            the individual subplots. The keys are the same as in
            f_feature_dict.
        """
        #---------------------------------------------------------------------#
        return self.som.visualise_SOM(
            f_figure_width = f_figure_width,
            f_feature_dict = f_feature_dict
        )

#%% Function to plot map structures.
def plot_map_structure(
    f_figure = None,
    f_axis = None,
    f_colourbar_params = None,
    f_data = None,
    f_lattice = ("hexa", "rectangle"),
    f_projection = "2d",
    f_show_grid_cell_boundaries = True
):
    """
    Visualise data defined on a rectangular or hexagonal lattice either using a
    planar or a toroidal map structure. In case of a rhombus-shaped lattice,
    the y-axis labels are added as annotations instead of regular y-ticks to
    account for the slanting of the axis.

    Parameters
    ----------
    f_figure : matplotlib.figure.Figure
        A matplotlib.figure.Figure instance in which to plot the SOM grid.
    f_axis : matplotlib.axes._axes.Axes
        The matplotlib.axes._axes.Axes instance of f_figure in which to plot
        the SOM grid.
    f_colourbar_params : dict or None
        Optional parameters to define the colourmap and set the colourbar. If
        provided, possible entries are:
            vmin : float
                The minimium value of the colourbar in data units. The default
                is the global minimum of the dataset.
            vmax : float
                The maximum value of the colourbar in data units. The default
                is the global maximum of the dataset.
            levels : int, float or numpy.array
                The levels of the colourbar. If int, defines the number of
                levels between vmin and vmax, i.e.
                colour_levels = numpy.linspace(vmin, vmax, levels). If float,
                defines the distance between colour levels in data units, i.e.
                colour_levels = numpy.arange(vmin, vmax + levels, levels). If
                numpy.array, defines the colour levels. The default is 21
                levels.
            cmap : str
                The name of the colourmap. The default is "viridis".
            orientation : str
                The orientation of the colourbar, can be one of "horizontal" or
                "vertical". The default is "vertical".
            label : str
                The label of the colourbar.
    f_data : numpy.array
        A 2-dimensional numpy.array containing the data to plot. The columns
        are assumed to be oriented along the x-axis, the rows along the y-axis.
    f_lattice : tuple
        A tuple of str defining the type of grid on which the data is defined.
        The first item defines the type of lattice, either "hexa" or "rect",
        the second item defines the shape of the lattice, either "rectangle",
        "rhombus", "hexagon", or "toroidal". "rhombus" and "hexagon" are only
        valid options for type "hexa" lattice.
    f_projection : str
        The projection of the plot, can be one of "2d" or "3d". 3D plots
        are only available for data defined on a toroidal map.
    f_show_grid_cell_boundaries : bool
        If True, show the outline of the grid cells as black lines.
    
    Returns
    -------
    return_values : dict
        A dictionary containing the following variables:
            ax : matplotlib.axes._axes.Axes
                The matplotlib.axes._axes.Axes instance in which the SOM grid
                is plotted.
            cbar : matplotlib.colorbar.Colorbar
                The matplotlib.colorbar.Colorbar instance of the plot.
            cax : matplotlib.axes._axes.Axes
                The matplotlib.axes._axes.Axes instance in which the colourbar
                is plotted.
    """
    #=========================================================================#
    # Define function to scale coordinates to the range [0, 2pi].
    def coordinate_scaler(
        f_coord_max = 10.,
        f_coord_min = 0.
    ):
        #---------------------------------------------------------------------#
        # Define scaling function.
        coordinate_range = f_coord_max - f_coord_min
        scaler_func = lambda f_coord_values : (
            ((f_coord_values - f_coord_min) * 2. * np.pi)
            / coordinate_range
        )
        #---------------------------------------------------------------------#
        return scaler_func
    #=========================================================================#
    # Define function to calculate torus coordinates.
    def torus_coordinates(
        f_coords_x = None,
        f_coords_y = None,
        f_radius_torus = 10.,
        f_radius_tube = 2.
    ):
        #---------------------------------------------------------------------#
        # Compute torus coordinates.
        torus_X = (
            f_radius_torus + f_radius_tube * np.cos(f_coords_y)
        ) * np.cos(f_coords_x)
        torus_Y = (
            f_radius_torus + f_radius_tube * np.cos(f_coords_y)
        ) * np.sin(f_coords_x)
        torus_Z = f_radius_tube * np.sin(f_coords_y)
        torus_coords = np.column_stack([torus_X, torus_Y, torus_Z])
        #---------------------------------------------------------------------#
        return torus_coords
    #=========================================================================#
    # Check for correct input data.
    if f_lattice[0] not in ("rect", "hexa"):
        raise ValueError(
            "The first item in f_lattice should be one of 'rect' or 'hexa', "
            f"but is '{f_lattice[0]}'."
        )
    if (
        f_lattice[0] == "rect"
        and f_lattice[1] not in ("rectangle", "toroid")
    ):
        raise ValueError(
            "The second item in f_lattice should be one of 'rectangle' or "
            f"'toroid', but is '{f_lattice[1]}'."
        )
    elif (
        f_lattice[0] == "hexa"
        and f_lattice[1] not in ("rectangle", "rhombus", "hexagon", "toroid")
    ):
        raise ValueError(
            "The second item in f_lattice should be one of 'rectangle', "
            f"'rhombus', 'hexagon', or 'toroid', but is '{f_lattice[1]}'."
        )
    if f_lattice[1] in ("rectangle", "rhombus", "hexagon"):
        f_projection = "2d"
    if f_projection not in ("2d", "3d"):
        raise ValueError(
            "f_projection should be one of '2d' or'3d' but is "
            f"'{f_projection}'."
        )
    if len(f_data.shape) != 2:
        raise ValueError(
            "Input data is expected to be 2-dimensional, but is "
            f"{len(f_data.shape)}-dimensional."
        )
    try:
        f_data = f_data.values
    except:
        pass
    finally:
        nrows, ncols = f_data.shape
        f_data = f_data.flatten()
    if f_lattice[1] == "hexagon":
        finite_mask = np.isfinite(f_data)
    else:
        finite_mask = np.ones_like(f_data, dtype = "bool")
        f_data = np.ma.masked_invalid(f_data)
    #=========================================================================#
    # Determine colourbar parameters.
    ifelse = lambda key, other : (
        f_colourbar_params[key] if key in f_colourbar_params.keys()
        else other
    )
    if f_colourbar_params is None:
        f_colourbar_params = {}
    f_colourbar_params.update({
        "orientation" : ifelse("orientation", "vertical")
    })
    f_colourbar_params.update({
        "cmap" : ifelse("cmap", plt.cm.viridis)
    })
    if type(f_colourbar_params["cmap"]) is str:
        f_colourbar_params.update({
            "cmap" : plt.colormaps[f_colourbar_params["cmap"]]
        })
    if "levels" in f_colourbar_params.keys():
        if type(f_colourbar_params["levels"]) is int:
            f_colourbar_params.update({
                "levels" : np.linspace(
                    ifelse("vmin", f_data.min()),
                    ifelse("vmax", f_data.max()),
                    f_colourbar_params["levels"]
                )
            })
        elif type(f_colourbar_params["levels"]) is float:
            f_colourbar_params.update({
                "levels" : np.arange(
                    ifelse("vmin", f_data.min()),
                    ifelse("vmax", f_data.max())+f_colourbar_params["levels"],
                    f_colourbar_params["levels"]
                )
            })
    elif "levels" not in f_colourbar_params.keys():
        f_colourbar_params.update({
            "levels" : np.linspace(
                ifelse("vmin" , f_data.min()),
                ifelse("vmax", f_data.max()),
                21
            )
        })
    if (
        f_colourbar_params["levels"].min() > f_data.min()
        and f_colourbar_params["levels"].max() < f_data.max()
    ):
        f_colourbar_params.update({"extend" : "both"})
    elif (
        f_colourbar_params["levels"].min() > f_data.min()
        and f_colourbar_params["levels"].max() >= f_data.max()
    ):
        f_colourbar_params.update({"extend" : "min"})
    elif (
        f_colourbar_params["levels"].min() <= f_data.min()
        and f_colourbar_params["levels"].max() < f_data.max()
    ):
        f_colourbar_params.update({"extend" : "max"})
    else:
        f_colourbar_params.update({"extend" : "neither"})
    #=========================================================================#
    # Map data to colours.
    norm = mcolors.BoundaryNorm(
        f_colourbar_params["levels"],
        f_colourbar_params["cmap"].N,
        extend = f_colourbar_params["extend"]
    )
    mapper = plt.cm.ScalarMappable(
        norm = norm,
        cmap = f_colourbar_params["cmap"]
    )
    rgba_values = mapper.to_rgba(f_data)
    #=========================================================================#
    # Define polygon parameters.
    if f_lattice[0] == "rect":
        plotting_parameters = {
            "radius" : 1. / np.sqrt(2.),
            "num_vertices" : 4.,
            "orientation" : np.radians(45.)
        }
    elif f_lattice[0] == "hexa":
        plotting_parameters = {
            "radius" : 1. / np.sqrt(3.),
            "num_vertices" : 6.,
            "orientation" : np.radians(120.)
        }
    #=========================================================================#
    # Define plotting coordinates.
    coords_x, coords_y = plotting_coordinates = np.meshgrid(
        np.arange(ncols, dtype = "float32"),
        np.arange(nrows, dtype = "float32")
    )
    if f_lattice[0] == "hexa":
        if f_lattice[1] in ("rectangle", "toroid"):
            coords_x[1::2, :] += 0.5
        elif f_lattice[1] == "rhombus":
            coords_x += np.arange(0., nrows * 0.5, 0.5).reshape(-1, 1)
        elif f_lattice[1] == "hexagon":
            coords_x += np.arange(0., nrows * 0.5, 0.5).reshape(-1, 1)
        coords_y *= (np.sqrt(3.) / 2.)
    coords_x = coords_x.flatten()
    coords_y = coords_y.flatten()
    coords_x = coords_x[finite_mask]
    coords_y = coords_y[finite_mask]
    f_data = f_data[finite_mask]
    #=========================================================================#
    # Define polygons.
    polygons = [
        RegularPolygon(
            (x, y),
            numVertices = plotting_parameters["num_vertices"],
            radius = plotting_parameters["radius"],
            orientation = plotting_parameters["orientation"]
        )
        for x, y in zip(coords_x, coords_y)
    ]
    if f_projection == "3d":
        col_dim = ncols
        if f_lattice[0] == "hexa":
            row_dim = coords_y.max() + np.sqrt(3.) / 2.
        else:
            row_dim = nrows
        scaler_x = coordinate_scaler(f_coord_max = col_dim, f_coord_min = 0.)
        scaler_y = coordinate_scaler(f_coord_max = row_dim, f_coord_min = 0.)
        radius_torus = col_dim / (2. * np.pi)
        radius_tube = row_dim / (2. * np.pi)
        polygons = [
            np.column_stack([
                scaler_x(p.get_verts()[:, 0]),
                scaler_y(p.get_verts()[:, 1])
            ])
            for p in polygons
        ]
        polygons = [
            torus_coordinates(
                f_coords_x = p[:, 0],
                f_coords_y = p[:, 1],
                f_radius_torus = radius_torus,
                f_radius_tube = radius_tube
            )
            for p in polygons
        ]
    #=========================================================================#
    # Plot data.
    if f_projection == "2d":
        f_axis.add_artist(
            PatchCollection(
                patches = polygons,
                facecolors = rgba_values,
                edgecolors = "black",
                linewidth = 1 if f_show_grid_cell_boundaries is True else 0
            )
        )
    elif f_projection == "3d":
        f_axis.add_collection3d(
            Poly3DCollection(
                verts = polygons,
                color = rgba_values,
                edgecolor = "black",
                linewidth = 1 if f_show_grid_cell_boundaries is True else 0
            )
        )
    #=========================================================================#
    # Determine axes parameters.
    if f_projection == "2d":
        ax_ratio = nrows / ncols
        axes_limits = {
            lim : (coord.min(), coord.max())
            for lim, coord in zip(
                ("xlim", "ylim"),
                plotting_coordinates
            )
        }
        factor = 1. if f_lattice[0] == "rect" else 1.1
        axes_limits.update({
            ax : (
                lim[0] - plotting_parameters["radius"] * factor,
                lim[1] + plotting_parameters["radius"] * factor
            )
            for ax, lim in axes_limits.items()
        })
        axes_limits.update({
            "ylim" : (axes_limits["ylim"][1], axes_limits["ylim"][0])
        })
    elif f_projection == "3d":
        ax_ratio = 0.2
        axes_limits = {
            "xlim" : (
                -np.ceil(radius_torus + radius_tube),
                np.ceil(radius_torus + radius_tube)
            ),
            "ylim" : (
                -np.ceil(radius_torus + radius_tube),
                np.ceil(radius_torus+ radius_tube)
            ),
            "zlim" : (-np.ceil(radius_tube), np.ceil(radius_tube))
        }
    #=========================================================================#
    # Adjust axes.
    for ax, lim in axes_limits.items():
        eval(f"f_axis.set_{ax}({lim})")
    if f_projection == "2d":
        if f_lattice[1] == "hexagon":
            f_axis.axis("off")
        else:
            f_axis.tick_params(
                length = 0,
                labelsize = 12,
                labeltop = True,
                labelbottom = False
            )
            for spine in f_axis.spines.values():
                spine.set_visible(False)
            xticks = f_axis.get_xticks()
            xticks = np.array([
                tick for tick in xticks
                if tick >= 0.
                and tick <= ncols
            ])
            xticks = np.linspace(
                np.ceil(0),
                np.ceil(ncols - 1),
                xticks.size,
                dtype = "int"
            )
            f_axis.set_xticks(xticks)
            num_yticks = int(np.ceil(xticks.size * ax_ratio))
            yticks = np.linspace(
                np.ceil(0),
                np.ceil(nrows - 1),
                num_yticks,
                dtype = "int"
            )
            if f_lattice[0] == "hexa":
                if f_lattice[1] == "rhombus":
                    ytick_locs = [
                        (coords_x[np.isclose(coords_y, y)].min(), y)
                        for y in yticks * np.sqrt(3.) / 2.
                    ]
                    for loc, lbl in zip(ytick_locs, yticks):
                        f_axis.annotate(
                            lbl,
                            xy = loc,
                            xycoords = "data",
                            xytext = (-18, -1),
                            textcoords = "offset points",
                            fontsize = 12,
                            horizontalalignment = "right",
                            verticalalignment = "center"
                        )
                    f_axis.set_yticks([])
                else:
                    f_axis.set_yticks(yticks * np.sqrt(3.) / 2.)
                    f_axis.set_yticklabels(yticks.astype("int"))
            elif f_lattice[0] == "rect":
                f_axis.set_yticks(yticks)
            for ax, lim in axes_limits.items():
                eval(f"f_axis.set_{ax}({lim})")
    elif f_projection == "3d":
        f_axis.axis("off")
    f_axis.set_aspect("equal")
    #=========================================================================#
    # Set colourbar.
    axis_position = f_axis.get_position()
    if f_colourbar_params["orientation"] == "vertical":
        if f_projection == "2d":
            cbar_shift = -0.1 if f_lattice[1] == "hexagon" else 0.01
            cbar_axis_position = [
                axis_position.x1 + cbar_shift,
                axis_position.y0,
                0.03,
                axis_position.y1 - axis_position.y0
            ]
        elif f_projection == "3d":
            cbar_height = (axis_position.y1 - axis_position.y0) * 0.5
            cbar_axis_position = [
                axis_position.x1 - 0.08,
                axis_position.y0 + cbar_height / 2.,
                0.03,
                cbar_height
            ]
        colourbar_axis = f_figure.add_axes(cbar_axis_position)
    elif f_colourbar_params["orientation"] == "horizontal":
        if f_projection == "2d":
            if f_lattice[1] == "hexagon":
                cbar_width = (axis_position.x1 - axis_position.x0) * 0.75
                cbar_axis_position = [
                    axis_position.x0 + cbar_width / 6.,
                    axis_position.y0 - 0.04,
                    cbar_width,
                    0.03
                ]
            else:
                cbar_axis_position = [
                    axis_position.x0,
                    axis_position.y0 - 0.04,
                    axis_position.x1 - axis_position.x0,
                    0.03
                ]
        elif f_projection == "3d":
            cbar_width = (axis_position.x1 - axis_position.x0) * 0.75
            cbar_axis_position = [
                axis_position.x0 + cbar_width / 6.,
                axis_position.y0 + 0.08,
                cbar_width,
                0.03
            ]
        colourbar_axis = f_figure.add_axes(cbar_axis_position)
    cbar_ticks = f_colourbar_params["levels"][::int(np.ceil(
        f_colourbar_params["levels"].size * ax_ratio
    ))]
    if cbar_ticks.size <= 2:
        cbar_ticks = [
            f_colourbar_params["levels"].min(),
            np.median(f_colourbar_params["levels"]),
            f_colourbar_params["levels"].max()
        ]
    colourbar = f_figure.colorbar(
        plt.cm.ScalarMappable(
            norm = norm,
            cmap = f_colourbar_params["cmap"]
        ),
        ax = f_axis,
        cax = colourbar_axis,
        ticks = cbar_ticks,
        extend = f_colourbar_params["extend"],
        orientation = f_colourbar_params["orientation"]
    )
    if "label" in f_colourbar_params.keys():
        if f_colourbar_params["orientation"] == "vertical":
            colourbar_axis.set_ylabel(
                f_colourbar_params["label"],
                fontsize = 12
            )
        elif f_colourbar_params["orientation"] == "horizontal":
            colourbar_axis.set_xlabel(
                f_colourbar_params["label"],
                fontsize = 12
            )
    colourbar.ax.minorticks_off()
    colourbar.ax.tick_params(labelsize = 12)
    #=========================================================================#
    # Combine return values in dictionary.
    return_values = {
        "ax" : f_axis,
        "cbar" : colourbar,
        "cax" : colourbar_axis
    }
    #=========================================================================#
    return return_values