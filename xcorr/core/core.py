r"""

:mod:`core.core` -- Core
========================

Main functions of xcorr to init, process. bias_correct,
read, write and merge N-D labelled arrays of data.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import obspy
import scipy
import json
import warnings
import os
from glob import glob
try:
    import h5netcdf
except ModuleNotFoundError:
    h5netcdf = False
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..version import version as __version__
from ..client import Client
from ..import cc as correlate
from ..signal.unbias import unbias, get_weights
from .. import util
from ..preprocess import (
    hash_operations,
    check_operations_hash,
    operations_to_dict,
    preprocess_operations_to_dict,
    preprocess_operations_to_json
)


__all__ = ['init', 'read', 'write', 'merge', 'mfread', 'process',
           'bias_correct']


def init(
    pair: str, starttime: pd.Timestamp, endtime: pd.Timestamp,
    preprocess: dict, attrs: dict, sampling_rate: float,
    window_length: float = 86400., window_overlap: float = 0.875,
    clip_lag=None, unbiased_cc: bool = False, closed: str = 'left',
    dtype: np.dtype = np.float32, inventory: obspy.Inventory = None,
    stationary_poi: dict = None, hash_waveforms: bool = False,
):
    """
    Initiate an xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    pair : `str`
        Receiver couple separated by a '-'. Each receiver is specified by its
        SEED-id string: '{network}.{station}.{location}.{channel}'.

    starttime : `pd.Timestamp`
        Start time for generating crosscorrelation time windows.

    endtime : `pd.Timestamp`
        End time for generating crosscorrelation time windows.

    preprocess : `dict`
        Preprocessing operations dictionary, containing a list of operations
        per SEED channel as key. Each list item should be a tuple
        ('operation', {parameters}).
        Use :func:`xcorr.preprocess.help` to list all valid operations and
        their documentation.

    attrs : `dict`
        Dictionary with global attributes the comply with COARDS and CF
        conventions. Following keys are required:
        ['institution', 'author', 'source']

    sampling_rate : `float`
        Sampling rate for the crosscorrelation lag time.

    window_length : `float`, optional
        Crosscorrelation window length, in seconds. Defaults to 86400s.

    window_overlap : `float`, optional
        Crosscorrelation window overlap, [0,1). Defaults to 0.875.

    clip_lag : `float` or `tuple`, optional
        Clip the crosscorrelation lag time. Defaults to `None`. If ``clip_lag``
        is a single value ``lag`` is clipped symmetrically around zero with
        radius ``clip_lag`` (`float` seconds).
        Provide a `tuple` with two elements, (lag_min, lag_max), both of type
        `float` to clip a specific lag range of interest.

    unbiased_cc : `bool`, optional
        Automatically bias correct the crosscorrelation estimate in place if
        `True`. Default is `False`.

    closed : {`None`, 'left', 'right'}, optional
        Make the time interval closed with respect to the given frequency to
        the 'left' (default), 'right', or both sides (`None`).

    dtype : `np.dtype`, optional
        Set the crosscorrelation estimate dtype. Defaults to `np.float32`.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    stationary_poi : `dict`, optional
        Specify a point-of-interest `dict` with keys ['longitude','latitude']
        in decimal degrees to obtain a relative distance.
        If `None` (default), the receiver pair geodetic distance is calculated.

    hash_waveforms : `bool`, optional
        Create a sha256 hash of the preprocessed waveforms used for each
        correlation step. Caution: hashing can take some time (~10s per step).

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The initiated `xcorr` N-D labelled set of data arrays.

    """
    # check
    assert 'institution' in attrs, (
        "attrs['institution'] = 'Institution, department'!"
    )
    assert 'author' in attrs, (
        "attrs['author'] = 'Name - E-mail'!"
    )
    assert 'source' in attrs, (
        "attrs['source'] = 'Data source description'!"
    )
    assert isinstance(pair, str), (
        "pair should be receiver pair string!"
    )
    # config
    delta = 1/sampling_rate
    npts = int(window_length*sampling_rate)
    encoding = {'zlib': True, 'complevel': 9}

    # start dataset
    dataset = xr.Dataset()

    # global attributes
    dataset.attrs = {
        'title': (
            (attrs['title'] if 'title' in attrs else '') +
            ' Crosscorrelations - {}{}'
            .format(
                starttime.strftime('%Y.%j'),
                ' to {}'.format(endtime.strftime('%Y.%j'))
                if starttime.strftime('%Y.%j') != endtime.strftime('%Y.%j')
                else ''
            )
        ).strip(),
        'institution': attrs['institution'],
        'author': attrs['author'],
        'source': attrs['source'],
        'history': 'Created @ {}'.format(pd.to_datetime('now')),
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
        'comment': attrs['comment'] if 'comment' in attrs else 'n/a',
        'Conventions': 'CF-1.9',
        'xcorr_version': __version__,
        'dependencies_version': dependencies_version(as_str=True),
    }

    # pair
    dataset.coords['pair'] = np.array([pair], dtype=object)
    dataset.pair.attrs = {
        'long_name': 'Crosscorrelation receiver pair',
        'standard_name': 'receiver_pair',
        'units': '-',
        'preprocess': hash_operations(preprocess),
    }

    # time
    dataset.coords['time'] = pd.date_range(
        start=starttime,
        end=endtime,
        freq='{0:.0f}s'.format(window_length*(1-window_overlap)),
        closed=closed,
    )
    dataset.time.attrs = {
        'window_length': window_length,
        'window_overlap': window_overlap,
        'closed': closed,
    }

    # lag
    lag = correlate.lag(npts, delta, pad=True)
    if clip_lag is not None:
        msg = ('``clip_lag`` should be in seconds of type `float` or of type'
               '`tuple` with length 2 specifying start and end.')
        if isinstance(clip_lag, float):
            clip_lag = abs(clip_lag)
            clip_lag = tuple(-clip_lag, clip_lag)
        elif isinstance(clip_lag, tuple) and len(clip_lag) == 2:
            if (
                not(isinstance(clip_lag[0], float)) or
                not(isinstance(clip_lag[0], float))
            ):
                raise TypeError(msg)
        else:
            raise TypeError(msg)
        nmin = np.argmin(abs(lag - clip_lag[0]))
        nmax = np.argmin(abs(lag - clip_lag[1]))
    else:
        nmin = 0
        nmax = 2*npts-1
    dataset.coords['lag'] = lag[nmin:nmax]
    dataset.lag.attrs = {
        'long_name': 'Lag time',
        'standard_name': 'lag_time',
        'units': 's',
        'sampling_rate': float(sampling_rate),
        'delta': float(delta),
        'npts': int(npts),
        'clip_lag': (
            np.array(clip_lag if clip_lag is not None else [])
        ),
        'index_min': int(nmin),
        'index_max': int(nmax),
    }

    # pair distance
    dataset['distance'] = (
        ('pair'),
        np.ones((1), dtype=np.float64) * util.receiver.get_pair_distance(
            pair=pair,
            inventory=inventory,
            poi=stationary_poi,
            ellipsoid='WGS84',
            km=True,
        ),
        {
            'long_name': 'receiver pair distance',
            'standard_name': 'receiver_pair_distance',
            'units': 'km',
            'description': (
                ('relative to poi' if stationary_poi else 'absolute') +
                ' WGS84 geodetic distance'
            ),
            'relative_to_poi': (
                json.dumps(stationary_poi) if stationary_poi else 'n/a'
            ),
        },
    )

    # status
    dataset['status'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.byte),
        {
            'long_name': 'processing status',
            'standard_name': 'processing_status',
            'units': '-',
            'valid_range': np.byte([-1, 1]),
            'flag_values': np.byte([-1, 0, 1]),
            'flag_meanings': 'missing_data not_processed processed',
        },
    )

    # hash waveforms
    if hash_waveforms:
        dataset['hash'] = (
            ('pair', 'time'),
            np.array([['n/a']*len(dataset.time)], dtype=object),
            {
                'long_name': 'pair preprocessed stream hash',
                'standard_name': 'pair_preprocessed_stream_hash',
                'units': '-',
                'description': (
                    "Openssl SHA256 hash of the pair preprocessed waveform "
                    "stream. Be aware that stream/pair order matters! "
                    "The hash is updated per `obspy.Trace` and includes the "
                    "stats with keys=['network', 'station', 'location', "
                    "'channel', 'starttime', 'endtime', 'sampling_rate', "
                    "'delta', 'npts'], sorted and dumped to json with 4 "
                    "character space indentation and separators ',' and ':', "
                    "followed by the hash of each sample byte representation."
                ),
            },
        )

    # pair offset
    dataset['pair_offset'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.timedelta64),
        {
            'long_name': 'receiver pair start sample offset',
            'standard_name': 'receiver_pair_start_sample_offset',
            'description': (
                'offset = receiver[0].starttime - receiver[1].starttime'
            ),
        },
    )

    # time offset
    dataset['time_offset'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.timedelta64),
        {
            'long_name': 'first receiver start sample offset',
            'standard_name': 'first_receiver_start_sample_offset',
            'description': (
                'offset = receiver[0].starttime - time + window_length/2'
            ),
        },
    )

    # cc
    dataset['cc'] = (
        ('pair', 'time', 'lag'),
        np.zeros((1, len(dataset.time), len(dataset.lag)), dtype=dtype),
        {
            'long_name': 'Crosscorrelation Estimate',
            'standard_name': 'crosscorrelation_estimate',
            'units': '-',
            'add_offset': dtype(0.),
            'scale_factor': dtype(1.),
            'valid_range': dtype([-1., 1.]),
            'normalize': np.byte(1),
            'bias_correct': np.byte(unbiased_cc),
            'unbiased': np.byte(0),
        },
        encoding
    )

    if unbiased_cc:
        dataset['w'] = get_weights(dataset.lag)

    # add metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )

    return dataset


def read(
    path: str, extract: bool = False, engine: str = None, verb: int = 0,
    **kwargs
):
    """
    Open an xcorr N-D labelled set of data arrays from a netCDF4 file.

    Parameters
    ----------
    path : `str`
        NetCDF4 filename.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    engine : `str`
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to the :func:`validate`.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays read from netCDF4.

    """
    # open if exists
    if not os.path.isfile(path):
        return None

    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # open and validate
    dataset = validate(xr.open_dataset(path, engine=engine, autoclose=True),
                       verb=verb, **kwargs)

    # verbose status
    if verb > 0:
        print('{s} #(status==1): {n} of {m}'.format(
            s=dataset.encoding['source'],
            n=np.sum(dataset.status.data == 1),
            m=dataset.time.size,
        ))

    # extract valid data
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    return dataset


def mfread(
    paths, extract: bool = False, engine: str = None, parallel: bool = True,
    chunks=None, naive: bool = False, **kwargs
):
    """
    Open multiple xcorr N-D labelled files as a single dataset using
    :func:`xarray.open_mfdataset`.

    Parameters
    ----------
    paths : `str` or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open. Paths can be given as strings or as pathlib
        Paths.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    parallel : `bool`, optional
        Enabled parallellization if `True` (defaults). Requires Dask.

    chunks : `int` or `dict` , optional
        Dictionary with keys given by dimension names and values given by
        chunk sizes. In general, these should divide the dimensions of each
        dataset. If int, chunk each dimension by chunks. By default, chunks
        will be chosen to load entire input files into memory at once. This
        has a major impact on performance: see :func:`xarray.open_mfdataset`
        for more details.

    naive : `bool`, optional
        If `True`, ``paths`` is directly passed on to
        :func:`xarray.open_mfdataset` disabling :func:`validate_list`.
        Defaults to `False`.

    Any additional keyword arguments will be passed to the :func:`validate`.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays read from netCDF4.

    """
    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # get a list of validated datasets
    if naive:
        validated = paths
    else:
        validated = validate_list(paths, keep_opened=False, paths_only=True,
                                  engine=engine, **kwargs)

    # init chunks
    chunks = chunks or {'pair': 1, 'time': 1}

    # validate wrapper to pass arguments
    def _validate(ds):
        return validate(ds, quick_and_dirty=True)

    # open multiple files using dask
    dataset = xr.open_mfdataset(
        paths=validated,
        chunks=chunks,
        combine='by_coords',
        preprocess=_validate,
        engine=engine,
        lock=False,
        data_vars='minimal',
        parallel=parallel,
        join='outer',
        autoclose=True,
    )

    # extract valid data
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    # update some global attrs
    del dataset.attrs['sha256_hash']

    strt0 = pd.to_datetime(dataset.time.values[0]).strftime('%Y.%j')
    strt1 = pd.to_datetime(dataset.time.values[-1]).strftime('%Y.%j')
    dataset.attrs['title'] = (
        (dataset.attrs['title']).split(' - ')[0] +
        ' - ' +
        strt0 +
        ' to {}'.format(strt1) if strt0 != strt1 else ''
    ).strip()

    dataset.attrs['history'] = 'Merged @ {}'.format(pd.to_datetime('now'))

    return dataset


def validate(
    dataset: xr.Dataset, fast: bool = True, quick_and_dirty: bool = False,
    metadata_hash: str = None, preprocess_hash: str = None,
    xcorr_version: str = None, verb: int = 0
):
    """
    Read an xcorr N-D labelled data array from a netCDF4 file.

    Parameters
    ----------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays to be validated.

    fast : `bool`, optional
        Omit verifying the `sha256_hash` if `True`. Default is `False`.

    quick_and_dirty : `bool`, optional
        Omit verifying both the `sha256_hash` and `sha256_hash_metadata`
        if `True`. Default is `False`.

    metadata_hash : `str`, optional
        Provide a template metadata sha256 hash to filter data arrays.
        A mismatch in ``dataset.sha256_hash_metadata`` will return `None`.
        Defaults to `None`.

    preprocess_hash : `str`, optional
        Provide a template pair preprocess sha256 hash to filter data arrays.
        A mismatch in ``dataset.pair.attrs['preprocess']['sha256_hash']`` will
        return `None`.
        Defaults to `None`.

    xcorr_version : `str`, optional
        Provide a template xcorr version number to filter data arrays.
        A mismatch in ``dataset.xcorr_version`` will return `None`.
        Defaults to `None`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    validated : :class:`xarray.Dataset`
        The validated `xcorr` N-D labelled set of data arrays.

    """

    # cannot be empty
    if not isinstance(dataset, xr.Dataset):
        return None

    # check existance of main attributes
    if (
        'xcorr_version' not in dataset.attrs or
        'sha256_hash_metadata' not in dataset.attrs or
        'sha256_hash' not in dataset.attrs
    ):
        dataset.close()
        return None

    # fix single-element float/integers represented as np.arrays (h5netcdf)
    for var in dataset.variables:
        for attr in dataset[var].attrs.keys():
            if (
                isinstance(dataset[var].attrs[attr], np.ndarray) and
                len(dataset[var].attrs[attr]) == 1
            ):
                dataset[var].attrs[attr] = dataset[var].attrs[attr].item()

    # extract source
    src = (dataset.encoding['source'] if 'source' in dataset.encoding
           else '[memory]')

    # Verify metadata_hash input
    if metadata_hash:
        if not isinstance(metadata_hash, str):
            dataset.close()
            raise TypeError('``metadata_hash`` should be a string.')
        if not len(metadata_hash) == 64:
            dataset.close()
            raise ValueError('``metadata_hash`` should be of length 64.')

    # check if at least pair and time variables exist
    if not ('pair' in dataset.coords and 'time' in dataset.coords):

        if verb > 0:
            warnings.warn('Dataset contains no pair and time coordinate.',
                          UserWarning)
        dataset.close()
        return None

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)

    # calculate some hashes
    if not quick_and_dirty:

        sha256_hash_metadata = util.hasher.hash_Dataset(
            dataset, metadata_only=True, debug=False
        )

        if sha256_hash_metadata != dataset.sha256_hash_metadata:
            if verb > 0:
                warnings.warn(
                    f'Dataset metadata sha256 hash in {src} is inconsistent.',
                    UserWarning
                )
            if verb > 1:
                print('source :', src)
                print(
                    'sha256 hash metadata in ncfile :',
                    dataset.sha256_hash_metadata
                )
                print(
                    'sha256 hash metadata computed  :',
                    sha256_hash_metadata
                )
            dataset.close()
            return None

    if not (quick_and_dirty or fast):
        sha256_hash = util.hasher.hash_Dataset(
            dataset, metadata_only=False, debug=False
        )
        if sha256_hash != dataset.sha256_hash:
            if verb > 0:
                warnings.warn(f'Dataset sha256 hash in {src} is inconsistent.',
                              UserWarning)
            if verb > 1:
                print('source :', src)
                print('sha256 hash in ncfile :', dataset.sha256_hash)
                print('sha256 hash computed  :', sha256_hash)
            dataset.close()
            return None

    # compare metadata_hash with template
    if metadata_hash and dataset.sha256_hash_metadata != metadata_hash:
        if verb > 0:
            warnings.warn('Dataset metadata hash does not match.',
                          UserWarning)
        dataset.close()
        return None

    # compare preprocess_hash with template
    if (
        preprocess_hash and
        dataset.pair.attrs['preprocess']['sha256_hash'] != preprocess_hash
    ):
        if verb > 0:
            warnings.warn('Dataset preprocess hash does not match.',
                          UserWarning)
        dataset.close()
        return None

    # compare xcorr_version with template
    if xcorr_version and dataset.xcorr_version != xcorr_version:
        if verb > 0:
            warnings.warn('Dataset xcorr version does not match.',
                          UserWarning)
        dataset.close()
        return None

    return dataset


def validate_list(
    datasets, strict: bool = False, paths_only: bool = False,
    keep_opened: bool = False, engine: str = None,
    parallel: bool = True, compute: bool = True, compute_args: dict = {},
    verb: int = 0, **kwargs
):
    """
    Validate a list of xcorr N-D labelled datasets.

    Parameters
    ----------
    datasets : `str` or `list`
        A glob string, or a list of either glob strings or a
        :class:`xr.Dataset` containing the `xcorr` N-D labelled data arrays.

    strict : `bool`, optional
        If `True`, do not merge data arrays with different `xcorr` versions.
        Defaults to `False`.

    paths_only : `bool`, optional
        If `True`, ``datasets`` can only be glob strings. Defaults to `False`.

    keep_opened : `bool`, optional
        If `True`, do not close the file after opening. Defaults to `False`.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    parallel : `bool`, optional
        Enabled parallellization if `True` (default). Requires Dask.

    compute : `bool`, optional
        Execute Dask compute if `True` (default) and ``parallel`` is enabled.
        Otherwise the delayed object is returned.

    compute_args : `dict`, optional
        Provide a dictionary with arguments for Dask compute.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to :func:`validate`.

    Returns
    -------
    validated : :class:`xarray.Dataset`
        The validated list of `xcorr` N-D labelled sets of data arrays.
        If ``datasets`` was a glob string the expanded file list is returned.

    """

    valErr = ('Datasets should either be a list of paths or xarray.Dataset '
              '(when paths_only is False) but not both.')

    if isinstance(datasets, str):
        datasets = [datasets]

    assert isinstance(datasets, list), (
        'datasets should be either a string or a list.'
    )

    parallel = dask and parallel

    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # expand path list with glob
    sources = []

    # expand glob strings and check for unique type in list
    for source in datasets:
        if isinstance(source, str):
            sources += glob(source)
        elif isinstance(source, xr.Dataset):
            if len(sources) > 0 or paths_only:
                raise ValueError(valErr)
        else:
            raise ValueError(valErr)

    isFile = len(sources) > 0
    sources = sorted(sources) if isFile else datasets
    validated = []

    # get dataset wrapper
    def get_dataset(source):
        if not isFile:
            return source
        if not os.path.isfile(source):
            if verb > 0:
                warnings.warn(f'Datasets item "{source}" does not exists.',
                              UserWarning)
            return None
        return xr.open_dataset(source, engine=engine, autoclose=True)

    # get output wrapper
    def get_output(ds):
        if not isinstance(ds, xr.Dataset):
            return None
        if isFile and not keep_opened:
            ds.close()
            return ds.encoding['source']
        else:
            return ds

    # find first validated dataset
    for i, source in enumerate(sources):
        ds = get_dataset(source)
        if ds is None:
            continue
        ds = validate(ds, verb=verb, **kwargs)
        if ds is not None:
            break

    # any valid dataset?
    if ds is None:
        raise RuntimeError('No valid dataset found.')

    # append first valid ds
    validated.append(get_output(ds))

    # set validate args based on first valid ds
    validate_args = {
        'metadata_hash': ds.attrs['sha256_hash_metadata'],
        'preprocess_hash': ds.pair.attrs['preprocess']['sha256_hash'],
        'xcorr_version': ds.attrs['xcorr_version'] if strict else None,
    }

    if parallel:
        for source in sources[i+1:]:
            ds = dask.delayed(get_dataset)(source)
            ds = dask.delayed(validate)(ds, **validate_args, **kwargs)
            validated.append(dask.delayed(get_output)(ds))
        if compute:
            validated = dask.compute(validated, **compute_args)
    else:
        for source in sources[i+1:]:
            ds = get_dataset(source)
            ds = validate(ds, **validate_args, **kwargs)
            validated.append(get_output(ds))

    return validated


def write(
    data, path: str, close: bool = True,
    force_write: bool = False, verb: int = 1,
    **kwargs
):
    """
    Write an xcorr N-D labelled data array to a netCDF4 file using a
    temporary file and replacing the final destination.

    Before writing the data, metadata and data hash hashes are verified and
    updated if necessary. This changes the data attributes in place.

    The preferred engine is set to h5netcdf if the module can be found.

    Parameters
    ----------
    data : :class:`xarray.Dataset` or :class:`xarray.DataArray`
        The `xcorr` N-D labelled data array or set of arrays.

    path : `str`
        The netCDF4 filename.

    close : `bool`, optional
        Close the data `True` (default).

    force_write : `bool`, optional
        Always write file if `True` even if its empty. Default is `False`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    Any additional keyword arguments will be passed to
    :func:`xarray.to_netcdf`.

    """
    isdataset = isinstance(data, xr.Dataset)

    if not (isdataset or isinstance(data, xr.DataArray)):
        raise TypeError('data should be an xarray DataArray or Dataset.')

    # metadata hash
    metadata_hash = util.hasher.hash(data, metadata_only=True)

    if 'sha256_hash_metadata' not in data.attrs:
        data.attrs['sha256_hash_metadata'] = metadata_hash

    if metadata_hash != data.attrs['sha256_hash_metadata']:
        warnings.warn(
            'Data metadata sha256 hash is updated.',
            UserWarning
        )

    # check status?
    if isdataset and 'status' in data.variables:
        if (
            np.sum(data.status.data == 1) == 0 or
            (np.sum(data.status.values == -1) == 0 and force_write)
        ):
            warnings.warn(
                'Dataset contains no data. No need to save it.',
                UserWarning
            )
            return

    # start verbose
    if verb > 0:
        print('Write data as "{}"'.format(path), end=': ')

    # folders and tmp name
    abspath, file = os.path.split(os.path.abspath(path))

    if not os.path.exists(abspath):
        os.makedirs(abspath)

    tmp = os.path.join(
        abspath,
        '{f}.{t}'.format(f=file, t=int(pd.to_datetime('now').timestamp()*1e3))
    )

    # close data
    if close:
        if verb > 0:
            print('Close', end='. ')
        data.close()

    # calculate data hash
    if verb > 0:
        print('Hash', end='. ')
    data.attrs['sha256_hash'] = util.hasher.hash(data, metadata_only=False)

    # convert preprocess operations
    if verb > 1:
        print('Operations to json.', end='. ')

    if 'pair' in data.dims:
        preprocess_operations_to_json(data.pair)

    # write to temporary file
    if verb > 0:
        print('To temporary netcdf', end='. ')
    data.to_netcdf(
        path=tmp, mode='w', format='netcdf4',
        **{'engine': 'h5netcdf' if h5netcdf else None, **kwargs}
    )

    # Replace file
    if verb > 0:
        print('Replace', end='. ')
    os.replace(tmp, os.path.join(abspath, file))

    # convert preprocess operations
    if verb > 1:
        print('Operations to dict.', end='. ')

    if 'pair' in data.dims:
        preprocess_operations_to_dict(data.pair)

    # final message
    if verb > 0:
        print('Done.')

    return


def merge(
    datasets: list, extract: bool = True, verb: int = 0, **kwargs
):
    """
    Merge a list of xcorr N-D labelled data arrays.

    Parameters
    ----------
    datasets : `list`
        A list with either a `str` specifying the path as a glob string or a
        :class:`xarray.Dataset` containing the `xcorr` N-D labelled data array.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to :func:`validate_list`.

    Returns
    -------
    datasets : :class:`xarray.Dataset`
        The merged `xcorr` N-D labelled set of data array.

    """

    validated = validate_list(datasets, verb=verb, keep_opened=True,
                              parallel=False, **kwargs)

    dsets = xr.combine_by_coords(validated, data_vars='minimal', join='outer')

    # extract valid data
    if extract:
        dsets['cc'] = dsets.cc.where(dsets.status == 1)

    # update some global attrs
    dsets.attrs = validated[0].attrs
    del dsets.attrs['sha256_hash']

    strt0 = pd.to_datetime(dsets.time.values[0]).strftime('%Y.%j')
    strt1 = pd.to_datetime(dsets.time.values[-1]).strftime('%Y.%j')
    dsets.attrs['title'] = (
        (dsets.attrs['title']).split(' - ')[0] +
        ' - ' +
        strt0 +
        ' to {}'.format(strt1) if strt0 != strt1 else ''
    ).strip()

    dsets.attrs['history'] = 'Merged @ {}'.format(pd.to_datetime('now'))

    return dsets


def process(
    dataset: xr.Dataset, client: Client, inventory: obspy.Inventory = None,
    retry_missing: bool = False, test_run: bool = False,
    hash_waveforms: bool = True, metadata_hash: str = None,
    verb: int = 1, **kwargs
):
    """
    Process the xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays to process.

    client : :class:`xcorr.Client`
        The initiated client to the local and remote data archives.

    inventory : :class:`obspy.Inventory`
        The inventory object with instrument responses.

    retry_missing : `bool`, optional
        If `True`, ``x.status`` with flag "-1" are reprocessed, otherwise only
        flag "0". Defaults is `False`.

    test_run : `bool`, optional
        If `True` the function is aborted after the first iteration.
        Default is `False`.

    hash_waveforms : `bool`, optional
        Compute the sha256 hash of the preprocessed waveforms used for each
        correlation step if `True` (default), and ``x`` contains the variable
        ``hash``. Caution: hashing can take some time (~10s per step).

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    **kwargs :
        Arguments passed to :meth:``client.get_pair_preprocessed_waveforms``.

    """
    dataset.attrs['history'] += (
        ', Process started @ {}'.format(pd.to_datetime('now'))
    )
    # extract and validate preprocess operations
    if isinstance(dataset.pair.preprocess, dict):

        o = dataset.pair.preprocess
        check_operations_hash(o, raise_error=True)

    else:

        o = operations_to_dict(dataset.pair.preprocess)

    # check lag indices and update if necessary
    util.time.update_lag_indices(dataset.lag)

    # hash?
    hash_waveforms = hash_waveforms and 'hash' in dataset.variables

    # process each pair per time step
    for p in dataset.pair:

        for t in dataset.time:

            # set location
            pt = {'pair': p, 'time': t}

            if verb > 0:
                print(str(p.values), str(t.values)[:19], end=': ')

            # skip processed
            if dataset.status.loc[pt].values != 0:

                if not (
                    retry_missing and
                    dataset.status.loc[pt].values == -1
                ):

                    if verb > 0:
                        print('Has status "{}". Skip.'.format(
                              dataset.status.loc[pt].values))

                    continue

            # waveforms
            if verb > 0:
                print('Waveforms', end='. ')

            st = client.get_pair_preprocessed_waveforms(
                pair=p.values,
                time=t.values,
                preprocess=o,
                duration=t.window_length,
                buffer=t.window_length/20,  # 5%
                inventory=inventory,
                verb=verb-1,
                strict=True,
                **kwargs
            )

            if not isinstance(st, obspy.Stream) or len(st) != 2:

                if verb > 0:
                    print('Missing data. Set status "-1" and skip.')

                dataset.status.loc[pt] = -1

                if test_run:
                    break

                continue

            # track timing offsets
            dataset.pair_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) -
                pd.to_datetime(st[1].stats.starttime.datetime)
            )
            dataset.time_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) +
                pd.to_timedelta(dataset.time.window_length / 2, unit='s') -
                dataset.time.loc[{'time': t}].values
            )

            # hash
            if hash_waveforms:

                if verb > 0:
                    print('Hash', end='. ')

                dataset.hash.loc[pt] = util.hash_Stream(st)

            # cc
            if verb > 0:
                print('CC', end='. ')

            dataset.cc.loc[pt] = correlate.cc(
                x=st[0].data[:dataset.lag.attrs['npts']],
                y=st[1].data[:dataset.lag.attrs['npts']],
                normalize=dataset.cc.attrs['normalize'] == 1,
                pad=True,
                unbiased=False,  # apply correction for full dataset!
                dtype=dataset.cc.dtype,
            )[dataset.lag.attrs['index_min']:dataset.lag.attrs['index_max']]

            # Status
            dataset.status.loc[pt] = 1

            # Finish
            if verb > 0:
                print('Done.')

            if test_run:
                break

    # update history
    dataset.attrs['history'] += (
        ', Process ended @ {}'.format(pd.to_datetime('now'))
    )

    # bias correct?
    if dataset.cc.attrs['bias_correct'] == 1:
        dataset = bias_correct(dataset)

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )


def bias_correct(
    dataset: xr.Dataset, biased_var: str = 'cc', unbiased_var: str = None,
    weight_var: str = 'w'
):
    """
    Bias correct the xcorr N-D labelled data array.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        The data array to process.

    biased_var : `str`, optional
        The name of the biased correlation variable. Defaults to 'cc'.

    unbiased_var : `str`, optional
        The name of the biased correlation variable. If `None`,
        `unbiased_var` = `biased_var`. Defaults to `None`.

    weight_var: `str`, optional
        The name of unbiased correlation weight variable. Defaults to 'w'.

    """
    if dataset[biased_var].unbiased != 0:
        print('No need to bias correct again.')
        return
    unbiased_var = unbiased_var or biased_var

    if weight_var not in dataset.data_vars:
        dataset[weight_var] = get_weights(dataset.lag, name=weight_var)

    dataset[unbiased_var] = unbias(
        x=dataset[biased_var],
        w=dataset[weight_var],
        name=unbiased_var
    )

    # update dataset history
    dataset.attrs['history'] += (
        ', Bias corrected CC @ {}'.format(pd.to_datetime('now'))
    )

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )


def dependencies_version(as_str: bool = False):
    """Returns a `dict` with core dependencies and its version.
    """
    versions = {
        json.__name__: json.__version__,
        np.__name__: np.__version__,
        obspy.__name__: obspy.__version__,
        pd.__name__: pd.__version__,
        scipy.__name__: scipy.__version__,
        xr.__name__: xr.__version__,
    }
    if as_str:
        return ', '.join(['-'.join(item) for item in versions.items()])
    else:
        return versions
