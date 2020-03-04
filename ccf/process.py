#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import xarray as xr
import pandas as pd
import json
from obspy import UTCDateTime, Trace, Stream, Inventory
from scipy import signal
from numpy.fft import fftshift, fftfreq
try:
    from pyfftw.interfaces.numpy_fft import fft, ifft
except Exception as e:
    warnings.warn(
        "Could not import fft and ifft from pyfftw. Fallback on numpy's (i)fft. Import error: " + e.strerror, 
        ImportWarning
    )
    from numpy.fft import fft, ifft

from ccf.helpers import Helpers

class CC:
     
    def cc(x:np.ndarray, y:np.ndarray, normalize:bool = True, pad:bool = True, unbiased:bool = True, dtype = np.float32):
        """
        Returns the cross-correlation estimate for vectors `x` and `y`. 
        Cross-correlation is performed in the frequency domain.
        """
        n = len(x)
        assert n == len(y), "Vectors `x` and `y` should have the same length!"
        if pad:
            nn = 2*n-1
            xx = np.zeros(nn,dtype=dtype)
            xx[nn-n:] = x
            yy = np.zeros(nn,dtype=dtype)
            yy[nn-n:] = y
        else:
            xx = x
            yy = y
        fg = fft(xx) * np.conjugate(fft(yy))
        if normalize:
            fg = fg / (np.linalg.norm(xx) * np.linalg.norm(yy))
        Rxy = fftshift(np.real(ifft(fg)))
        return Rxy * CC.weight(nn, False) if unbiased else Rxy

    def lag(n, delta, pad:bool = True):
        """
        Returns an array with lag times given the number of samples `n` and time step `delta`.
        """
        nn = n*2-1 if pad else n
        return fftshift(np.fft.fftfreq(nn, 1/(nn*delta)))
    
    def weight(n, pad:bool = True, clip = None):
        """
        Returns an array with scale factors to obtain the unbaised cross-correction estimate.
        """
        nn = n*2-1 if pad else n
        n = np.int((nn+1)/2)
        w = n / (n - np.abs(np.arange(1-n,nn+1-n,1,np.float64)))
        if clip is not None:
            w[np.where(w > clip)] = clip
        return w

    def extract_shift(cc, delta = None):
        """
        Returns the sample (or time) shift at the maximum of the cross-correlation estimate `Rxy`.
        """
        zero_index = int(len(cc) / 2)
        shift =  np.argmax(cc) - zero_index
        return shift * ( delta or 1 )
    
    def extract_shift_and_max(Rxy, delta = None):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation estimate `Rxy` and the maximum.
        """
        zero_index = int(len(Rxy) / 2)
        index_max = np.argmax(Rxy)
        shift =  index_max - zero_index
        return shift * ( delta or 1 ), Rxy[index_max]
    
    def compute_shift_and_max(x:np.ndarray, y:np.ndarray, delta = None, **kwargs):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation estimate and the maximum.
        """
        Rxy = CC.cc(x, y, **kwargs)
        return CC.extract_shift_and_max(Rxy,delta)

    def compute_shift(x:np.ndarray, y:np.ndarray, delta = None, **kwargs):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation estimate.
        """
        Rxy = CC.cc(x, y, **kwargs)
        return CC.extract_shift_and_max(Rxy,delta)[0]


class Preprocess:
    
    __stream_operations__ = {
        'decimate' : { 'method': 'self', 'inject': []},
        'detrend' : { 'method': 'self', 'inject': []},
        'filter' : { 'method': 'self', 'inject': []},
        'interpolate' : { 'method': 'self', 'inject': []},
        'merge' : { 'method': 'self', 'inject': []},
        'normalize' : { 'method': 'self', 'inject': []},
        'remove_response' : { 'method': 'self', 'inject': ['inventory']},
        'remove_sensitivity': { 'method': 'self', 'inject': ['inventory']},
        'resample' : { 'method': 'self', 'inject': []},
        'rotate' : { 'method': 'self', 'inject': ['inventory']},
        'select' : { 'method': 'self', 'inject': []},
        'taper' : { 'method': 'self', 'inject': []},
        'trim' : { 'method': 'self', 'inject': ['starttime','endtime']},
        'running_rms': { 'method': 'Preprocess.running_rms', 'inject': []},
    }

    def is_stream_operation(operation:str):
        """
        Verify if the operation is a valid stream operation.
        """
        return operation in Preprocess.__stream_operations__

    def stream_operations():
        """
        Returns a list with all valid stream operations.
        """
        return Preprocess.__stream_operations__
    
    def inject_parameters(operation:str, parameters:dict, inventory:Inventory = None, starttime = None, endtime = None):
        """
        Inject starttime, endtime and inventory to the parameters dictionary when needed.
        """
        if 'inventory' in Preprocess.__stream_operations__[operation]['inject']:
            parameters['inventory'] = inventory
        if 'starttime' in Preprocess.__stream_operations__[operation]['inject']:
            parameters['starttime'] = Helpers.to_UTCDateTime(starttime)
        if 'endtime' in Preprocess.__stream_operations__[operation]['inject']:
            parameters['endtime'] = Helpers.to_UTCDateTime(endtime)
        return parameters
    
    def apply_stream_operation(stream, operation:str, parameters:dict, verbose: bool = False):
        """
        Apply a stream operation with the provided parameters.
        """
        if not Preprocess.is_stream_operation(operation):
            return
        method = Preprocess.__stream_operations__[operation]['method']
        if verbose:
            print(operation, ':', parameters)
        if method == 'self':
            return eval(f'stream.{operation}(**parameters)')
        else:
            return eval(f'{method}(stream,**parameters)')

    def preprocess(
        stream, operations:list, inventory:Inventory = None, starttime = None, endtime = None, 
        verbose:bool = False, debug:bool = False
    ):
        """
        Preprocess a `stream` (~obspy.Stream or ~obspy.Trace) given a list of operations.
        Optionally provide the `inventory`, `starttime`, `endtime` to inject in the parameters.
        """
        st = stream.copy()
        for operation_params in operations:
            if not(isinstance(operation_params,tuple) or isinstance(operation_params,list)) or len(operation_params) != 2:
                warnings.warn(
                    'Provided operation should be a tuple or list with length 2 (method:str,params:dict).',
                    UserWarning
                )
                continue
            operation, parameters = operation_params
            if not Preprocess.is_stream_operation(operation):
                warnings.warn(
                    'Provided operation "{}" is invalid thus ignored.'.format(operation),
                    UserWarning
                )
                continue
            try:
                st = Preprocess.apply_stream_operation(
                    stream = st,
                    operation = operation,
                    parameters = Preprocess.inject_parameters(operation,parameters,inventory,starttime,endtime),
                    verbose = verbose,
                )
            except Exception as e:
                warnings.warn(
                    'Failed to execute operation "{}".'.format(operation),
                    RuntimeWarning
                )
                if verbose:
                    print(e)
            if debug:
                print(st)
        return st

    def example_operations():
        return {            
            'BHZ': [
                ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples': 0 }),
                ('filter', {'type':'highpass','freq':.05}),
                ('detrend', { 'type': 'demean' }),
                ('remove_response', {'output': 'VEL'}),
                ('filter', { 'type': 'highpass', 'freq': 3. }),
                ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20 }),
                ('filter', { 'type': 'lowpass', 'freq': 20. }),
                ('trim', {}),
                ('detrend', { 'type': 'demean' }),
                ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
            ],
            'BHR': [
                ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples': 0 }),
                ('filter', {'type':'highpass','freq':.05}),
                ('detrend', { 'type': 'demean' }),
                ('remove_response', {'output': 'VEL'}),
                ('rotate', {'method':'->ZNE'}),
                ('rotate', {'method':'NE->RT', 'back_azimuth':250.30 }), # toward MVC
                ('select', {'channel':'BHR'}),
                ('filter', { 'type': 'highpass', 'freq': 3. }),
                ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20 }),
                ('filter', { 'type': 'lowpass', 'freq': 20. }),
                ('trim', {}),
                ('detrend', { 'type': 'demean' }),
                ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
            ],
            'EDH': [
                ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples': 0 }),
                ('detrend', { 'type': 'demean' }),
                ('remove_sensitivity', {}),
                ('filter', { 'type': 'bandpass', 'freqmin': 3., 'freqmax': 20. }),
                ('decimate', { 'factor': 5 }),
                ('trim', {}),
                ('detrend', { 'type': 'demean' }),
                ('taper', {'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
            ],
        }

    def add_operations_to_DataArray(darray:xr.DataArray, preprocess:dict, attribute:str = 'preprocess'):
        """
        Add the preprocess dict{ channel : [operations] } `DataArray`.
        Optionally change the `attribute` name (default is `preprocess`).
        """
        darray.attrs[attribute] = json.dumps(preprocess)

    def get_operations_from_DataArray(darray:xr.DataArray, channel:str, attribute:str = 'preprocess'):
        """
        Get the list of preprocess operations from the `DataArray` attribute given the `channel`.
        Optionally provide the `attribute` name (default is `preprocess`).
        """
        assert attribute in darray.attrs, 'Attribue "{}" not found in DataArray!'.format(attribute)
        return json.loads(darray.attrs[attribute][channel])
    
    def running_rms(signal, **kwargs):
        """
        Returns a new `obspy.Stream` or `obpsy.Trace` with the running root-mean-square amplitude per `window` (seconds).
        """
        if isinstance(signal,Stream):
            return Preprocess.running_rms_stream(signal, **kwargs)
        else:
            return Preprocess.running_rms_trace(signal, **kwargs)
    
    def running_rms_stream(stream:Stream, **kwargs):
        """
        Returns a new `obspy.Stream` with the running root-mean-square amplitude per `window` (seconds).
        """
        rms_stream = Stream()
        for trace in stream:
            rms_stream += Preprocess.running_rms_trace(trace, **kwargs)
        return rms_stream
    
    def running_rms_trace(trace:Trace, window = 1.):
        """
        Returns a new `obspy.Trace` with the running root-mean-square amplitude per `window` (seconds).
        """
        npts = int(trace.stats.endtime-trace.stats.starttime) / window
        rms_trace = Trace(data=np.zeros(int(npts),dtype=np.float64))
        rms_trace.stats.network = trace.stats.network
        rms_trace.stats.station = trace.stats.station
        rms_trace.stats.location = trace.stats.location
        if window >= 100.:
            band = 'U'
        elif window >= 10:
            band = 'V'
        else:
            band = 'L'
        rms_trace.stats.channel = band + trace.stats.channel[1:]
        rms_trace.stats.delta = window
        rms_trace.stats.starttime = trace.stats.starttime + window/2
        rms_trace.stats.npts = npts

        for index, windowed_trace in enumerate(trace.slide(window_length=window, step=window)):
            rms_trace.data[index] = np.sqrt(np.sum(np.power(windowed_trace.data,2))/windowed_trace.stats.npts)
            
        return rms_trace


class Postprocess:

    dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}

    def leap_year(year, calendar='standard'):
        """
        Determine if year is a leap year
        """
        leap = False
        if ((calendar in ['standard', 'gregorian',
            'proleptic_gregorian', 'julian']) and
            (year % 4 == 0)):
            leap = True
            if ((calendar == 'proleptic_gregorian') and
                (year % 100 == 0) and
                (year % 400 != 0)):
                leap = False
            elif ((calendar in ['standard', 'gregorian']) and
                     (year % 100 == 0) and (year % 400 != 0) and
                     (year < 1583)):
                leap = False
        return leap

    def get_dpm(time, calendar='standard'):
        """
        return an array of days per month corresponding to the months provided in `months`
        """
        month_length = np.zeros(len(time), dtype=np.int)

        cal_days = Postprocess.dpm[calendar]

        for i, (month, year) in enumerate(zip(time.month, time.year)):
            month_length[i] = cal_days[month]
            if Postprocess.leap_year(year, calendar=calendar):
                month_length[i] += 1
        return month_length

    def get_dpy(time, calendar='standard'):
        """
        return an array of days per year corresponding to the years provided in `years`
        """
        year_length = np.zeros(len(time), dtype=np.int)

        for i, year in enumerate(time.year):
            year_length[i] = 365
            if Postprocess.leap_year(year, calendar=calendar):
                year_length[i] += 1
        return year_length
    
    def detrend_cc(dataset:xr.Dataset):
        dims = dataset.cc.dims
        if 'pair' in dims:
            for p in dataset.pair:
                for t in dataset.time:
                    dataset.cc.loc[:,t,p] -= dataset.cc.loc[:,t,p].mean()
        else:
            for t in dataset.time:
                dataset.cc.loc[:,t] -= dataset.cc.loc[:,t].mean()
        return dataset
    
    def list_variables(dataset:xr.Dataset,dim='time'):
        if isinstance(dim,str):
            d = dim
        elif isinstance(dim,xr.DataArray):
            d = dim.name
        else:
            raise TypeError('Only xr.Dataset and str are allowed.')
        var = []
        for v in dataset.data_vars:
            if d in dataset[v].dims:
                var.append(v)
        return var
    
    def lag_window(darray:xr.DataArray, window, scalar=1., **kwargs):
        """
        Return the trimmed lag window of the given DataArray (with dim 'lag').
        """
        assert isinstance(window,list) or isinstance(window,tuple), 'Window should be list or tuple of length 2!'
        assert len(window)==2, 'Window should be list or tuple of length 2!'
        assert window[1]*scalar > window[0]*scalar, 'Window start should be greater than window end!'
        return darray.where( (darray.lag >= window[0]*scalar) & (darray.lag <= window[1]*scalar), drop=True )
    
    def rms(darray:xr.DataArray, dim:str='lag', keep_attrs=True):
        """
        Return the root-mean-square of the DataArray.
        """
        darray = xr.ufuncs.square(darray) # square
        return xr.ufuncs.sqrt(darray.mean(dim=dim,skipna=True,keep_attrs=keep_attrs)) # mean and root
    
    def snr(darray:xr.DataArray, signal_lag_window, noise_percentages = (.2, .8), **kwargs):
        """
        Return the signal-to-noise ratio of the DataArray.
        """
        signal = Postprocess.lag_window(darray, window=signal_lag_window)
        noise = Postprocess.lag_window(darray, window=noise_percentages, scalar=signal_lag_window[0])
        
        snr = Postprocess.rms( signal ) / Postprocess.rms( noise )        
        snr.attrs = {
            'long_name': 'signal-to-noise ratio',
            'standard_name': 'signal_to_noise_ratio',
            'units': '-',
            'signal_lag_window': tuple(signal_lag_window),
            'noise_lag_window': tuple([noise_percentages[0]*signal_lag_window[0], noise_percentages[1]*signal_lag_window[0]]),
            'noise_percentages': tuple(noise_percentages),
        }
        return snr
    
    def stack_dataset(dataset:xr.Dataset, dim:xr.DataArray=None, **kwargs):
        """
        Return the averaged dataset over the coordinate `dim` (default 'time') preserving attributes.
        The first element of the coordinate is re-added to the dataset.
        """
        ds = dataset.mean( dim='time' if dim is None else dim.name, keep_attrs=True, **kwargs )
        return ds.assign_coords( dim or {'time':dataset.time[0]} )
    
    def stack_year_month(dataset:xr.Dataset):
        year_month_idx = pd.MultiIndex.from_arrays([dataset['time.year'], dataset['time.month']])
        dataset.coords['year_month'] = ('time', year_month_idx)
        
        month_length = xr.DataArray(Postprocess.get_dpm(dataset.time.to_index()),coords=[dataset.time], name='month_length')
        weights = month_length.groupby('time.month') / month_length.groupby('time.month').sum()
        return (dataset * weights).groupby('year_month').sum(dim='time',keep_attrs=True) 
    
    def stack_year_dayofyear(dataset:xr.Dataset, **kwargs):
        year_doy_idx = pd.MultiIndex.from_arrays([dataset['time.year'], dataset['time.dayofyear']])
        dataset.coords['year_dayofyear'] = ('time', year_doy_idx)
        return dataset[Postprocess.list_variables(dataset,**kwargs)].groupby('year_dayofyear').mean(dim='time',keep_attrs=True)
        
    def butterworth_filter(darray:xr.DataArray, order:int, btype:str, frequency, **kwargs):
        sos = signal.butter(N=order, Wn=frequency, btype=btype, output='sos', fs=darray.lag.sampling_rate)        
        fun = lambda x, sos: signal.sosfiltfilt(sos, x)
        
        darray_filt = xr.apply_ufunc(fun, darray, sos)
        darray_filt.attrs = {
            **darray.attrs,
            'filtered': np.int8(True),
            'filter_design': 'butterworth',
            'filter_method': 'cascaded second-order sections (sos)',
            'filter_zerophase': np.int8(True),
            'filter_order': order,
            'filter_btype': btype,
            'filter_frequency': frequency,
            
        }
        return darray_filt
    
    def psd(darray:xr.DataArray, duration:float = None, padding:int = None, overlap:float = None, **kwargs):
        
        padding = padding if padding and padding >= 2 else 2
        duration = duration if duration and duration > darray.lag.delta else darray.lag.delta
        overlap = overlap if overlap and (0. < overlap < 1.) else .9
        
        f, t, Sxx = signal.spectrogram(
            x = darray.values,
            fs = darray.lag.sampling_rate,
            nperseg = int(duration * darray.lag.sampling_rate),
            noverlap = int(duration * darray.lag.sampling_rate * overlap),
            nfft = int(padding * duration * darray.lag.sampling_rate),
            scaling = 'density',
            mode = 'psd',
            axis = darray.dims.index('lag'),
            **kwargs
        )
        
        t += Helpers.to_seconds(darray.lag.values[0])
        
        coords = {}
        for dim in darray.dims:
            if dim != 'lag':
                coords[dim] = darray[dim]
        coords['psd_f'] = (
            'psd_f',
            f,
            {'long_name': 'Frequency', 'standard_name': 'frequency', 'units': 'Hz'}
        )
        coords['psd_t'] = (
            'psd_t',
            t, # pd.to_timedelta(t,unit='s'),
            {'long_name': 'Time', 'standard_name': 'time'}
        )        
        
        return xr.DataArray(
            data = Sxx,
            dims = coords.keys(),
            coords = coords,
            name = 'psd',
            attrs = {
                'long_name': 'Power Spectral Density',
                'standard_name': 'power_spectral_density',
                'units': 'Hz**-1',
                'from_variable': darray.name,
                'scaling': 'density',
                'mode': 'psd',
                'overlap': overlap,
                'duration': duration,
                'padding': padding,
                **kwargs
            },
        )