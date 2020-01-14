#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import UTCDateTime, Trace, Stream, Inventory
import numpy as np
import xarray as xr
import pandas as pd
import json

from numpy.fft import fftshift, fftfreq
from pyfftw.interfaces.numpy_fft import fft, ifft

from ccf.core import toUTCDateTime

class CC:
    
    def cc(x, y, normalize:bool = True, pad:bool = True, dtype = np.float32):
        """
        Returns the cross-correlation function for vectors `x` and `y`. 
        Cross-correlation is performed in the frequency domain using the pyfftw library.
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
        return fftshift(np.real(ifft(fg)))

    def lag(n, delta, pad = True):
        """
        Returns an array with lag times given the number of samples `n` and time step `delta`.
        """
        nn = n*2-1 if pad else n
        return fftshift(np.fft.fftfreq(nn, 1/(nn*delta)))

    def extract_shift(ccf, delta = None):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation function `ccf`.
        """
        zero_index = int(len(ccf) / 2)
        shift =  np.argmax(ccf) - zero_index
        return shift * ( delta or 1 )
    
    def extract_shift_and_max(ccf, delta = None):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation function `ccf` and the maximum.
        """
        zero_index = int(len(ccf) / 2)
        index_max = np.argmax(ccf)
        shift =  index_max - zero_index
        return shift * ( delta or 1 ), ccf[index_max]
    
    def compute_shift_and_max(x, y, delta = None, **kwargs):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation function and the maximum.
        """
        c = CC.cc(x, y, **kwargs)
        return CC.extract_shift_and_max(c,delta)

    def compute_shift(x, y, delta = None, **kwargs):
        """
        Returns the sample (or time) shift at the maximum of cross-correlation function.
        """
        c = CC.cc(x, y, **kwargs)
        return CC.extract_shift_and_max(c,delta)[0]


class Preprocess: 
    
    __signal_operations__ = [
        'decimate',
        'detrend',
        'filter',
        'interpolate',
        'merge',
        'normalize',
        'remove_response',
        'remove_sensitivity',
        'resample',
        'taper',
        'trim',
        'running_rms',
        'running_am',
    ]

    def is_signal_operation(operation:str):
        """
        Verify if the operation is a valid signal operation.
        """
        return operation in Preprocess.__signal_operations__

    def signal_operations():
        """
        Returns a list with all valid signal operations.
        """
        return Preprocess.__signal_operations__

    def preprocess(signal, operations:list, inventory:Inventory = None, starttime = None, endtime = None, verbose:bool = False):
        sig = signal.copy()
        """
        Preprocess a `signal` (~obspy.Stream or ~obspy.Trace) given a list of operations.
        Optionally provide the `inventory`, `starttime`, `endtime` or `verbose`.
        """
        for operation_params in operations:
            if not(isinstance(operation_params,tuple) or isinstance(operation_params,list)) or len(operation_params) != 2:
                warn('Provided operation should be a tuple or list with length 2 (method:str,params:dict).')
                continue    
            operation, parameters = operation_params
            if verbose:
                print(operation)
            if not Preprocess.is_signal_operation(operation):
                warn('Provided operation "{}" is invalid thus ignored.'.format(operation))
                continue 
            try:
                if operation == 'decimate':
                    sig.decimate(**parameters)
                elif operation == 'detrend':
                    sig.detrend(**parameters)
                elif operation == 'filter':
                    sig.filter(**parameters)
                elif operation == 'interpolate':
                    sig.interpolate(**parameters)
                elif operation == 'merge':
                    sig.merge(**parameters)
                elif operation == 'normalize':
                    sig.normalize(**parameters)
                elif operation == 'remove_response':
                    sig.remove_response(inventory, **parameters)
                elif operation == 'remove_sensitivity':
                    sig.remove_sensitivity(inventory, **parameters)
                elif operation == 'resample':
                    sig.resample(**parameters)
                elif operation == 'taper':
                    sig.taper(**parameters)
                elif operation == 'trim': 
                    sig.trim(starttime = toUTCDateTime(starttime), endtime = toUTCDateTime(endtime), **parameters)
                elif operation == 'running_rms':
                    sig = Preprocess.running_rms(sig,**parameters)
            except Exception as e:
                warn('Failed to execute operation "{}".'.format(operation))
                if verbose:
                    print(e)
        return sig

    def example_operations():
        return {
            'BHZ': [
                ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
                ('filter', {'type':'highpass','freq':.05}),
                ('detrend', { 'type': 'demean' }),
                ('remove_response', {'output': 'VEL'}),
                ('filter', { 'type': 'highpass', 'freq': 4. }),
                ('trim', {}),
                ('detrend', { 'type': 'demean' }),
                ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20}),
                ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
            ],
            'EDH': [
                ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
                ('detrend', { 'type': 'demean' }),
                ('remove_sensitivity', {}),
                ('filter', { 'type': 'bandpass', 'freqmin': 4., 'freqmax': 10. }),
                ('trim', {}),
                ('detrend', { 'type': 'demean' }),
                ('decimate', { 'factor': 5 }),
                ('taper', {'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
            ],
        }

    def add_operations_to_dataset(dataset:xr.Dataset, channel:str, operations:list, variable:str = 'preprocess'):
        """
        Add the preprocess operations list to the `dataset` given the `channel`.
        Optionally change `variable` name (default is `preprocess`).
        """
        if not variable in dataset.data_vars:
            dataset[variable] = True
        dataset[variable].attrs[channel] = json.dumps(operations)

    def get_operations_from_dataset(dataset:xr.Dataset, channel:str, variable:str = 'preprocess'):
        """
        Get the list of preprocess operations from the `dataset` given the `channel`.
        Optionally provide the `variable` name (default is `preprocess`).
        """
        assert variable in dataset.data_vars, 'Variable "{}" not found in dataset!'.format(variable)
        return json.loads(dataset[variable].attrs[channel])
    
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
    
    def lag_window(da:xr.DataArray, window, scalar=1., **kwargs):
        assert isinstance(window,list) or isinstance(window,tuple), 'Window should be list or tuple of length 2!'
        assert len(window)==2, 'Window should be list or tuple of length 2!'
        assert window[1]*scalar > window[0]*scalar, 'Window start should be greater than window end!'
        return da.where( (da.lag >= window[0]*scalar) & (da.lag <= window[1]*scalar), drop=True )
    
    def rms(da:xr.DataArray,dim:str='lag',keep_attrs=True):
        """
        Return the root-mean-square of the dataarray.
        """
        da = xr.ufuncs.square(da) # square
        return xr.ufuncs.sqrt(da.mean(dim=dim,keep_attrs=keep_attrs)) # mean and root
    
    def snr(da:xr.DataArray, signal_lag_window, noise_percentages = (.2, .8), **kwargs):
        """
        Return the signal-to-noise ratio of the dataarray.
        """
        signal = Postprocess.lag_window(da, window=signal_lag_window)
        noise = Postprocess.lag_window(da, window=noise_percentages, scalar=signal_lag_window[0])
        
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
