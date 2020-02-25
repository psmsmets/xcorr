#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import UTCDateTime, Trace, Stream, Inventory
import numpy as np
import xarray as xr
import pandas as pd
import json
import re

from pyproj import Geod
from obspy import UTCDateTime, Trace, Stream, Inventory
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient
from nms_tools.nms_client import Client as nmsClient
from nms_tools.datafetch import stream2SDS

from ccf.core import toUTCDateTime
from ccf.process import Preprocess

class Clients:
    
    sds_root = None
    sds = None
    fdsn = None
    nms = None
    
    def check(raise_error:bool = False):
        """
        Verify if the client globals are set.
        """
        error = Clients.sds_root is None or Clients.sds is None or Clients.fdsn is None or Clients.nms is None
        if error and raise_error:
            raise RuntimeError('Clients not yet set! Run Clients.set(..) first!')
        return not error
    
    def verify_receiver(receiver:str, allow_wildcards:bool = False, raise_error:bool = False):
        """
        Verify if the receiver string matche the SEED-id regex pattern and optionally verify if
        it contains wildcards (by default allowd).
        """
        if allow_wildcards == False:
            if '*' in receiver or '?' in receiver:
                if raise_error:
                    raise ValueError('Receiver SEED-id cannot contain wildcards (? or *)! Be specific.')
                return False
            if not re.match('^([A-Z]{2})\.([A-Z,0-9]{3,5})\.([0-9]{0,2})\.([A-Z]{2}[0-9,A-Z]{1})', receiver):
                if raise_error:
                    raise ValueError('Receiver SEED-id is not of valid format "network.station.location.channel".')
                return False
        else:
            if not re.match('^([A-Z,?*]{1,2})\.([A-Z,0-9,?*]{1,5})\.([0-9,?*]{0,2})\.([0-9,A-Z,?*]{1,3})', receiver):
                if raise_error:
                    raise ValueError('Receiver SEED-id is not of valid format "network.station.location.channel".')
                return False
        return True

    def set(sds_root:str, fdsn_base_url:str = 'IRIS', **kwargs):
        """
        Set the client globals `sds_root`. Optionally change the `fdsn_base_url` (default 'IRIS')
        """
        Clients.sds_root = sds_root
        Clients.sds = sdsClient(sds_root)
        Clients.fdsn = fdsnClient(fdsn_base_url, **kwargs)
        Clients.nms = nmsClient()
        
    def split_pair(
        pair, separator:str = '-', split_receiver:bool = False,
    ):
        """
        Split a receiver SEED-ids `pair` string.
        """
        if isinstance(pair,xr.DataArray):
            pair = str(pair.values)
        elif isinstance(pair,np.ndarray):
            pair = str(pair)
        assert isinstance(pair,str), "Pair should be either a string, numpy.ndarray or an xarray.DataArray"
            
        return [Clients.split_seed_id(p) for p in pair.split(separator)] if split_receiver else pair.split(separator)
    
    def split_receiver(
        receiver:str
    ):
        """
        Split a receiver SEED-id string.
        """
        return dict(zip(['network','station','location','channel'], receiver.split('.')))
    
    def get_pair_inventory(
        pair, inventory:Inventory
    ):
        """
        Return a filtered inventory given receiver SEED-ids `pair` and `inventory`.
        """
        if isinstance(pair,xr.DataArray):
            pair = str(pair.values)
        assert isinstance(pair,str), "Pair should be either a string or a xarray.DataArray"
        
        r = Clients.split_pair(pair)
        return inventory.select(**r[0]) + inventory.select(**r[1])
    
    def get_pair_distance(
        pair, inventory:Inventory, ellipsoid:str = 'WGS84', poi:dict = None, km:bool = True,
    ):
        """
        Calculate the receiver pair distance. Optionally, specify the ellipsoid (default = WGS84), or specify
        a point-of-interest (a dictionary with longitude and latitude in decimal degrees) to obtain a relative distance.
        """
        g = Geod(ellps=ellipsoid)
        
        r = Clients.split_pair(pair)  
        c = [inventory.get_coordinates(i) for i in r]
        
        if poi:
            az12, az21, d0 = g.inv( 
                poi['longitude'], poi['latitude'], c[0]['longitude'], c[0]['latitude']
            )
            az12, az21, d1 = g.inv( 
                poi['longitude'], poi['latitude'], c[1]['longitude'], c[1]['latitude']
            )
            d = d0 - d1
            
        else:
            az12, az21, d = g.inv( 
                c[0]['longitude'], c[0]['latitude'], c[1]['longitude'], c[1]['latitude']
            )
        return d*1e-3 if km else d

    def get_waveforms(
        receiver:str, time:np.datetime64, centered:bool = True, duration = 86400., buffer = 60., 
        allow_wildcards: bool = False, verbose:bool = False
    ):
        """
        Get waveforms given the SEED-id `receiver` and `time` (default `centered`)
        for `duration` (default 86400s) and `buffer` (default 60s).
        """
        # check if clients are set
        Clients.check(raise_error = True)
                        
        # check if receiver SEED-id is valid
        Clients.verify_receiver(receiver, allow_wildcards = allow_wildcards, raise_error = True)
        
        # split receiver SEED-id
        network, station, location, channel = receiver.split('.')
        
        t0 = UTCDateTime(pd.to_datetime(time)) # center time of 24h window -12h
        if centered:
            t0 -= duration/2
        t1 = t0 + duration
        if buffer > 0.:
            t0 -= buffer
            t1 += buffer
        if verbose:
            print('Get waveforms for ', network, station, location, channel, t0, t1)

        stream = Stream()   
        for day in pd.date_range(UTCDateTime(t0).date, (UTCDateTime(t1)+43200).date, name='days', freq='D'):
            t = UTCDateTime(day) 

            if verbose:
                print('Get Waveform data for {}.'.format(t))

            daystream = Clients.sds.get_waveforms(
                network = network,
                station = station,
                location = location,
                channel = channel,
                starttime = t,
                endtime = t + 86400,
            )
            if Clients.daystream_length_passed(daystream, verbose):
                if verbose:
                    print('Waveform data for {} loaded from archive.'.format(t))
                stream += daystream
                continue
            if verbose:
                print('No waveform data found for day. Try IRIS.')    
            try:
                daystream = Clients.fdsn.get_waveforms(
                    network = network,
                    station = station,
                    location = location,
                    channel = channel,
                    starttime = t,
                    endtime = t + 86400,
                )
                if Clients.daystream_length_passed(daystream, verbose):
                    stream2SDS(daystream,sds_path=Clients.sds_root,force_override=True,verbose=False)
                    if verbose:
                        print('Waveform data for {} downloaded and added to archive.'.format(t))
                    stream += daystream
                    continue
            except KeyboardInterrupt:
                exit()
            except FDSNNoDataException:
                if verbose:
                    print('No waveform data found for day. Try NMS_Client')
            except Exception as e:
                if verbose:
                    print('an error occurred:')
                    print(e)
            try:   
                daystream = Clients.nms.get_waveforms(
                    starttime = t,
                    station = station,
                    channel = channel,
                    verbose = False,
                )
                if Clients.daystream_length_passed(daystream, verbose):
                    stream2SDS(daystream,sds_path=Clients.sds_root,force_override=True,verbose=False)
                    if verbose:
                        print('Waveform data downloaded and added to archive.')
                    stream += daystream
                    continue
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                if verbose:
                    print('an error occurred:')
                    print(e)

        return stream.trim(starttime=t0, endtime=t1)

    def daystream_length_passed(
        stream:Stream, verbose:bool = False, max_gap = 300.
    ):
        """
        Return if a stream (assuming a uniqe SEED-id) contains a day of data not exceeding the allowed `gap` (default 300s.)
        """
        if not isinstance(stream,Stream) or len(stream) == 0:
            return False
        npts_day = int(stream[0].stats.sampling_rate * 86400)
        npts_gap = int(stream[0].stats.sampling_rate * max_gap)
        npts_str = sum([trace.stats.npts for trace in stream])
        if verbose:
            print('Samples in day = {}, samples in stream = {}, max gaps = {}.'.format(npts_day,npts_str, npts_gap))
        return npts_str >= ( npts_day - npts_gap )

    def get_preprocessed_pair_stream(
        pair, time:np.datetime64, operations:dict, **kwargs
    ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-ids receiver `pair`, `time` and preprocess `operations`.
        """
         
        rA, rB = Clients.split_pair(pair)
        
        return (
            Clients.get_preprocessed_stream(rA, time, operations, **kwargs) + 
            Clients.get_preprocessed_stream(rB, time, operations, **kwargs)
        )

    def get_preprocessed_stream( 
        receiver:str, time:np.datetime64, operations:dict, duration = 86400., inventory:Inventory = None,
        operations_from_json:bool = False, three_components:str = '12Z', verbose:bool = False, debug:bool = False, **kwargs 
    ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-id `receiver`, `time`, 
        preprocess `operations` and `duration` (default 86400s).
        Optionally provide the `obspy.Inventory` and some other options.
        """
        # check if receiver SEED-id is valid
        Clients.verify_receiver(receiver, allow_wildcards = False, raise_error = True)
        
        t0 = toUTCDateTime(time) - duration/2
        t1 = t0 + duration
        ch = receiver.split('.')[-1]
        
        # radial or transverse component? Request all Z,1,2 channels manually.
        if ch[-1] == 'R' or ch[-1] == 'T':
            st = Stream()
            for c in three_components:
                st += Clients.get_waveforms( 
                    receiver = receiver[:-1]+c, 
                    time = time, 
                    duration = duration, 
                    centered = True, 
                    verbose = debug, 
                    **kwargs
                )
        else:
            st = Clients.get_waveforms( 
                receiver = receiver,
                time = time,
                duration = duration,
                centered = True,
                verbose = debug,
                **kwargs
            )
        
        if verbose:
            print(st)
        if not isinstance(st,Stream) or len(st)==0:
            return Stream()
        try:
            st = Preprocess.preprocess( 
                stream = st, 
                operations = json.loads(operations[ch]) if operations_from_json else operations[ch],
                inventory = inventory, 
                starttime = t0, 
                endtime = t1,
                verbose = verbose,
                debug = debug,
            )
        except KeyboardInterrupt:
                exit()
        except:
            return Stream()
        if not isinstance(st,Stream) or len(st)!=1:
            return Stream()
        if st[0].stats.npts * st[0].stats.delta < duration:
            return Stream()
        return st
