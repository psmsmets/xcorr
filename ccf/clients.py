#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import UTCDateTime, Trace, Stream, Inventory
import numpy as np
import xarray as xr
import pandas as pd
import json

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
    
    def check(raise_error=False):
        """
        Verify if the client globals are set.
        """
        error = Clients.sds_root is None or Clients.sds is None or Clients.fdsn is None or Clients.nms is None
        if error and raise_error:
            raise RuntimeError('Clients not yet set! Run Clients.set(..) first!')
        return not error

    def set(sds_root:str, fdsn_base_url:str = 'IRIS', **kwargs):
        """
        Set the client globals `sds_root`. Optionally change the `fdsn_base_url` (default 'IRIS')
        """
        Clients.sds_root = sds_root
        Clients.sds = sdsClient(sds_root)
        Clients.fdsn = fdsnClient(fdsn_base_url, **kwargs)
        Clients.nms = nmsClient()
        
        
    def get_pair_inventory(pair:str,inventory:Inventory):
        """
        Return the filtered inventory given receiver SEED-ids `pair` and `inventory`.
        """
        rA, rB = pair.split('-')
        rA_net, rA_sta, rA_loc, rA_cha = rA.split('.')
        rB_net, rB_sta, rB_loc, rB_cha = rB.split('.')
        return (
            inventory.select(network=rA_net, station=rA_sta, channel=rA_cha) + 
            inventory.select(network=rB_net, station=rB_sta, channel=rB_cha)
        )

    def get_waveforms(receiver:str, time:np.datetime64, centered:bool = True, duration = 86400., buffer = 60., verbose:bool = True ):
        """
        Get waveforms given the SEED-id `receiver` and `time` (default `centered`) for `duration` (default 86400s) and `buffer` (default 60s).
        """
        network, station, location, channel = receiver.split('.')
        Clients.check(raise_error = True)
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
            except Exception as e:
                if verbose:
                    print('an error occurred:')
                    print(e)

        return stream.trim(starttime=t0, endtime=t1)

    def daystream_length_passed(stream:Stream, verbose:bool = False, max_gap = 300.):
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

    def get_preprocessed_pair_stream( pair:str, time:np.datetime64, operations:dict, **kwargs ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-ids receiver `pair`, `time` and preprocess `operations`.
        """
        rA, rB = str(pair).split('-')
        return (
            Clients.get_preprocessed_stream(rA, time, operations, **kwargs) + 
            Clients.get_preprocessed_stream(rB, time, operations, **kwargs)
        )

    def get_preprocessed_stream( receiver:str, time:np.datetime64, operations:dict, duration = 86400., inventory:Inventory = None, operations_from_json:bool = False, verbose:bool = False, debug:bool = False, **kwargs ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-id `receiver`, `time`, preprocess `operations` and `duration` (default 86400s). Optionally provide the `obspy.Inventory` and some other options.
        """
        t0 = toUTCDateTime(time) - duration/2
        t1 = t0 + duration
        ch = receiver.split('.')[-1]
        st = Clients.get_waveforms( receiver, time, duration = duration, centered = True, verbose = debug, **kwargs )
        if verbose:
            print(st)
        if not isinstance(st,Stream) or len(st)==0:
            return Stream()
        try:
            st = Preprocess.preprocess( 
                signal = st, 
                operations = json.loads(operations[ch]) if operations_from_json else operations[ch],
                inventory = inventory, 
                starttime = t0, 
                endtime = t1,
                verbose = verbose
            )
        except:
            return Stream()
        if not isinstance(st,Stream) or len(st)!=1:
            return Stream()
        if st[0].stats.npts * st[0].stats.delta < duration:
            return Stream()
        return st
