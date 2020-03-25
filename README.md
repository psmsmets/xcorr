# xcorr - xarray crosscorrelations -

**xcorr** is an open-source project containing tools to crosscorrelate
waveform timeseries  (`obspy.Stream` objects)  and store all output and
metadata as a self-describing N-D labeled  `xarray.Dataset` following
CF-1.9 Conventions and FAIR data guidelines. Data can be stored as
`netCDF4` and opened/postprocessed in other languages, packages,
machines or platforms.

**xcorr** contains various modules such as waveform preprocessing, a client
waterfall-based wrapping various getters from local archives as well as
remote services, frequency domain crosscorrelationm and postprocessing/
analysis tools.

## Installation

### Clone
Create a clone, or copy of the **xcorr** repository
```shell
$ git clone https://github.com/psmsmets/xcorr.git
```
Run `git pull` to update the local repository to this master repository.

### Install
```shell
$ cd xcorr/
$ pip install -e .
```

## A working Python environment
Obspy, numpy and some other packages tend to cause problems which kills the
`remove_response` function of `obspy`. Create a working (non-conflicting)
Python 3.7 environment in conda as follows:
```shell
conda env create -f xcorr.yml
```

## More info

To be continued.
