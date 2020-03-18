# CCF - CrossCorrelation Functions -

**ccf** is an open-source project containing tools to calculate and store
crosscorrelation functions automatically with proper documentation. Results,
parameters and meta data are stored as xarray dataset and stored to disk as a
netCDF4 file, following COARDS and CF-1.7 standards.
It contains pre- and postprocess routines and various clients to retrieve
waveforms, all based on the xarray dataset metadata.

## Installation

### Clone
Create a clone, or copy of the **ccf** repository
```shell
$ git clone https://github.com/psmsmets/ccf.git
```
Run `git pull` to update the local repository to this master repository.

### Install
```shell
$ cd ccf/
$ pip install -e .
```

## A working Python environment
Obspy, numpy and some other packages tend to cause problems which kills the
`remove_response` function of `obspy`. Create a working (non-conflicting)
Python 3.7 environment in conda as follows:
```shell
conda env create -f ccf.yml
```

## More info

To be continued.
