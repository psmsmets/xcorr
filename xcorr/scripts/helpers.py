r"""

:mod:`scripts.helpers` -- Helpers
=================================

Xcorr scripts helper functions.

"""

# Mandatory imports
from pandas import to_datetime
from argparse import ArgumentParser
import distributed
import warnings
import logging
import json
import os

# Relative imports
from ..version import version


__all__ = ['init_logging', 'init_dask', 'ncfile', 'add_common_arguments',
           'add_attrs_group', 'parse_attrs_group']
_global_attrs = ('title', 'institution', 'author', 'source',
                 'references', 'comment')


def init_logging(debug=False):
    """Init the logging format
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=(logging.DEBUG if debug else logging.INFO),
    )


def init_dask(n_workers=None, scheduler_file=None):
    """
    """
    # separate scheduler and worker
    if scheduler_file:
        cluster = None
        client = distributed.Client(scheduler_file=scheduler_file)
    # local cluster
    else:
        cluster = distributed.LocalCluster(
            processes=False,
            threads_per_worker=1,
            n_workers=n_workers,
        )
        client = distributed.Client(cluster)

    # feedback
    print(f".. Dask {client}")

    if n_workers and scheduler_file:
        print(f".. Waiting for {n_workers} workers.", end=' ')
        client.wait_for_workers(n_workers=n_workers)
        print('OK.')

    return cluster, client


def ncfile(title, pair, start, end, prefix=None, suffix=None):
    """Construct netCDF filename.
    """
    start = to_datetime(start)
    end = to_datetime(end)
    if pair in ('*', ''):
        pair = 'all'
    else:
        pair = pair.translate({ord(c): None for c in '*?'})
    nc = "{}{}_{}_{}_{}{}.nc".format(
        f"{prefix}" if prefix else '',
        title, pair, start.strftime('%Y%j'), end.strftime('%Y%j'),
        f"{suffix}" if suffix else '',
    )
    return nc


def add_common_arguments(parser: ArgumentParser, dask: bool = True):
    """Add common arguments to the argument parser object.
    """
    if not isinstance(parser, ArgumentParser):
        raise TypeError('parser should be :class:`ArgumentParser`')

    if dask:
        parser.add_argument(
            '-n', '--nworkers', metavar='..', type=int, default=None,
            help=('Set number of dask workers for the local client. '
                  'If a scheduler is set the client will wait until the '
                  'given number of workers is available.')
        )
        parser.add_argument(
            '--scheduler', metavar='..', type=str, default=None,
            help='Connect to a dask scheduler by a scheduler-file'
        )
    parser.add_argument(
        '--prefix', metavar='..', type=str, default=None,
        help='Set the prefix of the output file (default: None)'
    )
    parser.add_argument(
        '--suffix', metavar='..', type=str, default=None,
        help='Set the suffix of the output file (default: None)'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite if output file exists (default: skip)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots during processing (stalls)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Maximize verbosity'
    )
    parser.add_argument(
        '--version', action='version', version=version,
        help='Print xcorr version and exit'
    )


def add_attrs_group(parser: ArgumentParser):
    """Add attribute arguments to the argument parser object.
    """
    if not isinstance(parser, ArgumentParser):
        raise TypeError('parser should be :class:`ArgumentParser`')

    attrs = parser.add_argument_group(
        title='attribute arguments',
        description=('Set dataset global attributes following COARDS/CF-1.9 '
                     'standards.'),
    )
    attrs.add_argument(
        '--attrs', metavar='..', type=str, default=None,
        help='Set dataset global attributes given a JSON file.'
    )
    for attr in _global_attrs:
        attrs.add_argument(
            f'--{attr}', metavar='..', type=str, default=None,
            help=f'Set dataset {attr} (default: "n/a")'
        )


def parse_attrs_group(args):
    """
    """
    attrs = dict()
    # load from json
    if args.attrs and os.path.isfile(args.attrs):
        try:
            with open(args.attrs) as file:
                for key, value in json.load(file).items():
                    if key in _global_attrs:
                        attrs[key] = value
        except Exception as e:
            warnings.warn(('Failed loading dataset attributes from '
                           'JSON file "{}"\nError: {}').format(args.attrs, e))
    # load from command line arguments
    for attr in _global_attrs:
        arg = eval(f'args.{attr}')
        if arg:
            attrs[attr] = arg
    return attrs
