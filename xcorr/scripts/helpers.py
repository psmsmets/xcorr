r"""

:mod:`scripts.helpers` -- Helpers
=================================

Xcorr scripts helper functions.

"""

# Mandatory imports
from pandas import to_datetime
import distributed


__all__ = ['init_dask', 'close_dask', 'ncfile']


def init_dask(n_workers: int = None, scheduler_file: str = None):
    """
    """
    if scheduler_file is not None:
        print('.. dask scheduler:', scheduler_file)
        client = distributed.Client(scheduler_file=scheduler_file)
        cluster = None
        if n_workers:
            print(f'.. waiting for {n_workers} workers', end=' ')
            client.wait_for_workers(n_workers=n_workers)
            print('OK.')
        print('{:>20} : {}'.format('scheduler', client.scheduler.addr))
    else:
        cluster = distributed.LocalCluster(
            processes=False, threads_per_worker=1, n_workers=n_workers,
        )
        client = distributed.Client(cluster)
        print('.. dask client:', repr(client))
    return client, cluster


def close_dask(client, cluster):
    """
    """
    print('.. close dask connections')
    client.close()
    if cluster is not None:
        cluster.close()


def ncfile(prefix, pair, start, end):
    """Construct netcdf filename.
    """
    start = to_datetime(start)
    end = to_datetime(end)
    if pair in ('*', ''):
        pair = 'all'
    else:
        pair = pair.translate({ord(c): None for c in '*?'})
    return '{}_{}_{}_{}.nc'.format(
        prefix, pair, start.strftime('%Y%j'), end.strftime('%Y%j'),
    )
