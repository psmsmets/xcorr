r"""

:mod:`scripts.helpers` -- Helpers
=================================

Xcorr scripts helper functions.

"""

# Mandatory imports
import distributed


__all__ = ['init_dask', 'close_dask', 'ncfile']


def init_dask(n_workers: int = None, scheduler_file: str = None):
    """
    """
    if scheduler_file is not None:
        print('... dask scheduler:', scheduler_file)
        client = distributed.Client(scheduler_file=scheduler_file)
        cluster = None
        if n_workers:
            print(f'.. waiting for {n_workers} workers', end=' ')
            client.wait_for_workers(n_workers=n_workers)
            print('OK.')
    else:
        cluster = distributed.LocalCluster(
            processes=False, threads_per_worker=1, n_workers=n_workers or 4,
        )
        print('... dask local cluster:', repr(cluster))
        client = distributed.Client(cluster)
        print('... dask client:', repr(client))
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
    if pair in ('*', ''):
        pair = 'all'
    else:
        pair = pair.translate({ord(c): None for c in '*?'})
    return '{}_{}_{}_{}.nc'.format(
        prefix, pair, start.strftime('%Y%j'), end.strftime('%Y%j'),
    )