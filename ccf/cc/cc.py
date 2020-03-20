r"""

:mod:`cc.cc` -- Crosscorrelation
==========================================

Crosscorrelation functions for ``ccf``.

"""

# Mandatory imports
import warnings
import numpy as np
from numpy.fft import fftshift, fftfreq
try:
    from pyfftw.interfaces.numpy_fft import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft
    warnings.warn(
        "Could not import fft and ifft from pyfftw. "
        "Fallback on numpy's (i)fft.",
        ImportWarning
    )


__all__ = ['cc', 'lag', 'weight', 'extract_shift', 'extract_shift_and_max',
           'compute_shift', 'compute_shift_and_max']


def cc(
    x: np.ndarray, y: np.ndarray, normalize: bool = True,
    pad: bool = True, unbiased: bool = True, dtype: np.dtype = None
):
    r"""Crosscorrelate two vectors `x` and `y` in the frequency domain.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Data vector x.

    y : :class:`numpy.ndarray`
        Data vector y.

    normalize : bool, optional
        Normalize the crosscorrelation by the norm of `x` and `y` if True
        (default).

    pad : bool, optional
        Pad the length of `x` and `y` to 2*N-1 samples if True (default),
        with N the length of `x` and `y`.

    unbiased : bool, optional
        Correct the crosscorrelation estimate for the reduced number of
        samples if True, else the biased crosscorrelation estimate is
        returned (default).

    dtype : :class:`numpy.dtype`, optional
        Change the dtype of the crosscorrelation estimate `cc`. If None
        (default), `cc` has the samen dtype as `x` and `y`.

    Returns
    -------
    Rxy : :class:`numpy.ndarray`
        Crosscorrelation estimate of `x` and `y`.

    """
    n = len(x)
    dtype = dtype or x.dtype
    assert n == len(y), "Vectors `x` and `y` should have the same length!"
    assert x.dtype == y.dtype, "Vectors `x` and `y` should have the same dtype!"
    if pad:
        nn = 2*n-1
        xx = np.zeros(nn, dtype=dtype)
        xx[nn-n:] = x
        yy = np.zeros(nn, dtype=dtype)
        yy[nn-n:] = y
    else:
        xx = x
        yy = y
    fg = fft(xx) * np.conjugate(fft(yy))
    if normalize:
        fg = fg / (np.linalg.norm(xx) * np.linalg.norm(yy))
    Rxy = fftshift(np.real(ifft(fg)))
    return Rxy * CC.weight(nn, False) if unbiased else Rxy


def lag(
    n: int, delta: float, pad: bool = True
):
    r"""Construct the crosscorrelation lag vector.

    Parameters
    ----------
    n : int
        Length of the crosscorrelation input vectors `x` and `y`.

    delta : float
        Time step (in seconds) of the crosscorrelation input vectors `x`
        and `y`.

    pad : bool, optional
        Pad the length of `x` and `y` to 2*N-1 samples if True (default),
        with N the length of `x` and `y`.

    Returns
    -------
    lag : :class:`numpy.ndarray`
        Crosscorrelation lag vector (in seconds).

    """
    nn = n*2-1 if pad else n
    return fftshift(np.fft.fftfreq(nn, 1/(nn*delta)))


def weight(
    n: int, pad: bool = True, clip: float = None
):
    r"""Construct the unbiased crosscorrelation weight vector.

    Parameters
    ----------
    n : int
        Length of the crosscorrelation input vectors `x` and `y`.

    pad : bool, optional
        Pad the length of `x` and `y` to 2*N-1 samples if True (default),
        with N the length of `x` and `y`.

    clip : float, optional
        Clip the `weight` vector up to a maximum value. If None (default),
        no clipping is applied.

    Returns
    -------
    w : :class:`numpy.ndarray`
        Unbiased crosscorrelation weight vector.

    """
    nn = n*2-1 if pad else n
    n = np.int((nn+1)/2)
    w = n / (n - np.abs(np.arange(1-n, nn+1-n, 1, np.float64)))
    if clip is not None:
        w[np.where(w > clip)] = clip
    return w


def extract_shift(
    Rxy: np.ndarray, delta: float = None
):
    r"""Extract crosscorrelation shift.

    Returns the sample (or time) shift at the maximum of the
    crosscorrelation estimate `Rxy`.

    Parameters
    ----------
    Rxy : :class:`numpy.ndarray`
        Crosscorrelation estimate vector.

    delta : float, optional
        Time step (in seconds) of the crosscorrelation data vectors `x`
        and `y`. If None (default), the integer sample shift is returned.

    Returns
    -------
    shift : int or float
        Sample shift (or lag time if `delta` is provided) at the maximum
        crosscorrelation.

    """
    zero_index = int(len(Rxy) / 2)
    shift = np.argmax(Rxy) - zero_index
    return shift * (delta or 1)


def extract_shift_and_max(
    Rxy: np.ndarray, delta: float = None
):
    r"""Extract crosscorrelation shift.

    Returns the sample (or time) shift at the maximum of the
    crosscorrelation estimate `Rxy` and the maximum itself.


    Parameters
    ----------
    Rxy : :class:`numpy.ndarray`
        Crosscorrelation estimate vector.

    delta : float, optional
        Time step (in seconds) of the crosscorrelation data vectors `x`
        and `y`. If None (default), the integer sample shift is returned.

    Returns
    -------
    shift : int or float
        Sample shift (or lag time if `delta` is provided) at the maximum
        of the crosscorrelation estimate `Rxy`.

    Rxy_max : float
        The maximum of the crosscorrelation estimate `Rxy`.

    """
    zero_index = int(len(Rxy) / 2)
    index_max = np.argmax(Rxy)
    shift = index_max - zero_index
    return shift * (delta or 1), Rxy[index_max]


def compute_shift(
    x: np.ndarray, y: np.ndarray, delta: float = None, **kwargs
):
    r"""Calculate crosscorrelation shift.

    Calculate the sample (or time) shift at the maximum of the
    crosscorrelation estimate `Rxy` of vectors `x` and `y` in the frequency
    domain using :func:`cc`.

    For all input arguments see :func:`cc`.


    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Data vector x.

    y : :class:`numpy.ndarray`
        Data vector y.

    delta : float, optional
        Time step (in seconds) of the crosscorrelation data vectors `x`
        and `y`. If None (default), the integer sample shift is returned.

    Returns
    -------
    shift : int or float
        Sample shift (or lag time if `delta` is provided) at the maximum
        of the crosscorrelation estimate `Rxy`.

    """
    Rxy = CC.cc(x, y, **kwargs)
    return CC.extract_shift_and_max(Rxy, delta)[0]


def compute_shift_and_max(
    x: np.ndarray, y: np.ndarray, delta: float = None, **kwargs
):
    r"""Calculate crosscorrelation shift.

    Calculate the sample (or time) shift at the maximum of the
    crosscorrelation estimate `Rxy` of vectors `x` and `y` in the frequency
    domain using :func:`cc`.

    For all input arguments see :func:`cc`.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Data vector x.

    y : :class:`numpy.ndarray`
        Data vector y.

    delta : float, optional
        Time step (in seconds) of the crosscorrelation data vectors `x`
        and `y`. If None (default), the integer sample shift is returned.

    Returns
    -------
    shift : int or float
        Sample shift (or lag time if `delta` is provided) at the maximum
        of the crosscorrelation estimate `Rxy`.

    Rxy_max : float
        The maximum of the crosscorrelation estimate `Rxy`.

    """
    Rxy = CC.cc(x, y, **kwargs)
    return CC.extract_shift_and_max(Rxy, delta)
