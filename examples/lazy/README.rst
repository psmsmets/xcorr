Lazy examples
=============

Lazy examples to verify client waveform availability, channel based stream
processing and cross-correlation processing.

Make sure that no subroutine (of ObsPy, SciPy, NumPy, ...) runs uncontrolled
multithreaded.

.. code-block:: console

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python lazy_availability.py
