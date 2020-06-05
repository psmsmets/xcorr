Lazy process
============

Lazy examples to verify client data availability, per receiver preprocessing
methods, and crosscorrelation processing.


Make sure that no subroutine (of ObsPy, SciPy, NumPy, ...) runs uncontrolled
multithreaded.

.. code-block:: console

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python lazy_availability.py
