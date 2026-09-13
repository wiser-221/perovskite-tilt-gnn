#!/usr/bin/env bash
# User-local VASP build environment. Source this file before make/test.
export VASP_ROOT="${HOME}/home/software/vasp.6.5.1"
export VASP_DEPS="${HOME}/home/software/vasp-deps"
export PATH="${VASP_ROOT}/bin:${VASP_DEPS}/usr/bin:${PATH}"
export OPAL_PREFIX="${VASP_DEPS}/usr"
export OMPI_HOME="${VASP_DEPS}/usr"
export LD_LIBRARY_PATH="${VASP_DEPS}/usr/lib/x86_64-linux-gnu:${VASP_DEPS}/usr/lib/x86_64-linux-gnu/openblas-pthread:${VASP_DEPS}/usr/lib/gcc/x86_64-linux-gnu/15:${VASP_DEPS}/usr/lib/x86_64-linux-gnu/openmpi:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${VASP_DEPS}/usr/include/c++/15:${VASP_DEPS}/usr/include/x86_64-linux-gnu/c++/15:${VASP_DEPS}/usr/include/x86_64-linux-gnu:${VASP_DEPS}/usr/include:${CPLUS_INCLUDE_PATH:-}"
export OPENBLAS_ROOT="${VASP_DEPS}/usr"
export SCALAPACK_ROOT="${VASP_DEPS}/usr"
export FFTW_ROOT="${VASP_DEPS}/usr"
export VASP_PP_PATH="${HOME}/home/software/potpaw_PBE.64"
export PMG_VASP_PSP_DIR="${HOME}/home/software/vasp_psp"
