#
# Generic setup for using gcc
#
CXX = xlc++_r
CC  = xlc_r
FC  = xlf_r
F90 = xlf90_r

CXXFLAGS = -g
CFLAGS   = -g
FFLAGS   = -g -qextname=flush
F90FLAGS = -g -qlanglvl=2003std -qextname=flush

########################################################################

intel_version := $(shell $(CXX) -dumpversion)
COMP_VERSION := $(intel_version)

########################################################################

ifeq ($(DEBUG),TRUE)

  CXXFLAGS += -O0
  CFLAGS   += -O0
  FFLAGS   += -O0 
  F90FLAGS += -O0 

else

  CXXFLAGS += -O2
  CFLAGS   += -O2
  FFLAGS   += -O2
  F90FLAGS += -O2

endif



########################################################################

FFLAGS += -qdpc -qarch=auto -qtune=auto -qmaxmem=-1  -qfixed=132 -Xptxas -v -J$(fmoddir) -I $(fmoddir)
F90FLAGS += -qdpc -qarch=auto -qtune=auto -qmaxmem=-1 -qfixed -J$(fmoddir) -I $(fmoddir) -fimplicit-none

########################################################################

GENERIC_COMP_FLAGS =

ifeq ($(USE_OMP),TRUE)
  GENERIC_COMP_FLAGS += -qsmp=noauto:omp -qoffload
endif

CXXFLAGS += $(GENERIC_COMP_FLAGS) -std=c++11
CFLAGS   += $(GENERIC_COMP_FLAGS)
FFLAGS   += $(GENERIC_COMP_FLAGS)
F90FLAGS += $(GENERIC_COMP_FLAGS)

########################################################################

# ask gfortran the name of the library to link in.  First check for the
# static version.  If it returns only the name w/o a path, then it
# was not found.  In that case, ask for the shared-object version.
xlfr_liba  := $(shell $(F90) -print-file-name=libxlf90_r.a)
xlfr_libso := $(shell $(F90) -print-file-name=libxlf90_r.so)
quadmath_liba  := $(shell $(F90) -print-file-name=libquadmath.a)
quadmath_libso := $(shell $(F90) -print-file-name=libquadmath.so)
ifneq ($(xlfr_liba),libxlf90_r.a)  # if found the full path is printed, thus `neq`.
  xlfr_lib = $(xlfr_liba)
else
  xlfr_lib = $(xlfr_libso)
endif
ifneq ($(quadmath_liba),libquadmath.a)
  quadmath_lib = $(quadmath_liba)
else
  quadmath_lib = $(quadmath_libso)
endif

override XTRALIBS += $(xlfr_lib) $(quadmath_lib)
