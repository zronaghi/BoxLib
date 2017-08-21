#
# Generic setup for using Intel
#
CXX = icpc
CC  = icc
FC  = ifort
F90 = ifort

CXXFLAGS =
CFLAGS   =
FFLAGS   =
F90FLAGS =

########################################################################

intel_version := $(shell $(CXX) -dumpversion)

COMP_VERSION := $(intel_version)

########################################################################

ifeq ($(DEBUG),TRUE)

  CXXFLAGS += -g -O0 -traceback -Wcheck
  CFLAGS   += -g -O0 -traceback -Wcheck
  FFLAGS   += -g -O0 -traceback -check bounds,uninit,pointers
  F90FLAGS += -g -O0 -traceback -check bounds,uninit,pointers

else

  CXXFLAGS += -g -O2 -qopt-report=5 -qopt-report-phase=vec -ip
  CFLAGS   += -g -O2 -qopt-report=5 -qopt-report-phase=vec -ip
  FFLAGS   += -g -O2 -qopt-report=5 -qopt-report-phase=vec -ip
  F90FLAGS += -g -O2 -qopt-report=5 -qopt-report-phase=vec -ip

endif

########################################################################

CXXFLAGS += -std=c++11
CFLAGS   += -std=c99

F90FLAGS += -module $(fmoddir) -I$(fmoddir) -implicitnone
FFLAGS   += -module $(fmoddir) -I$(fmoddir)

########################################################################

GENERIC_COMP_FLAGS =

ifeq ($(USE_OMP),TRUE)
  ifeq ($(firstword $(sort 16.0 $(intel_version))), 16.0) 
    GENERIC_COMP_FLAGS += -qopenmp -qopenmp-offload=host
  else
    GENERIC_COMP_FLAGS += -openmp -qopenmp-offload=host
  endif
endif

CXXFLAGS += $(GENERIC_COMP_FLAGS)
CFLAGS   += $(GENERIC_COMP_FLAGS)
FFLAGS   += $(GENERIC_COMP_FLAGS)
F90FLAGS += $(GENERIC_COMP_FLAGS)

########################################################################

override XTRALIBS += -lifcore

ifeq ($(USE_OMP),TRUE)
  override XTRALIBS += -lifcoremt
endif
