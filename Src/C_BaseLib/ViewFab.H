#ifndef BL_VIEWFAB_H
#define BL_VIEWFAB_H

#include <winstd.H>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <BLassert.H>
#include <Box.H>
#include <BoxList.H>
#include <CArena.H>
#include <Looping.H>
#include <REAL.H>
#include <BLProfiler.H>
#include <PArray.H>

#include <Kokkos_Core.hpp>


#if BL_SPACEDIM == 3
//ND
template<typename T>
using hostview = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using devspace = Kokkos::DefaultExecutionSpace;
#ifdef KOKKOS_ENABLE_CUDA
//device view
template<typename T>
using devview = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
//iteration policies
static constexpr Kokkos::Experimental::Iterate outer_iter_policy = Kokkos::Experimental::Iterate::Right;
static constexpr Kokkos::Experimental::Iterate inner_iter_policy = Kokkos::Experimental::Iterate::Left;
////subview class
//template <typename T>
//using devsubview = Kokkos::View<T*, Kokkos::LayoutStride, Kokkos::CudaSpace, Kokkos::MemoryTraits< Kokkos::Restrict | Kokkos::Unmanaged > >;
#else
//device view
template<typename T>
using devview = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::HostSpace>;
//iteration policies
static constexpr Kokkos::Experimental::Iterate outer_iter_policy = Kokkos::Experimental::Iterate::Left;
static constexpr Kokkos::Experimental::Iterate inner_iter_policy = Kokkos::Experimental::Iterate::Left;
////subview class
//template <typename T>
//using devsubview = Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits< Kokkos::Restrict | Kokkos::Unmanaged > >;
#endif
#endif

template <int dim>
using mdpolicy = Kokkos::Experimental::MDRangePolicy<
    Kokkos::Experimental::Rank<dim, outer_iter_policy, inner_iter_policy>,
    Kokkos::IndexType<int> >;

//a small class for wrapping kokkos views nicely
template <typename T>
class ViewFab {
public:

#if BL_SPACEDIM == 3
    hostview<T****> h_data;
    devview<T****> d_data;
    //access operator
    KOKKOS_FORCEINLINE_FUNCTION
    T& operator()(const int i, const int j, const int k, const int n = 0) const {
        return d_data(i-smallend[0], j-smallend[1], k-smallend[2], n);
    }
#else
#error "ViewFab only supports 3D!";
#endif
    int smallend[BL_SPACEDIM + 1];

    void syncH2D(){
        Kokkos::deep_copy(d_data,h_data);
    }

    void syncD2H(){
        Kokkos::deep_copy(h_data,d_data);
    }

};

namespace Kokkos {
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
const T& max2(const T& a, const T& b) {
  return (b < a) ? a : b;
}
}

#endif
