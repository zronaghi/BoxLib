#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>
#include <iostream>
#include <Kokkos_Core.hpp>
//ask Brian CONSTANTS
//MultiGrid.cpp

//Average Kernel
void C_AVERAGE(const Box& bx,
const int nc,
FArrayBox& c,
const FArrayBox& f){
	
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
        typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>> t_policy;
        Kokkos::OpenMP::print_configuration(std::cout);
        for (int n = 0; n<nc; n++){
                Kokkos::Experimental::md_parallel_for(t_policy({lo[2],lo[1],lo[0]},{hi[2],hi[1],hi[0]},{4,4,32}),
                                                                KOKKOS_LAMBDA(const int &k, const int &j, const int &i){
		//for (int k = lo[2]; k <= hi[2]; ++k) {
			int k2 = 2*k;
		//	for (int j = lo[1]; j <= hi[1]; ++j) {
				int j2 = 2*j;
		//		for (int i = lo[0]; i <= hi[0]; ++i) {
					int i2 = 2*i;

					c(IntVect(i,j,k),n) =  (f(IntVect(i2+1,j2+1,k2),n) + f(IntVect(i2,j2+1,k2),n) + f(IntVect(i2+1,j2,k2),n) + f(IntVect(i2,j2,k2),n))*0.125;
					c(IntVect(i,j,k),n) += (f(IntVect(i2+1,j2+1,k2+1),n) + f(IntVect(i2,j2+1,k2+1),n) + f(IntVect(i2+1,j2,k2+1),n) + f(IntVect(i2,j2,k2+1),n))*0.125;
//				}
//			}
		});
	}
}

