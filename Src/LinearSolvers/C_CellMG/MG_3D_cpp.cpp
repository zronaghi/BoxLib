#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>
#include <iostream>

//ask Brian CONSTANTS
//MultiGrid.cpp

//Average Kernel
void C_AVERAGE(const Box* bx,
const int ng,
const int nc,
Real* c,
const Real* f){

 
	//const Box* bx;
	//Real *cp, *fp;
    
	int i2, j2, k2;
	int ijkn;
	int i2p1_j2p1_k2_n, i2_j2p1_k2_n, i2p1_j2_k2_n, i2_j2_k2_n;
	int i2p1_j2p1_k2p1_n, i2_j2p1_k2p1_n, i2p1_j2_k2p1_n, i2_j2_k2p1_n;     

	//const int ng;
	//c-field offsets
	const int BL_jStride = bx->length(0) + 2*ng;
	const int BL_kStride = BL_jStride * (bx->length(1) + 2*ng);
	const int BL_nStride = BL_kStride * (bx->length(2) + 2*ng);
	//f-field offsets
	const int BL_j2Stride = 2*(bx->length(0)) + 2*ng;
	const int BL_k2Stride = BL_j2Stride * (2*(bx->length(1)) + 2*ng);
	const int BL_n2Stride = BL_k2Stride * (2*(bx->length(2)) + 2*ng);
	
	//const int *lo = bx->loVect();
	//const int *hi = bx->hiVect();

	//const int offset = 
	//int abs_i, abs_j, abs_k;

	for (int n = 0; n<nc; n++){
		for (int k = 0; k < bx->length(2); ++k) {
			k2 = 2*k;
			//k2p1 = k2 + 1
			for (int j = 0; j < bx->length(1); ++j) {
				j2 = 2*j;
				//j2p1 = j2 + 1
				for (int i = 0; i < bx->length(0); ++i) {
					i2 = 2*i;
					//i2p1 = i2 + 1

					//abs_i = lo[0] + i;
					//abs_j = lo[1] + j;
					//abs_k = lo[2] + k;

					ijkn = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride + n*BL_nStride;

					//add k2Stide, n2Stride, et
					i2_j2_k2_n =       (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2p1_k2_n =   (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2_j2p1_k2_n =     (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2_k2_n =     (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2p1_k2p1_n = (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2_j2p1_k2p1_n =   (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2p1_j2_k2p1_n =   (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2_j2_k2p1_n =     (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;

					c[ijkn] =  (f[i2p1_j2p1_k2_n] + f[i2_j2p1_k2_n] + f[i2p1_j2_k2_n] + f[i2_j2_k2_n])*(0.125);
					c[ijkn] += (f[i2p1_j2p1_k2p1_n] + f[i2_j2p1_k2p1_n] + f[i2p1_j2_k2p1_n] + f[i2_j2_k2p1_n])*(0.125);
				}
			}
		}
	}
}


//Interpolation Kernel
void C_INTERP(const Box* bx,
const int ng,
const int nc,
Real* f,
const Real* c){

 
	//const Box* bx;
	//Real *cp, *fp;
	int i2, j2, k2;
	int ijkn;
	int i2p1_j2p1_k2_n, i2_j2p1_k2_n, i2p1_j2_k2_n, i2_j2_k2_n;
	int i2p1_j2p1_k2p1_n, i2_j2p1_k2p1_n, i2p1_j2_k2p1_n, i2_j2_k2p1_n;     

	//const int ng;
	//c-field offsets
	const int BL_jStride = bx->length(0) + 2*ng;
	const int BL_kStride = BL_jStride * (bx->length(1) + 2*ng);
	const int BL_nStride = BL_kStride * (bx->length(2) + 2*ng);
	//f-field offsets
	const int BL_j2Stride = 2*(bx->length(0)) + 2*ng;
	const int BL_k2Stride = BL_j2Stride * (2*(bx->length(1)) + 2*ng);
	const int BL_n2Stride = BL_k2Stride * (2*(bx->length(2)) + 2*ng);
	
	//const int *lo = bx->loVect();
	//const int *hi = bx->hiVect();

	//const int offset = 
	//int abs_i, abs_j, abs_k;

	for (int n = 0; n<nc; n++){
		for (int k = 0; k < bx->length(2); ++k) {
			k2 = 2*k;
			//k2p1 = k2 + 1
			for (int j = 0; j < bx->length(1); ++j) {
				j2 = 2*j;
				//j2p1 = j2 + 1
				for (int i = 0; i < bx->length(0); ++i) {
					i2 = 2*i;
					//i2p1 = i2 + 1

					//abs_i = lo[0] + i;
					//abs_j = lo[1] + j;
					//abs_k = lo[2] + k;
					
					//diagonal index
					ijkn = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride + n*BL_nStride;

					//add k2Stide, n2Stride, et
					i2_j2_k2_n =       (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2p1_k2_n =   (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2_j2p1_k2_n =     (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2_k2_n =     (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
					i2p1_j2p1_k2p1_n = (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2_j2p1_k2p1_n =   (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2p1_j2_k2p1_n =   (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					i2_j2_k2p1_n =     (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
					
					f[i2p1_j2p1_k2_n]       += c[ijkn];
					f[i2_j2p1_k2_n]         += c[ijkn];
					f[i2p1_j2_k2_n]         += c[ijkn];
					f[i2_j2_k2_n]           += c[ijkn];
					f[i2p1_j2p1_k2p1_n]     += c[ijkn];
					f[i2_j2p1_k2p1_n]       += c[ijkn];
					f[i2p1_j2_k2p1_n]       += c[ijkn];
					f[i2_j2_k2p1_n]         += c[ijkn];
				}
			}
		}
	}
}


//GSRB kernel
void C_GSRB_3D(
const Box& bx,
const Box& bbx,
const int ng,
const int nc,
const int rb,
const Real alpha,
const Real beta,
FArrayBox& phi,
const FArrayBox& rhs,
const FArrayBox& a,
const FArrayBox& bX,
const FArrayBox& bY,
const FArrayBox& bZ,
const FArrayBox& f0,
const Mask& m0,
const FArrayBox& f1,
const Mask& m1,
const FArrayBox& f2,
const Mask& m2,
const FArrayBox& f3,
const Mask& m3,
const FArrayBox& f4,
const Mask& m4,
const FArrayBox& f5,
const Mask& m5,
const Real* h)
{
	//box extends:
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
	//blo
	const int *blo = bbx.loVect();
	const int *bhi = bbx.hiVect();
	
	//some parameters
	Real omega= 1.15;
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);
	
	for (int n = 0; n<nc; n++){
		for (int k = lo[2]; k <= hi[2]; ++k) {
			for (int j = lo[1]; j <= hi[1]; ++j) {
				int ioff = (lo[0] + j + k + rb)%2;
				for (int i = lo[0] + ioff; i <= hi[0]; i+=2) {
					
					//BC terms
					Real cf0 = ( (i==blo[0]) && (m0(IntVect(blo[0]-1,j,k))>0) ? f0(IntVect(blo[0],j,k)) : 0. );
					Real cf1 = ( (j==blo[1]) && (m1(IntVect(i,blo[1]-1,k))>0) ? f1(IntVect(i,blo[1],k)) : 0. );
					Real cf2 = ( (k==blo[2]) && (m2(IntVect(i,j,blo[2]-1))>0) ? f2(IntVect(i,j,blo[2])) : 0. );
					Real cf3 = ( (i==bhi[0]) && (m3(IntVect(bhi[0]+1,j,k))>0) ? f3(IntVect(bhi[0],j,k)) : 0. );
					Real cf4 = ( (j==bhi[1]) && (m4(IntVect(i,bhi[1]+1,k))>0) ? f4(IntVect(i,bhi[1],k)) : 0. );
					Real cf5 = ( (k==bhi[2]) && (m5(IntVect(i,j,bhi[2]+1))>0) ? f5(IntVect(i,j,bhi[2])) : 0. );
					
					//assign ORA constants
					double gamma = alpha * a(IntVect(i,j,k))
									+ dhx * (bX(IntVect(i,j,k)) + bX(IntVect(i+1,j,k)))
									+ dhy * (bY(IntVect(i,j,k)) + bY(IntVect(i,j+1,k)))
									+ dhz * (bZ(IntVect(i,j,k)) + bZ(IntVect(i,j,k+1)));
					
					double g_m_d = gamma
									- dhx * (bX(IntVect(i,j,k))*cf0 + bX(IntVect(i+1,j,k))*cf3)
									- dhy * (bY(IntVect(i,j,k))*cf1 + bY(IntVect(i,j+1,k))*cf4)
									- dhz * (bZ(IntVect(i,j,k))*cf2 + bZ(IntVect(i,j,k+1))*cf5);
					
					double rho =  dhx * (bX(IntVect(i,j,k))*phi(IntVect(i-1,j,k),n) + bX(IntVect(i+1,j,k))*phi(IntVect(i+1,j,k),n))
								+ dhy * (bY(IntVect(i,j,k))*phi(IntVect(i,j-1,k),n) + bY(IntVect(i,j+1,k))*phi(IntVect(i,j+1,k),n))
								+ dhz * (bZ(IntVect(i,j,k))*phi(IntVect(i,j,k-1),n) + bZ(IntVect(i,j,k+1))*phi(IntVect(i,j,k+1),n));
					
					double res = rhs(IntVect(i,j,k),n) - gamma * phi(IntVect(i,j,k),n) + rho;
					phi(IntVect(i,j,k),n) += omega/g_m_d * res;
				}
			}
		}
	}
}