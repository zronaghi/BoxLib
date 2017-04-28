#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>

//ask Brian CONSTANTS
//MultiGrid.cpp

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
	
	const int *lo = bx->loVect();
	const int *hi = bx->hiVect();

	//const int offset = 
	int abs_i, abs_j, abs_k;

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

					abs_i = lo[0] + i;
					abs_j = lo[1] + j;
					abs_k = lo[2] + k;

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
	
	const int *lo = bx->loVect();
	const int *hi = bx->hiVect();

	//const int offset = 
	int abs_i, abs_j, abs_k;

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

					abs_i = lo[0] + i;
					abs_j = lo[1] + j;
					abs_k = lo[2] + k;

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