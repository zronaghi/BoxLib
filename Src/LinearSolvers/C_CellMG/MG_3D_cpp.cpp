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
const Box* bx,
const Box* bbx,
const int ng,
const int nc,
const int rb,
const Real alpha,
const Real beta,
Real* phi,
const Real* rhs,
const Real* a,
const Real* bX,
const Real* bY,
const Real* bZ,
const Real* f0,
const int* m0,
const Real* f1,
const int* m1,
const Real* f2,
const int* m2,
const Real* f3,
const int* m3,
const Real* f4,
const int* m4,
const Real* f5,
const int* m5,
const Real* h)
{

	//field offsets
	const int BL_jStride = bx->length(0) + 2*ng;
	const int BL_kStride = BL_jStride * (bx->length(1) + 2*ng);
	const int BL_nStride = BL_kStride * (bx->length(2) + 2*ng);

	//box extends:
	const int *lo = bx->loVect();
	const int *hi = bx->hiVect();
	//blo
	const int *blo = bbx->loVect();
	const int *bhi = bbx->hiVect();
	
	//std::cout << "bbox: lg=(" << bbx->length(0) << "," << bbx->length(1) << "," << bbx->length(2) << ")" << std::endl;
	//std::cout << "bbox: lo=(" << blo[0] << "," << blo[1] << "," << blo[2] << ")" << std::endl;
	//std::cout << "bbox: hi=(" << bhi[0] << "," << bhi[1] << "," << bhi[2] << ")" << std::endl;
	//std::cout << "tbox: lg=(" << bx->length(0) << "," << bx->length(1) << "," << bx->length(2) << ")" << std::endl;
	//std::cout << "tbox: lo=(" << lo[0] << "," << lo[1] << "," << lo[2] << ")" << std::endl;
	//std::cout << "tbox: hi=(" << hi[0] << "," << hi[1] << "," << hi[2] << ")" << std::endl;
	
	//some parameters
	Real omega= 1.15;
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);
	
	//useful params:
	int indf, indm;
		
	for (int n = 0; n<nc; n++){
		for (int k = 0; k < bx->length(2); ++k) {
			for (int j = 0; j < bx->length(1); ++j) {
				//might need to revisit this in terms of offsets:
				int ioff = (lo[0] + j + k + rb)%2;
				for (int i = ioff; i < bx->length(0); i+=2) {
					
					//diagonal index
					int i_j_k = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					int ip1_j_k = (i+1 + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					int im1_j_k = (i-1 + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					int i_jp1_k = (i + ng) + (j+1 + ng)*BL_jStride + (k + ng)*BL_kStride;
					int i_jm1_k = (i + ng) + (j-1 + ng)*BL_jStride + (k + ng)*BL_kStride;
					int i_j_kp1 = (i + ng) + (j + ng)*BL_jStride + (k+1 + ng)*BL_kStride;
					int i_j_km1 = (i + ng) + (j + ng)*BL_jStride + (k-1 + ng)*BL_kStride;
					
					//std::cout << i << " " << j << " " << k << std::endl;
					//std::cout << i_j_k << " " << ip1_j_k << " " << im1_j_k << " " << i_jp1_k << " " << i_jm1_k << " " << i_j_kp1 << " " << i_j_km1 << std::endl;
					
					//deal with BC: for indexing, we need to shift it relative to blo!
					//cf0:
					indf = (blo[0]-blo[0] + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					indm = (blo[0]-blo[0]-1 + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					Real cf0 = 0.; // ( (i+lo[0])==blo[0] && (m0[indm]>0) ? f0[indf] : 0. );
					//cf1:
					indf = (i + ng) + (blo[1]-blo[1] + ng)*BL_jStride + (k + ng)*BL_kStride;
					indm = (i + ng) + (blo[1]-blo[1]-1 + ng)*BL_jStride + (k + ng)*BL_kStride;
					Real cf1 = 0.; //( (j+lo[1])==blo[1] && (m1[indm]>0) ? f1[indf] : 0. );
					//cf2
					indf = (i + ng) + (j + ng)*BL_jStride + (blo[2]-blo[2] + ng)*BL_kStride;
					indm = (i + ng) + (j + ng)*BL_jStride + (blo[2]-blo[2]-1 + ng)*BL_kStride;
					Real cf2 = 0.; //( (k+lo[2])==blo[2] && (m2[indm]>0) ? f2[indf] : 0. );
					//cf3
					indf = (bhi[0]-blo[0] + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					indm = (bhi[0]-blo[0]+1 + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride;
					Real cf3 = 0.; //( (i+lo[0])==bhi[0] && (m3[indm]>0) ? f3[indf] : 0. );
					//cf4
					indf = (i + ng) + (bhi[1]-blo[1] + ng)*BL_jStride + (k + ng)*BL_kStride;
					indm = (i + ng) + (bhi[1]-blo[1]+1 + ng)*BL_jStride + (k + ng)*BL_kStride;
					Real cf4 = 0.; //( (j+lo[1])==bhi[1] && (m4[indm]>0) ? f4[indf] : 0. );
					//cf5
					indf = (i + ng) + (j + ng)*BL_jStride + (bhi[2]-blo[2] + ng)*BL_kStride;
					indm = (i + ng) + (j + ng)*BL_jStride + (bhi[2]-blo[2]+1 + ng)*BL_kStride;
					Real cf5 = 0.; //( (k+lo[2])==bhi[2] && (m5[indm]>0) ? f5[indf] : 0. );
					
					//assign ORA constants
					double gamma = alpha * a[i_j_k]
									+ dhx * (bX[i_j_k] + bX[ip1_j_k])
									+ dhy * (bY[i_j_k] + bY[i_jp1_k])
									+ dhz * (bZ[i_j_k] + bZ[i_j_kp1]);
					
					double g_m_d = gamma
									- dhx * (bX[i_j_k]*cf0 + bX[ip1_j_k]*cf3)
									- dhy * (bY[i_j_k]*cf1 + bY[i_jp1_k]*cf4)
									- dhz * (bZ[i_j_k]*cf2 + bZ[i_j_kp1]*cf5);
					
					double rho =  dhx * (bX[i_j_k]*phi[im1_j_k + n*BL_nStride] + bX[ip1_j_k]*phi[ip1_j_k + n*BL_nStride])
								+ dhy * (bY[i_j_k]*phi[i_jm1_k + n*BL_nStride] + bY[i_jp1_k]*phi[i_jp1_k + n*BL_nStride])
								+ dhz * (bZ[i_j_k]*phi[i_j_km1 + n*BL_nStride] + bZ[i_j_kp1]*phi[i_j_kp1 + n*BL_nStride]);
					
					double res = rhs[i_j_k + n*BL_nStride] - gamma * phi[i_j_k + n*BL_nStride] + rho;
					phi[i_j_k + n*BL_nStride] += omega/g_m_d * res;
					
					//std::cout << gamma << " " << g_m_d << " " << rho << " " << res << " " << phi[i_j_k + n*BL_nStride] << std::endl;
				}
			}
		}
	}
}