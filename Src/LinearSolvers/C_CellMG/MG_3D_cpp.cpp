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
void C_AVERAGE(const Box& bx,
const int nc,
FArrayBox& c,
const FArrayBox& f){
	
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
	
	//pointer, needed for c++ mapping
	//c:
	const Real* cpt=c.dataPtr();
	const int* c_lo=c.loVect();
	const int* c_hi=c.hiVect();
	//f
	const Real* fpt=c.dataPtr();
	const int* f_lo=c.loVect();
	const int* f_hi=c.hiVect();
	
#pragma omp target data map(to: hi, lo, f_lo, f_hi, cpt, c_lo, c_hi)
#pragma omp target update to(fpt)
	{
#pragma omp target teams distribute parallel for collapse(4) 
		for (int n = 0; n<nc; n++){
			for (int k = lo[2]; k <= hi[2]; ++k) {
				for (int j = lo[1]; j <= hi[1]; ++j) {
					for (int i = lo[0]; i <= hi[0]; ++i) {
						c(IntVect(i,j,k),n) =  (f(IntVect(2*i+1,2*j+1,2*k),n) + f(IntVect(2*i,2*j+1,2*k),n) + f(IntVect(2*i+1,2*j,2*k),n) + f(IntVect(2*i,2*j,2*k),n))*0.125;
						c(IntVect(i,j,k),n) += (f(IntVect(2*i+1,2*j+1,2*k+1),n) + f(IntVect(2*i,2*j+1,2*k+1),n) + f(IntVect(2*i+1,2*j,2*k+1),n) + f(IntVect(2*i,2*j,2*k+1),n))*0.125;
					}
				}
			}
		}
	}
#pragma omp target update from(cpt)
}


//Interpolation Kernel
void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){
	
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
	
	//pointer, needed for c++ mapping
	//c:
	const Real* cpt=c.dataPtr();
	const int* c_lo=c.loVect();
	const int* c_hi=c.hiVect();
	//f
	const Real* fpt=c.dataPtr();
	const int* f_lo=c.loVect();
	const int* f_hi=c.hiVect();

#pragma omp target data map(to: hi, lo, f_lo, f_hi, cpt, c_lo, c_hi)
#pragma omp target update to(cpt)
	{
#pragma omp target teams distribute parallel for collapse(4) 
		for (int n = 0; n<nc; n++){
			for (int k = lo[2]; k <= hi[2]; ++k) {
				for (int j = lo[1]; j <= hi[1]; ++j) {
					for (int i = lo[0]; i <= hi[0]; ++i) {
						f(IntVect(2*i+1,2*j+1,2*k  ),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i  ,2*j+1,2*k  ),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i+1,2*j  ,2*k  ),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i  ,2*j  ,2*k  ),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i+1,2*j+1,2*k+1),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i  ,2*j+1,2*k+1),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i+1,2*j  ,2*k+1),n)       += c(IntVect(i,j,k),n);
						f(IntVect(2*i  ,2*j  ,2*k+1),n)       += c(IntVect(i,j,k),n);
					}
				}
			}
		}
	}
#pragma omp target update from(fpt)
}


//-----------------------------------------------------------------------
//      
//     Gauss-Seidel Red-Black (GSRB):
//     Apply the GSRB relaxation to the state phi for the equation
//     L(phi) = alpha*a(x)*phi(x) - beta*Div(b(x)Grad(phi(x))) = rhs(x)
//     central differenced, according to the arrays of boundary
//     masks (m#) and auxiliary data (f#).
//     
//     In general, if the linear operator L=gamma*y-rho, the GS relaxation
//     is y = (R - rho)/gamma.  Near a boundary, the ghost data is filled
//     using a polynomial interpolant based on the "old" phi values, so
//     L=(gamma-delta)*y - rho + delta*yOld.  The resulting iteration is
//     
//     y = (R - delta*yOld + rho)/(gamma - delta)
//     
//     This expression is valid additionally in the interior provided
//     delta->0 there.  delta is constructed by summing all the
//     contributions to the central stencil element coming from boundary 
//     interpolants.  The f#s contain the corresponding coefficient of 
//     the interpolating polynomial.  The masks are set > 0 if the boundary 
//     value was filled with an interpolant involving the central stencil 
//     element.
//     
//-----------------------------------------------------------------------
void C_GSRB_3D(
const Box& bx,
const Box& bbx,
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
	
	//pointer, needed for c++ mapping
	//phi:
	Real* phipt=phi.dataPtr();
	const int* phi_lo=phi.loVect();
	const int* phi_hi=phi.hiVect();
	//rhs
	const Real* rhspt=rhs.dataPtr();
	const int* rhs_lo=rhs.loVect();
	const int* rhs_hi=rhs.hiVect();
	//a
	const Real* apt=a.dataPtr();
	const int* a_lo=a.loVect();
	const int* a_hi=a.hiVect();
	//bX
	const Real* bXpt=bX.dataPtr();
	const int* bX_lo=bX.loVect();
	const int* bX_hi=bX.hiVect();
	//bY
	const Real* bYpt=bY.dataPtr();
	const int* bY_lo=bY.loVect();
	const int* bY_hi=bY.hiVect();
	//bZ
	const Real* bZpt=bZ.dataPtr();
	const int* bZ_lo=bZ.loVect();
	const int* bZ_hi=bZ.hiVect();
	//m0
	const int* m0pt=m0.dataPtr();
	const int* m0_lo=m0.loVect();
	const int* m0_hi=m0.hiVect();
	//m1
	const int* m1pt=m1.dataPtr();
	const int* m1_lo=m1.loVect();
	const int* m1_hi=m1.hiVect();
	//m2
	const int* m2pt=m2.dataPtr();
	const int* m2_lo=m2.loVect();
	const int* m2_hi=m2.hiVect();
	//m3
	const int* m3pt=m3.dataPtr();
	const int* m3_lo=m3.loVect();
	const int* m3_hi=m3.hiVect();
	//m4
	const int* m4pt=m4.dataPtr();
	const int* m4_lo=m4.loVect();
	const int* m4_hi=m4.hiVect();
	//m5
	const int* m5pt=m5.dataPtr();
	const int* m5_lo=m5.loVect();
	const int* m5_hi=m5.hiVect();
	//f0
	const Real* f0pt=f0.dataPtr();
	const int* f0_lo=f0.loVect();
	const int* f0_hi=f0.hiVect();
	//f1
	const Real* f1pt=f1.dataPtr();
	const int* f1_lo=f1.loVect();
	const int* f1_hi=f1.hiVect();
	//f2
	const Real* f2pt=f2.dataPtr();
	const int* f2_lo=f2.loVect();
	const int* f2_hi=f2.hiVect();
	//f3
	const Real* f3pt=f3.dataPtr();
	const int* f3_lo=f3.loVect();
	const int* f3_hi=f3.hiVect();
	//f4
	const Real* f4pt=f4.dataPtr();
	const int* f4_lo=f4.loVect();
	const int* f4_hi=f4.hiVect();
	//f5
	const Real* f5pt=f5.dataPtr();
	const int* f5_lo=f5.loVect();
	const int* f5_hi=f5.hiVect();
	
	//some parameters
	Real omega= 1.15;
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);
	
#pragma omp target data map(to: blo, bhi, lo, hi)
#pragma omp target data map(to: phi_lo, phi_hi, rhs_lo, rhs_hi, a_lo, a_hi)
#pragma omp target data map(to: bX_lo, bX_hi, bY_lo, bY_hi, bZ_lo, bZ_hi)
#pragma omp target data map(to: m0_lo, m0_hi, m1_lo, m1_hi, m2_lo, m2_hi, m3_lo, m3_hi, m4_lo, m4_hi, m5_lo, m5_hi)
#pragma omp target data map(to: f0_lo, f0_hi, f1_lo, f1_hi, f2_lo, f2_hi, f3_lo, f3_hi, f4_lo, f4_hi, f5_lo, f5_hi) 
#pragma omp target update to(phipt,rhspt,apt,bXpt,bYpt,bZpt,m0pt,m1pt,m2pt,m3pt,m4pt,m5pt,f0pt,f1pt,f2pt,f3pt,f4pt,f5pt)
	{
#pragma omp target teams distribute collapse(3)
		for (int n = 0; n<nc; n++){
			for (int k = lo[2]; k <= hi[2]; ++k) {
				for (int j = lo[1]; j <= hi[1]; ++j) {
					int ioff = (lo[0] + j + k + rb)%2;
#pragma omp parallel for firstprivate(alpha,dhx,dhy,dhz,omega,ioff) default(shared)
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
#pragma omp target update from(phipt)
}

//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
void C_ADOTX(
const Box& bx,
const int nc,
FArrayBox& y,
const FArrayBox& x,
Real alpha,
Real beta,
const FArrayBox& a,
const FArrayBox& bX,
const FArrayBox& bY,
const FArrayBox& bZ,
const Real* h)
{

	//box extends:
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
	
	//pointer, needed for c++ mapping
	//y
	Real* ypt=y.dataPtr();
	const int* y_lo=y.loVect();
	const int* y_hi=y.hiVect();
	//x
	const Real* xpt=x.dataPtr();
	const int* x_lo=x.loVect();
	const int* x_hi=x.hiVect();
	//a
	const Real* apt=a.dataPtr();
	const int* a_lo=a.loVect();
	const int* a_hi=a.hiVect();
	//bX
	const Real* bXpt=bX.dataPtr();
	const int* bX_lo=bX.loVect();
	const int* bX_hi=bX.hiVect();
	//bY
	const Real* bYpt=bY.dataPtr();
	const int* bY_lo=bY.loVect();
	const int* bY_hi=bY.hiVect();
	//bZ
	const Real* bZpt=bZ.dataPtr();
	const int* bZ_lo=bZ.loVect();
	const int* bZ_hi=bZ.hiVect();
	
	//some parameters
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);

#pragma omp target data map(to: lo, hi)
#pragma omp target data map(to: y_lo, y_hi, x_lo, x_hi, a_lo, a_hi)
#pragma omp target data map(to: bX_lo, bX_hi, bY_lo, bY_hi, bZ_lo, bZ_hi)
#pragma omp target update to(apt,xpt,bXpt,bYpt,bZpt)
	{
#pragma omp target teams distribute parallel for collapse(4)
		for (int n = 0; n<nc; n++){
			for (int k = lo[2]; k <= hi[2]; ++k) {
				for (int j = lo[1]; j <= hi[1]; ++j) {
					for (int i = lo[0]; i <= hi[0]; ++i) {
						y(IntVect(i,j,k),n) = alpha*a(IntVect(i,j,k))*x(IntVect(i,j,k),n)
											- dhx * (   bX(IntVect(i+1,j,  k  )) * ( x(IntVect(i+1,j,  k),  n) - x(IntVect(i,  j,  k  ),n) )
													  - bX(IntVect(i,  j,  k  )) * ( x(IntVect(i,  j,  k),  n) - x(IntVect(i-1,j,  k  ),n) ) 
													)
											- dhy * (   bY(IntVect(i,  j+1,k  )) * ( x(IntVect(i,  j+1,k),  n) - x(IntVect(i,  j  ,k  ),n) )
													  - bY(IntVect(i,  j,  k  )) * ( x(IntVect(i,  j,  k),  n) - x(IntVect(i,  j-1,k  ),n) )
													)
											- dhz * (   bZ(IntVect(i,  j,  k+1)) * ( x(IntVect(i,  j,  k+1),n) - x(IntVect(i,  j  ,k  ),n) )
													  - bZ(IntVect(i,  j,  k  )) * ( x(IntVect(i,  j,  k),  n) - x(IntVect(i,  j,  k-1),n) )
													);
					}
				}
			}
		}
	}
#pragma omp target update from(ypt)
}

//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
void C_NORMA(
const Box& bx,
const int nc,
Real& res,
const Real alpha,
const Real beta,
const FArrayBox& a,
const FArrayBox& bX,
const FArrayBox& bY,
const FArrayBox& bZ,
const Real* h)
{

	//box extends:
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();
	
	//pointer, needed for c++ mapping
	//a
	const Real* apt=a.dataPtr();
	const int* a_lo=a.loVect();
	const int* a_hi=a.hiVect();
	//bX
	const Real* bXpt=bX.dataPtr();
	const int* bX_lo=bX.loVect();
	const int* bX_hi=bX.hiVect();
	//bY
	const Real* bYpt=bY.dataPtr();
	const int* bY_lo=bY.loVect();
	const int* bY_hi=bY.hiVect();
	//bZ
	const Real* bZpt=bZ.dataPtr();
	const int* bZ_lo=bZ.loVect();
	const int* bZ_hi=bZ.hiVect();
	
	//some parameters
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);
	
	//initialize to zero
    res = 0.0;

#pragma omp target data map(to: lo, hi)
#pragma omp target data map(to: a_lo, a_hi)
#pragma omp target data map(to: bX_lo, bX_hi, bY_lo, bY_hi, bZ_lo, bZ_hi)
#pragma omp target update to(apt,bXpt,bYpt,bZpt)
	{
#pragma omp teams distribute parallel for collapse(4) firstprivate(alpha,dhx,dhy,dhz) reduction(max:res)
		for (int n = 0; n<nc; n++){
			for (int k = lo[2]; k <= hi[2]; ++k) {
				for (int j = lo[1]; j <= hi[1]; ++j) {
					for (int i = lo[0]; i <= hi[0]; ++i) {
						Real tmpval= alpha*a(IntVect(i,j,k))
									+ dhx * ( bX(IntVect(i+1,j,k)) + bX(IntVect(i,j,k)) )
									+ dhy * ( bY(IntVect(i,j+1,k)) + bY(IntVect(i,j,k)) )
									+ dhz * ( bZ(IntVect(i,j,k+1)) + bZ(IntVect(i,j,k)) );
						res = std::max(res,std::abs(tmpval));
					
						//now add the rest
						res +=    std::abs( dhx * bX(IntVect(i+1,j,k)) ) + std::abs( dhx * bX(IntVect(i,j,k)) )
								+ std::abs( dhy * bY(IntVect(i,j+1,k)) ) + std::abs( dhy * bY(IntVect(i,j,k)) )
								+ std::abs( dhz * bZ(IntVect(i,j,k+1)) ) + std::abs( dhz * bZ(IntVect(i,j,k)) );
					}
				}
			}
		}
	}
}

//-----------------------------------------------------------------------
//
//     Fill in fluxes
//
void C_FLUX(
const Box& xbx,
const Box& ybx,
const Box& zbx,
const int nc,
FArrayBox& x,
FArrayBox& xflux,
FArrayBox& yflux,
FArrayBox& zflux,
Real alpha,
Real beta,
const FArrayBox& a,
const FArrayBox& bX,
const FArrayBox& bY,
const FArrayBox& bZ,
const Real* h)
{

	//box extends:
	const int *xlo = xbx.loVect();
	const int *xhi = xbx.hiVect();
	const int *ylo = ybx.loVect();
	const int *yhi = ybx.hiVect();
	const int *zlo = zbx.loVect();
	const int *zhi = zbx.hiVect();
	
	//some parameters
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);

	//fill the fluxes:
	for (int n = 0; n<nc; n++){
		//x-flux
		for (int k = xlo[2]; k <= xhi[2]; ++k) {
			for (int j = xlo[1]; j <= xhi[1]; ++j) {
				for (int i = xlo[0]; i <= xhi[0]; ++i) {
					xflux(IntVect(i,j,k),n) = - dhx * bX(IntVect(i,j,k))*( x(IntVect(i,j,k),n) - x(IntVect(i-1,j,k),n) );
				}
			}
		}
		//y-flux
		for (int k = ylo[2]; k <= yhi[2]; ++k) {
			for (int j = ylo[1]; j <= yhi[1]; ++j) {
				for (int i = ylo[0]; i <= yhi[0]; ++i) {
					yflux(IntVect(i,j,k),n) = - dhy * bY(IntVect(i,j,k))*( x(IntVect(i,j,k),n) - x(IntVect(i,j-1,k),n) );
				}
			}
		}
		//z-flux
		for (int k = zlo[2]; k <= zhi[2]; ++k) {
			for (int j = zlo[1]; j <= zhi[1]; ++j) {
				for (int i = zlo[0]; i <= zhi[0]; ++i) {
					zflux(IntVect(i,j,k),n) = - dhz * bZ(IntVect(i,j,k))*( x(IntVect(i,j,k),n) - x(IntVect(i,j,k-1),n) );
				}
			}
		}
	}
}
