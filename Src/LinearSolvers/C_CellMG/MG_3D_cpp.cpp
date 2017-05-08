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

	for (int n = 0; n<nc; n++){
		for (int k = lo[2]; k <= hi[2]; ++k) {
			int k2 = 2*k;
			for (int j = lo[1]; j <= hi[1]; ++j) {
				int j2 = 2*j;
				for (int i = lo[0]; i <= hi[0]; ++i) {
					int i2 = 2*i;

					c(IntVect(i,j,k),n) =  (f(IntVect(i2+1,j2+1,k2),n) + f(IntVect(i2,j2+1,k2),n) + f(IntVect(i2+1,j2,k2),n) + f(IntVect(i2,j2,k2),n))*0.125;
					c(IntVect(i,j,k),n) += (f(IntVect(i2+1,j2+1,k2+1),n) + f(IntVect(i2,j2+1,k2+1),n) + f(IntVect(i2+1,j2,k2+1),n) + f(IntVect(i2,j2,k2+1),n))*0.125;
				}
			}
		}
	}
}


//Interpolation Kernel
void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){
	
	const int *lo = bx.loVect();
	const int *hi = bx.hiVect();

	for (int n = 0; n<nc; n++){
		for (int k = lo[2]; k <= hi[2]; ++k) {
			int k2 = 2*k;
			for (int j = lo[1]; j <= hi[1]; ++j) {
				int j2 = 2*j;
				for (int i = lo[0]; i <= hi[0]; ++i) {
					int i2 = 2*i;
					
					f(IntVect(i2+1,j2+1,k2  ),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2  ,j2+1,k2  ),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2+1,j2  ,k2  ),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2  ,j2  ,k2  ),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2+1,j2+1,k2+1),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2  ,j2+1,k2+1),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2+1,j2  ,k2+1),n)       += c(IntVect(i,j,k),n);
					f(IntVect(i2  ,j2  ,k2+1),n)       += c(IntVect(i,j,k),n);
				}
			}
		}
	}
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
	
	//some parameters
	Real dhx = beta/(h[0]*h[0]);
	Real dhy = beta/(h[1]*h[1]);
	Real dhz = beta/(h[2]*h[2]);

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

////-----------------------------------------------------------------------
////
////     Fill in a matrix x vector operator here
////
//      subroutine C_NORMA(
//     &     res,
//     $     alpha, beta,
//     $     a, DIMS(a),
//     $     bX,DIMS(bX),
//     $     bY,DIMS(bY),
//     $     bZ,DIMS(bZ),
//     $     lo,hi,nc,
//     $     h
//     $     )
//      implicit none
//      REAL_T alpha, beta, res
//      integer lo(BL_SPACEDIM), hi(BL_SPACEDIM), nc
//      integer DIMDEC(a)
//      integer DIMDEC(bX)
//      integer DIMDEC(bY)
//      integer DIMDEC(bZ)
//      REAL_T  a(DIMV(a))
//      REAL_T bX(DIMV(bX))
//      REAL_T bY(DIMV(bY))
//      REAL_T bZ(DIMV(bZ))
//      REAL_T h(BL_SPACEDIM)
//
//      integer i,j,k,n
//      REAL_T dhx,dhy,dhz
//
//      dhx = beta/h(1)**2
//      dhy = beta/h(2)**2
//      dhz = beta/h(3)**2
//
//      res = 0.0D0
//
//      do n = 1, nc
//         do k = lo(3), hi(3)
//            do j = lo(2), hi(2)
//               do i = lo(1), hi(1)
//                  res = max(res, abs(alpha*a(i,j,k)
//     &                 + dhx*(bX(i+1,j,k) + bX(i,j,k))
//     &                 + dhy*(bY(i,j+1,k) + bY(i,j,k))
//     $                 + dhz*(bZ(i,j,k+1) + bZ(i,j,k)))
//     &                 + abs( -dhx*bX(i+1,j,k)) + abs( -dhx*bX(i,j,k))
//     &                 + abs( -dhy*bY(i,j+1,k)) + abs( -dhy*bY(i,j,k))
//     &                 + abs( -dhz*bZ(i,j,k+1)) + abs( -dhz*bZ(i,j,k)))
//               end do
//            end do
//         end do
//      end do
//
//      end