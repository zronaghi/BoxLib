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

//Average Functor:
struct C_AVERAGE_Functor {
    // Data used by the loop body
    FArrayBox* c;
    FArrayBox* f;
  
    // Constructor to initialize the data
    C_AVERAGE_Functor(FArrayBox* c_, const FArrayBox* f_){
        c=c_;
        f=const_cast<FArrayBox*>(f_);
    }

    // Loop body as an operator
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& n, const int& k, const int& j, const int& i) const {
        (*c)(IntVect(i,j,k),n) =  ((*f)(IntVect(2*i+1,2*j+1,2*k),n) + (*f)(IntVect(2*i,2*j+1,2*k),n) + (*f)(IntVect(2*i+1,2*j,2*k),n) + (*f)(IntVect(2*i,2*j,2*k),n))*0.125;
        (*c)(IntVect(i,j,k),n) += ((*f)(IntVect(2*i+1,2*j+1,2*k+1),n) + (*f)(IntVect(2*i,2*j+1,2*k+1),n) + (*f)(IntVect(2*i+1,2*j,2*k+1),n) + (*f)(IntVect(2*i,2*j,2*k+1),n))*0.125;
    }
};

//Average Kernel
void C_AVERAGE(
    const Box& bx,
const int nc,
FArrayBox& c,
const FArrayBox& f){
	
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
	
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>> t_policy;
    // Create a functor
    C_AVERAGE_Functor ave_functor(&c,&f);
    // Execute functor
    Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{1,4,4,1024000}),ave_functor);
}



//Interpolation Functor:
struct C_INTERP_Functor {
    // Data used by the loop body
    FArrayBox* f;
    FArrayBox* c;
  
    // Constructor to initialize the data
    C_INTERP_Functor(FArrayBox* f_, const FArrayBox* c_){
        f=f_;
        c=const_cast<FArrayBox*>(c_);
    }

    // Loop body as an operator
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& n, const int& k, const int& j, const int& i) const {
        (*f)(IntVect(2*i+1,2*j+1,2*k  ),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i  ,2*j+1,2*k  ),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i+1,2*j  ,2*k  ),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i  ,2*j  ,2*k  ),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i+1,2*j+1,2*k+1),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i  ,2*j+1,2*k+1),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i+1,2*j  ,2*k+1),n)       += (*c)(IntVect(i,j,k),n);
        (*f)(IntVect(2*i  ,2*j  ,2*k+1),n)       += (*c)(IntVect(i,j,k),n);
    }
};

//Interpolation Kernel
void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){
	
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
	
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>> t_policy;
    // Create a functor
    C_INTERP_Functor int_functor(&f,&c);
    // Execute functor
    Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{1,4,4,1024000}),int_functor);
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
//GSRB Functor
struct C_GSRB_Functor {
    // Data used by the loop body
    int rb, lo, hi, *blo, *bhi;
    Real alpha, beta;
    FArrayBox *phi, *rhs, *a, *bX, *bY, *bZ;
    FArrayBox *f0, *f1, *f2, *f3, *f4, *f5;
    Mask *m0, *m1, *m2, *m3, *m4, *m5;
    Real* h;
    Real omega, dhx, dhy, dhz;
  
    // Constructor to initialize the data
    C_GSRB_Functor(
    const int rb_,
    const int lo_,
    const int hi_,
    const Real alpha_,
    const Real beta_,
    FArrayBox* phi_,
    const FArrayBox* rhs_,
    const FArrayBox* a_,
    const FArrayBox* bX_,
    const FArrayBox* bY_,
    const FArrayBox* bZ_,
    const FArrayBox* f0_,
    const Mask* m0_,
    const FArrayBox* f1_,
    const Mask* m1_,
    const FArrayBox* f2_,
    const Mask* m2_,
    const FArrayBox* f3_,
    const Mask* m3_,
    const FArrayBox* f4_,
    const Mask* m4_,
    const FArrayBox* f5_,
    const Mask* m5_,
    const Real* h_,
    const int* blo_,
    const int* bhi_) : rb(rb_), lo(lo_), hi(hi_), alpha(alpha_), beta(beta_)
    {
		
        phi=phi_;
        rhs=const_cast<FArrayBox*>(rhs_);
        a=const_cast<FArrayBox*>(a_);
        bX=const_cast<FArrayBox*>(bX_);
        bY=const_cast<FArrayBox*>(bY_);
        bZ=const_cast<FArrayBox*>(bZ_);
        f0=const_cast<FArrayBox*>(f0_);
        f1=const_cast<FArrayBox*>(f1_);
        f2=const_cast<FArrayBox*>(f2_);
        f3=const_cast<FArrayBox*>(f3_);
        f4=const_cast<FArrayBox*>(f4_);
        f5=const_cast<FArrayBox*>(f5_);
        m0=const_cast<Mask*>(m0_);
        m1=const_cast<Mask*>(m1_);
        m2=const_cast<Mask*>(m2_);
        m3=const_cast<Mask*>(m3_);
        m4=const_cast<Mask*>(m4_);
        m5=const_cast<Mask*>(m5_);
        h=const_cast<Real*>(h_);
        blo=const_cast<int*>(blo_);
        bhi=const_cast<int*>(bhi_);
        
        //some parameters
        omega= 1.15;
        dhx = beta/(h[0]*h[0]);
        dhy = beta/(h[1]*h[1]);
        dhz = beta/(h[2]*h[2]);
    }

    // Loop body as an operator
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& n, const int& k, const int& j) const {
        int ioff = (lo + j + k + rb)%2;
        for (int i = lo + ioff; i <= hi; i+=2) {
        
            //BC terms
            Real cf0 = ( (i==blo[0]) && ((*m0)(IntVect(blo[0]-1,j,k))>0) ? (*f0)(IntVect(blo[0],j,k)) : 0. );
            Real cf1 = ( (j==blo[1]) && ((*m1)(IntVect(i,blo[1]-1,k))>0) ? (*f1)(IntVect(i,blo[1],k)) : 0. );
            Real cf2 = ( (k==blo[2]) && ((*m2)(IntVect(i,j,blo[2]-1))>0) ? (*f2)(IntVect(i,j,blo[2])) : 0. );
            Real cf3 = ( (i==bhi[0]) && ((*m3)(IntVect(bhi[0]+1,j,k))>0) ? (*f3)(IntVect(bhi[0],j,k)) : 0. );
            Real cf4 = ( (j==bhi[1]) && ((*m4)(IntVect(i,bhi[1]+1,k))>0) ? (*f4)(IntVect(i,bhi[1],k)) : 0. );
            Real cf5 = ( (k==bhi[2]) && ((*m5)(IntVect(i,j,bhi[2]+1))>0) ? (*f5)(IntVect(i,j,bhi[2])) : 0. );
		
            //assign ORA constants
            double gamma = alpha * (*a)(IntVect(i,j,k))
                        + dhx * ((*bX)(IntVect(i,j,k)) + (*bX)(IntVect(i+1,j,k)))
                        + dhy * ((*bY)(IntVect(i,j,k)) + (*bY)(IntVect(i,j+1,k)))
                        + dhz * ((*bZ)(IntVect(i,j,k)) + (*bZ)(IntVect(i,j,k+1)));
		
            double g_m_d = gamma
                        - dhx * ((*bX)(IntVect(i,j,k))*cf0 + (*bX)(IntVect(i+1,j,k))*cf3)
                        - dhy * ((*bY)(IntVect(i,j,k))*cf1 + (*bY)(IntVect(i,j+1,k))*cf4)
                        - dhz * ((*bZ)(IntVect(i,j,k))*cf2 + (*bZ)(IntVect(i,j,k+1))*cf5);
		
            double rho =  dhx * ((*bX)(IntVect(i,j,k))*(*phi)(IntVect(i-1,j,k),n) + (*bX)(IntVect(i+1,j,k))*(*phi)(IntVect(i+1,j,k),n))
                        + dhy * ((*bY)(IntVect(i,j,k))*(*phi)(IntVect(i,j-1,k),n) + (*bY)(IntVect(i,j+1,k))*(*phi)(IntVect(i,j+1,k),n))
                        + dhz * ((*bZ)(IntVect(i,j,k))*(*phi)(IntVect(i,j,k-1),n) + (*bZ)(IntVect(i,j,k+1))*(*phi)(IntVect(i,j,k+1),n));
		
            double res = (*rhs)(IntVect(i,j,k),n) - gamma * (*phi)(IntVect(i,j,k),n) + rho;
            (*phi)(IntVect(i,j,k),n) += omega/g_m_d * res;
        }
    }
};

//GSRB kernel
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
	
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>> t_policy;
    // Create a functor
    C_GSRB_Functor gsrb_functor(rb,lo[0],hi[0],alpha,beta,&phi,&rhs,&a,&bX,&bY,&bZ,&f0,&m0,&f1,&m1,&f2,&m2,&f3,&m3,&f4,&m4,&f5,&m5,h,blo,bhi);
    // Execute functor
    Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1]},{nc,hi[2]+1,hi[1]+1},{1,4,4}),gsrb_functor);
}

//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
//adotx functor
struct C_ADOTX_Functor {
    // Data used by the loop body
    FArrayBox *y, *x, *a, *bX, *bY, *bZ;
    Real* h;
    Real alpha, beta, dhx, dhy, dhz;
  
    // Constructor to initialize the data
    C_ADOTX_Functor(Real alpha_, Real beta_, FArrayBox* y_, const FArrayBox* x_, const FArrayBox* a_, const FArrayBox* bX_, const FArrayBox* bY_, const FArrayBox* bZ_, const Real* h_) : alpha(alpha_), beta(beta_){
        y=y_;
        x=const_cast<FArrayBox*>(x_);
        a=const_cast<FArrayBox*>(a_);
        bX=const_cast<FArrayBox*>(bX_);
        bY=const_cast<FArrayBox*>(bY_);
        bZ=const_cast<FArrayBox*>(bZ_);
        h=const_cast<Real*>(h_);
        
        //some parameters
        dhx = beta/(h[0]*h[0]);
        dhy = beta/(h[1]*h[1]);
        dhz = beta/(h[2]*h[2]);
    }

    // Loop body as an operator
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& n, const int& k, const int& j, const int& i) const {
        (*y)(IntVect(i,j,k),n) = alpha*(*a)(IntVect(i,j,k))*(*x)(IntVect(i,j,k),n)
                            - dhx * (   (*bX)(IntVect(i+1,j,  k  )) * ( (*x)(IntVect(i+1,j,  k),  n) - (*x)(IntVect(i,  j,  k  ),n) )
                                      - (*bX)(IntVect(i,  j,  k  )) * ( (*x)(IntVect(i,  j,  k),  n) - (*x)(IntVect(i-1,j,  k  ),n) ) 
                                    )
                            - dhy * (   (*bY)(IntVect(i,  j+1,k  )) * ( (*x)(IntVect(i,  j+1,k),  n) - (*x)(IntVect(i,  j  ,k  ),n) )
                                      - (*bY)(IntVect(i,  j,  k  )) * ( (*x)(IntVect(i,  j,  k),  n) - (*x)(IntVect(i,  j-1,k  ),n) )
                                    )
                            - dhz * (   (*bZ)(IntVect(i,  j,  k+1)) * ( (*x)(IntVect(i,  j,  k+1),n) - (*x)(IntVect(i,  j  ,k  ),n) )
                                      - (*bZ)(IntVect(i,  j,  k  )) * ( (*x)(IntVect(i,  j,  k),  n) - (*x)(IntVect(i,  j,  k-1),n) )
                                    );
    }
};

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
	
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,Kokkos::Experimental::Iterate::Right,Kokkos::Experimental::Iterate::Right>> t_policy;
    // Create a functor
    C_ADOTX_Functor adotx_functor(alpha,beta,&y,&x,&a,&bX,&bY,&bZ,h);
    // Execute functor
    Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{1,4,4,1024000}),adotx_functor);
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
	
    //some parameters
    Real dhx = beta/(h[0]*h[0]);
    Real dhy = beta/(h[1]*h[1]);
    Real dhz = beta/(h[2]*h[2]);
	
    //initialize to zero
    res = 0.0;

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
