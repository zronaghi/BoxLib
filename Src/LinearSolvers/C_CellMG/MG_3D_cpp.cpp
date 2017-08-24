#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>
#include <iostream>
#include <LO_BCTYPES.H>

//Average Functor
struct C_AVERAGE_FUNCTOR{
public:
    C_AVERAGE_FUNCTOR(const FArrayBox& c_, const FArrayBox& f_) :
      cv(c_.view_fab), fv(f_.view_fab)
    {
      //fv.syncH2D();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int i, const int j, const int k, const int n) const{
      cv(i,j,k,n) =  (fv(2*i+1,2*j+1,2*k,n) + fv(2*i,2*j+1,2*k,n) + fv(2*i+1,2*j,2*k,n) + fv(2*i,2*j,2*k,n))*0.125;
      cv(i,j,k,n) += (fv(2*i+1,2*j+1,2*k+1,n) + fv(2*i,2*j+1,2*k+1,n) + fv(2*i+1,2*j,2*k+1,n) + fv(2*i,2*j,2*k+1,n))*0.125;
    }

    void fill(){
      //cv.syncD2H();
    }
private:
    ViewFab<Real> cv, fv;
};

//Average Kernel
void C_AVERAGE(
  const Box& bx,
  const int nc,
  FArrayBox& c,
  const FArrayBox& f)
{
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int* cb = bx.cbVect();

    //create functor
    C_AVERAGE_FUNCTOR cavfunc(c,f);

    //execute
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), cavfunc);

    //write back
    cavfunc.fill();
}


//Interpolation Functor
struct C_INTERP_FUNCTOR{
public:
    C_INTERP_FUNCTOR(const FArrayBox& f_, const FArrayBox& c_)
      : fv(f_.view_fab), cv(c_.view_fab)
    {
        //fv.syncH2D();
        //cv.syncH2D();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int i, const int j, const int k, const int n) const{
        fv(2*i+1,2*j+1,2*k  ,n)       += cv(i,j,k,n);
        fv(2*i  ,2*j+1,2*k  ,n)       += cv(i,j,k,n);
        fv(2*i+1,2*j  ,2*k  ,n)       += cv(i,j,k,n);
        fv(2*i  ,2*j  ,2*k  ,n)       += cv(i,j,k,n);
        fv(2*i+1,2*j+1,2*k+1,n)       += cv(i,j,k,n);
        fv(2*i  ,2*j+1,2*k+1,n)       += cv(i,j,k,n);
        fv(2*i+1,2*j  ,2*k+1,n)       += cv(i,j,k,n);
        fv(2*i  ,2*j  ,2*k+1,n)       += cv(i,j,k,n);
    }

    void fill(){
        //fv.syncD2H();
    }
private:
    ViewFab<Real> fv, cv;
};

void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){

    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int* cb = bx.cbVect();

    //create functor
    C_INTERP_FUNCTOR cintfunc(f,c);

    // Execute functor
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc},{cb[0], cb[1], cb[2], nc}), cintfunc);

    //write back
    cintfunc.fill();
}


//-----------------------------------------------------------------------
//
//     Gauss-SeidelRed-Black (GSRB):
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
//GSRB functor
struct C_GSRB_FUNCTOR{
public:
    C_GSRB_FUNCTOR(const Box& bx_,
    const Box& bbx_,
    const int& rb_,
    const Real& alpha_,
    const Real& beta_,
    const FArrayBox& phi_,
    const FArrayBox& rhs_,
    const FArrayBox& a_,
    const FArrayBox& bX_,
    const FArrayBox& bY_,
    const FArrayBox& bZ_,
    const FArrayBox& f0_,
    const Mask& m0_,
    const FArrayBox& f1_,
    const Mask& m1_,
    const FArrayBox& f2_,
    const Mask& m2_,
    const FArrayBox& f3_,
    const Mask& m3_,
    const FArrayBox& f4_,
    const Mask& m4_,
    const FArrayBox& f5_,
    const Mask& m5_,
    const Real* h) :
    phiv(phi_.view_fab),
    rhsv(rhs_.view_fab),
    av(a_.view_fab),
    bXv(bX_.view_fab),
    bYv(bY_.view_fab),
    bZv(bZ_.view_fab),
    f0v(f0_.view_fab),
    f1v(f1_.view_fab),
    f2v(f2_.view_fab),
    f3v(f3_.view_fab),
    f4v(f4_.view_fab),
    f5v(f5_.view_fab),
    m0v(m0_.view_fab),
    m1v(m1_.view_fab),
    m2v(m2_.view_fab),
    m3v(m3_.view_fab),
    m4v(m4_.view_fab),
    m5v(m5_.view_fab),
    rb(rb_),
    comp(0),
    bx(bx_),
    bbx(bbx_),
    alpha(alpha_),
    beta(beta_)
  {
    //phiv.syncH2D();
    ////these should not be needed:
    //rhsv.syncH2D();
    //av.syncH2D();
    //bXv.syncH2D();
    //bYv.syncH2D();
    //bZv.syncH2D();
    //f0v.syncH2D();
    //f1v.syncH2D();
    //f2v.syncH2D();
    //f3v.syncH2D();
    //f4v.syncH2D();
    //f5v.syncH2D();
    //m0v.syncH2D();
    //m1v.syncH2D();
    //m2v.syncH2D();
    //m3v.syncH2D();
    //m4v.syncH2D();
    //m5v.syncH2D();
      
    //some parameters
    omega= 1.15;
    dhx = beta/(h[0]*h[0]);
    dhy = beta/(h[1]*h[1]);
    dhz = beta/(h[2]*h[2]);

    lo0=bx.loVect()[0];
    hi0=bx.hiVect()[0];

    for(unsigned int d=0; d<3; d++){
        blo[d]=bbx.smallEnd()[d];
        bhi[d]=bbx.bigEnd()[d];
    }
  }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int ii, const int j, const int k, const int n) const{
        int ioff = (lo0 + j + k + rb) % 2;
        int i = 2 * (ii-lo0) + lo0 + ioff;

        //be careful to not run over
        if(i<=hi0){

            //BC terms
            Real cf0 = ( (i==blo[0]) && (m0v(blo[0]-1,j,k)>0) ? f0v(blo[0],j,k) : 0. );
            Real cf1 = ( (j==blo[1]) && (m1v(i,blo[1]-1,k)>0) ? f1v(i,blo[1],k) : 0. );
            Real cf2 = ( (k==blo[2]) && (m2v(i,j,blo[2]-1)>0) ? f2v(i,j,blo[2]) : 0. );
            Real cf3 = ( (i==bhi[0]) && (m3v(bhi[0]+1,j,k)>0) ? f3v(bhi[0],j,k) : 0. );
            Real cf4 = ( (j==bhi[1]) && (m4v(i,bhi[1]+1,k)>0) ? f4v(i,bhi[1],k) : 0. );
            Real cf5 = ( (k==bhi[2]) && (m5v(i,j,bhi[2]+1)>0) ? f5v(i,j,bhi[2]) : 0. );

            //assign ORA constants
            double gamma = alpha * av(i,j,k)
                + dhx * (bXv(i,j,k) + bXv(i+1,j,k))
                    + dhy * (bYv(i,j,k) + bYv(i,j+1,k))
                        + dhz * (bZv(i,j,k) + bZv(i,j,k+1));

            double g_m_d = gamma
                - dhx * (bXv(i,j,k)*cf0 + bXv(i+1,j,k)*cf3)
                    - dhy * (bYv(i,j,k)*cf1 + bYv(i,j+1,k)*cf4)
                        - dhz * (bZv(i,j,k)*cf2 + bZv(i,j,k+1)*cf5);

            double rho =  dhx * (bXv(i,j,k)*phiv(i-1,j,k,n) + bXv(i+1,j,k)*phiv(i+1,j,k,n))
                + dhy * (bYv(i,j,k)*phiv(i,j-1,k,n) + bYv(i,j+1,k)*phiv(i,j+1,k,n))
                    + dhz * (bZv(i,j,k)*phiv(i,j,k-1,n) + bZv(i,j,k+1)*phiv(i,j,k+1,n));

            double res = rhsv(i,j,k,n) - gamma * phiv(i,j,k,n) + rho;
            phiv(i,j,k,n) += omega/g_m_d * res;
        }
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int j, const int k, const int n) const{
        int ioff = (lo0 + j + k + rb) % 2;

        for(int i=ioff; i<=hi0; i+=2){

            //BC terms
            Real cf0 = ( (i==blo[0]) && (m0v(blo[0]-1,j,k)>0) ? f0v(blo[0],j,k) : 0. );
            Real cf1 = ( (j==blo[1]) && (m1v(i,blo[1]-1,k)>0) ? f1v(i,blo[1],k) : 0. );
            Real cf2 = ( (k==blo[2]) && (m2v(i,j,blo[2]-1)>0) ? f2v(i,j,blo[2]) : 0. );
            Real cf3 = ( (i==bhi[0]) && (m3v(bhi[0]+1,j,k)>0) ? f3v(bhi[0],j,k) : 0. );
            Real cf4 = ( (j==bhi[1]) && (m4v(i,bhi[1]+1,k)>0) ? f4v(i,bhi[1],k) : 0. );
            Real cf5 = ( (k==bhi[2]) && (m5v(i,j,bhi[2]+1)>0) ? f5v(i,j,bhi[2]) : 0. );

            //assign ORA constants
            double gamma = alpha * av(i,j,k)
                + dhx * (bXv(i,j,k) + bXv(i+1,j,k))
                    + dhy * (bYv(i,j,k) + bYv(i,j+1,k))
                        + dhz * (bZv(i,j,k) + bZv(i,j,k+1));

            double g_m_d = gamma
                - dhx * (bXv(i,j,k)*cf0 + bXv(i+1,j,k)*cf3)
                    - dhy * (bYv(i,j,k)*cf1 + bYv(i,j+1,k)*cf4)
                        - dhz * (bZv(i,j,k)*cf2 + bZv(i,j,k+1)*cf5);

            double rho =  dhx * (bXv(i,j,k)*phiv(i-1,j,k,n) + bXv(i+1,j,k)*phiv(i+1,j,k,n))
                + dhy * (bYv(i,j,k)*phiv(i,j-1,k,n) + bYv(i,j+1,k)*phiv(i,j+1,k,n))
                    + dhz * (bZv(i,j,k)*phiv(i,j,k-1,n) + bZv(i,j,k+1)*phiv(i,j,k+1,n));

            double res = rhsv(i,j,k,n) - gamma * phiv(i,j,k,n) + rho;
            phiv(i,j,k,n) += omega/g_m_d * res;
        }
    }

    void fill(){
        //phiv.syncD2H();
    }

private:
    ViewFab<Real> phiv, rhsv, av, bXv, bYv, bZv;
    ViewFab<Real> f0v, f1v, f2v, f3v, f4v, f5v;
    ViewFab<int> m0v, m1v, m2v, m3v, m4v, m5v;
    Box bx, bbx;
    int rb, comp;
    Real alpha, beta, dhx, dhy, dhz, omega;
    int lo0, hi0;
    int blo[3], bhi[3];
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
    //cacheblock
    const int *cb = bx.cbVect();


    //create functors
    C_GSRB_FUNCTOR cgsrbfunc(bx, bbx, rb, alpha, beta, phi, rhs, a, bX, bY, bZ, f0, m0, f1, m1, f2, m2, f3, m3, f4, m4, f5, m5, h);

#if 1
    //execute
    Kokkos::fence();
    double start_time = omp_get_wtime();
    int length0 = std::floor( (hi[0]-lo[0]+1) / 2 );
    int up0 = lo[0] + length0;
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {up0+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), cgsrbfunc);
    Kokkos::fence();
    double end_time =  omp_get_wtime();
#else
    //execute
    Kokkos::fence();
    double start_time = omp_get_wtime();
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), cgsrbfunc);
    Kokkos::fence();
    double end_time =  omp_get_wtime();
#endif
    //std::cout << "GSRB Elapsed time: " << end_time - start_time << std::endl;

    //copy data back from the views
    cgsrbfunc.fill();
}


//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
//ADOTX Functor
struct C_ADOTX_FUNCTOR{
    
public:

    C_ADOTX_FUNCTOR(const FArrayBox& y_,
    const FArrayBox& x_,
    const Real& alpha_,
    const Real& beta_,
    const FArrayBox& a_,
    const FArrayBox& bX_,
    const FArrayBox& bY_,
    const FArrayBox& bZ_,
    const Real* h) :
    yv(y_.view_fab),
    xv(x_.view_fab),
    av(a_.view_fab),
    bXv(bX_.view_fab),
    bYv(bY_.view_fab),
    bZv(bZ_.view_fab),
    alpha(alpha_),
    beta(beta_) {
        
        //av.syncH2D();
        //xv.syncH2D();
        //bXv.syncH2D();
        //bYv.syncH2D();
        //bZv.syncH2D();

        //helpers
        dhx = beta/(h[0]*h[0]);
        dhy = beta/(h[1]*h[1]);
        dhz = beta/(h[2]*h[2]);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int i, const int j, const int k, const int n) const{
      yv(i,j,k,n) = alpha * av(i,j,k) * xv(i,j,k,n)
            - dhx * ( bXv(i+1,j  ,k  ) * ( xv(i+1,j  ,k  ,n) - xv(i,  j,  k,  n) ) - bXv(i,  j,  k  ) * ( xv (i,  j,  k, n) - xv(i-1,j  ,k  ,n) ) )
            - dhy * ( bYv(i,  j+1,k  ) * ( xv(i  ,j+1,k  ,n) - xv(i,  j,  k,  n) ) - bYv(i,  j,  k  ) * ( xv (i,  j,  k, n) - xv(i  ,j-1,k  ,n) ) )
            - dhz * ( bZv(i,  j  ,k+1) * ( xv(i  ,j  ,k+1,n) - xv(i,  j,  k,  n) ) - bZv(i,  j,  k  ) * ( xv (i,  j,  k, n) - xv(i  ,j  ,k-1,n) ) );
    }

    void fill(){
        //yv.syncD2H();
    }

private:
    ViewFab<Real> yv, xv, av, bXv, bYv, bZv;
    Box bx;
    int nc;
    Real alpha, beta, dhx, dhy, dhz;
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
    const int *cb = bx.cbVect();

    //create functor
    C_ADOTX_FUNCTOR cadxfunc(y,x,alpha,beta,a,bX,bY,bZ,h);

    //execute
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), cadxfunc);

    //write back result
    cadxfunc.fill();
}

//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
//ADOTX Functor
struct C_NORMA_FUNCTOR{
    
public:

    C_NORMA_FUNCTOR(const Real& alpha_,
    const Real& beta_,
    const FArrayBox& a_,
    const FArrayBox& bX_,
    const FArrayBox& bY_,
    const FArrayBox& bZ_,
    const Real* h) :
    av(a_.view_fab),
    bXv(bX_.view_fab),
    bYv(bY_.view_fab),
    bZv(bZ_.view_fab),
    alpha(alpha_),
    beta(beta_) {
        
        //av.syncH2D();
        //bXv.syncH2D();
        //bYv.syncH2D();
        //bZv.syncH2D();

        //helpers
        dhx = beta/(h[0]*h[0]);
        dhy = beta/(h[1]*h[1]);
        dhz = beta/(h[2]*h[2]);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void operator()(const int i, const int j, const int k, const int n, Real& tmpres) const{
        //first part:
        Real tmpval = std::abs( alpha*av(i,j,k)
                    + dhx * ( bXv(i+1,j,k) + bXv(i,j,k) )
                    + dhy * ( bYv(i,j+1,k) + bYv(i,j,k) )
                    + dhz * ( bZv(i,j,k+1) + bZv(i,j,k) ) );

        //add the rest
        tmpval +=  std::abs( dhx * bXv(i+1,j,k) ) + std::abs( dhx * bXv(i,j,k) )
                 + std::abs( dhy * bYv(i,j+1,k) ) + std::abs( dhy * bYv(i,j,k) )
                 + std::abs( dhz * bZv(i,j,k+1) ) + std::abs( dhz * bZv(i,j,k) );

        //max:
        tmpres = Kokkos::max2(tmpres, tmpval);
    }

    using value_type = Real;
    using execution_space = devspace;

    //Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src)  const {
      if ( src > dest )
        dest = src;
    }
  
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dest, const volatile value_type& src) const {
      if ( src > dest )
        dest = src;
    }

    //Required
    KOKKOS_INLINE_FUNCTION
    void init( value_type& val)  const {
      val = 0.0;
    }

private:    
    ViewFab<Real> av, bXv, bYv, bZv;
    Box bx;
    int nc;
    Real alpha, beta, dhx, dhy, dhz;
};

    //NORMA kernel
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
    const int *cb = bx.cbVect();

    //initialize to zero
    res = 0.0;

    //create functor
    C_NORMA_FUNCTOR cnormafunc(alpha,beta,a,bX,bY,bZ,h);

    //execute
    Kokkos::parallel_reduce(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}),
        cnormafunc,
        res);
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

static bool is_dirichlet(int i){ return (i == LO_DIRICHLET); }
static bool is_neumann(int i){ return (i == LO_NEUMANN); }

void polyInterpCoeff(const Real& xInt, const oarray<Real>& x, const int N, oarray<Real>& c){
    Real num, den;
    //correct offset in order to correct w/ respect to fortran
    int off = x.getOffset()-1;
    
    //std::cout << "N: " << N <<std::endl;
    
    for(int j=1; j<=N; j++){
        num = 1.;
        den = 1.;
        for(int i=1; i<=j-1; i++){
            num *= (xInt - x[i+off]);
            den *= (x[j+off] - x[i+off]);
            
            //std::cout << "num upper: " << i+1 << " " << j+1 << " " << num << std::endl;
            //std::cout << "den upper: " << i+1 << " " << j+1 << " " << den << std::endl;
        }
        for(int i=j+1; i<=N; i++){
            num *= (xInt - x[i+off]);
            den *= (x[j+off] - x[i+off]);
            
            //std::cout << "num lower: " << i+1 << " " << j+1 << " " << num << std::endl;
            //std::cout << "den lower: " << i+1 << " " << j+1 << " " << den << std::endl;
        }
        c[j+off] = num/den;
    }
    return;
}

void C_APPLYBC (
    const Box& bx,
    const int numcomp,
    const int src_comp,
    const int bndry_comp,
    int flagden, 
    int flagbc, 
    int maxorder,
    FArrayBox& phi,
    int cdir, 
    int bct, 
    int bcl,
    FArrayBox& bcval,
    const Mask& mask,
    FArrayBox& den,
const Real* h)
{
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int* cb = bx.cbVect();
    
    //iteration policies
     
    const int maxmaxorder=4;
    const Real xInt = -0.5;
    int Lmaxorder;
    oarray<Real> x(-1,maxmaxorder-2);
    oarray<Real> coef(-1,maxmaxorder-2);
    
    if ( maxorder == -1 ){
        Lmaxorder = maxmaxorder;
    }
    else{
        Lmaxorder = std::min(maxorder,maxmaxorder);
    }
    int lenx = std::min(hi[0]-lo[0], Lmaxorder-2);
    int leny = std::min(hi[1]-lo[1], Lmaxorder-2);
    int lenz = std::min(hi[2]-lo[2], Lmaxorder-2);
    
    for(int m=0; m<=maxmaxorder-2; m++){
       x[m] = Real(m) + 0.5;
    }
    
    ViewFab<Real> phiv = phi.view_fab;
    ViewFab<Real> bcvalv = bcval.view_fab;
    ViewFab<Real> denv = den.view_fab;
    ViewFab<int> maskv = mask.view_fab;
    
    //+/- X
    if (cdir==0 || cdir == 3){
        int comp = (cdir==0 ? lo[0] : hi[0]);
        int comps = (cdir==0 ? lo[0]-1 : hi[0]+1);
        int sign = (cdir==0 ? +1 : -1);
        
        if (is_neumann(bct)){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, numcomp}, {cb[1], cb[2], numcomp}),
            KOKKOS_LAMBDA(const int j, const int k, const int n){
                const int nsrc=n+src_comp;
                phiv(comps,j,k,nsrc) = (maskv(comps,j,k) > 0 ? phiv(comp,j,k,nsrc) : phiv(comps,j,k,nsrc));
            });
           if ( flagden == 1){
               Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[1], lo[2]}, {hi[1]+1, hi[2]+1}, {cb[1], cb[2]}),
               KOKKOS_LAMBDA(const int j, const int k){
                   denv(comp,j,k) = 1.;
               });
            }
        }
        else if(is_dirichlet(bct)){
            x[-1] = - bcl/h[0];
            polyInterpCoeff(xInt, x, lenx+2, coef);
            if ( flagbc == 1 ){
                const Real cf = coef[-1];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, numcomp}, {cb[1], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int j, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(comps, j, k, nsrc) = (maskv(comps,j,k) > 0 ? bcvalv(comps,j,k,n+bndry_comp)*cf : phiv(comps, j, k, nsrc));
                });
            }
            else{
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, numcomp}, {cb[1], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int j, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(comps, j, k, nsrc) = (maskv(comps,j,k) > 0 ? 0. : phiv(comps, j, k, nsrc));
                });
            }
            for(int m = 0; m<=lenx; m++){
                const Real cf = coef[m];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, numcomp}, {cb[1], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int j, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(comps, j, k, nsrc) = (maskv(comps,j,k) > 0 ? phiv(comps,j,k,nsrc) + phiv(comp+sign*m, j, k, nsrc)*cf : phiv(comps, j, k, nsrc));
                });
            }
            if ( flagden == 1){
                const Real cf = coef[0];
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[1], lo[2]}, {hi[1]+1, hi[2]+1}, {cb[1], cb[2]}),
                KOKKOS_LAMBDA(const int j, const int k){
                    denv(comp,j,k) = (maskv(comps,j,k)>0 ? cf : 0.);
                });
             }
        }
        else if ( bct == LO_REFLECT_ODD ){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[1], lo[2], 0}, {hi[1]+1, hi[2]+1, numcomp}, {cb[1], cb[2], numcomp}),
            KOKKOS_LAMBDA(const int j, const int k, const int n){
                const int nsrc = n+src_comp;
                phiv(comps, j, k, nsrc) = (maskv(comps,j,k) > 0 ? -phiv(comp, j, k, nsrc) : phiv(comps, j, k, nsrc) );
            });
            if ( flagden == 1){
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[1], lo[2]}, {hi[1]+1, hi[2]+1}, {cb[1], cb[2]}),
                KOKKOS_LAMBDA(const int j, const int k){
                    denv(comp,j,k) = (maskv(comps,j,k)>0 ? -1. : 0.);
                });
             }
        }
        else{
            BoxLib::Error("UNKNOWN BC ON LEFT FACE IN APPLYBC");
        }
    }
    
    //+/- Y
    if (cdir==1 || cdir == 4){
        int comp = (cdir==1 ? lo[1] : hi[1]);
        int comps = (cdir==1 ? lo[1]-1 : hi[1]+1);
        int sign = (cdir==1 ? +1 : -1);
        
        if (is_neumann(bct)){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[2], 0}, {hi[0]+1, hi[2]+1, numcomp}, {cb[0], cb[2], numcomp}),
            KOKKOS_LAMBDA(const int i, const int k, const int n){
                const int nsrc=n+src_comp;
                phiv(i,comp,k,nsrc) = (maskv(i,comps,k) > 0 ? phiv(i,comp,k,nsrc) : phiv(i,comps,k,nsrc));
            });
           if ( flagden == 1){
               Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[2]}, {hi[0]+1, hi[2]+1}, {cb[0], cb[2]}),
               KOKKOS_LAMBDA(const int i, const int k){
                   denv(i,comp,k) = 1.;
               });
            }
        }
        else if(is_dirichlet(bct)){
            x[-1] = - bcl/h[1];
            polyInterpCoeff(xInt, x, leny+2, coef);
            if ( flagbc == 1 ){
                const Real cf = coef[-1];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[2], 0}, {hi[0]+1, hi[2]+1, numcomp}, {cb[0], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int i, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,comps,k,nsrc) = (maskv(i,comps,k) > 0 ? bcvalv(i,comps,k,n+bndry_comp)*cf : phiv(i,comps,k,nsrc));
                });
            }
            else{
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[2], 0}, {hi[0]+1, hi[2]+1, numcomp}, {cb[0], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int i, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,comps,k,nsrc) = (maskv(i,comps,k) > 0 ? 0. : phiv(i,comps,k,nsrc));
                });
            }
            for(int m = 0; m<=leny; m++){
                const double cf = coef[m];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[2], 0}, {hi[0]+1, hi[2]+1, numcomp}, {cb[0], cb[2], numcomp}),
                KOKKOS_LAMBDA(const int i, const int k, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,comps,k,nsrc) = (maskv(i,comps,k) > 0 ? phiv(i,comps,k,nsrc) + phiv(i,comp+sign*m,k,nsrc)*cf : phiv(i,comps,k,nsrc));
                });
            }
            if ( flagden == 1){
                const Real cf = coef[0];
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[2]}, {hi[0]+1, hi[2]+1}, {cb[0], cb[2]}),
                KOKKOS_LAMBDA(const int i, const int k){
                    denv(i,comp,k) = (maskv(i,comps,k)>0 ? cf : 0.);
                });
             }
        }
        else if ( bct == LO_REFLECT_ODD ){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[2], 0}, {hi[0]+1, hi[2]+1, numcomp}, {cb[0], cb[2], numcomp}),
            KOKKOS_LAMBDA(const int i, const int k, const int n){
                const int nsrc = n+src_comp;
                phiv(i,comps,k,nsrc) = (maskv(i,comps,k) > 0 ? -phiv(i,comp,k,nsrc) : phiv(i,comps,k,nsrc) );
            });
            if ( flagden == 1){
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[2]}, {hi[0]+1, hi[2]+1}, {cb[0], cb[2]}),
                KOKKOS_LAMBDA(const int i, const int k){
                    denv(i,comp,k) = (maskv(i,comps,k)>0 ? -1. : 0.);
                });
             }
        }
        else{
            BoxLib::Error("UNKNOWN BC ON LEFT FACE IN APPLYBC");
        }
    }
    
    //+/- Z
    if (cdir==2 || cdir == 5){
        int comp = (cdir==2 ? lo[2] : hi[2]);
        int comps = (cdir==2 ? lo[2]-1 : hi[2]+1);
        int sign = (cdir==2 ? +1 : -1);
        
        if (is_neumann(bct)){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[1], 0}, {hi[0]+1, hi[1]+1, numcomp}, {cb[0], cb[1], numcomp}),
            KOKKOS_LAMBDA(const int i, const int j, const int n){
                const int nsrc=n+src_comp;
                phiv(i,j,comp,nsrc) = (maskv(i,j,comps) > 0 ? phiv(i,j,comp,nsrc) : phiv(i,j,comps,nsrc));
            });
           if ( flagden == 1){
               Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[1]}, {hi[0]+1, hi[1]+1}, {cb[0], cb[1]}),
               KOKKOS_LAMBDA(const int i, const int j){
                   denv(i,j,comp) = 1.;
               });
            }
        }
        else if(is_dirichlet(bct)){
            x[-1] = - bcl/h[2];
            polyInterpCoeff(xInt, x, lenz+2, coef);
            if ( flagbc == 1 ){
                const Real cf = coef[-1];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[1], 0}, {hi[0]+1, hi[1]+1, numcomp}, {cb[0], cb[1], numcomp}),
                KOKKOS_LAMBDA(const int i, const int j, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,j,comps,nsrc) = (maskv(i,j,comps) > 0 ? bcvalv(i,j,comps,n+bndry_comp)*cf : phiv(i,j,comps,nsrc));
                });
            }
            else{
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[1], 0}, {hi[0]+1, hi[1]+1, numcomp}, {cb[0], cb[1], numcomp}),
                KOKKOS_LAMBDA(const int i, const int j, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,j,comps,nsrc) = (maskv(i,j,comps) > 0 ? 0. : phiv(i,j,comps,nsrc));
                });
            }
            for(int m = 0; m<=lenz; m++){
                const Real cf = coef[m];
                Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[1], 0}, {hi[0]+1, hi[1]+1, numcomp}, {cb[0], cb[1], numcomp}),
                KOKKOS_LAMBDA(const int i, const int j, const int n){
                    const int nsrc=n+src_comp;
                    phiv(i,j,comps,nsrc) = (maskv(i,j,comps) > 0 ? phiv(i,j,comps,nsrc) + phiv(i,j,comp+sign*m,nsrc)*cf : phiv(i,j,comps,nsrc));
                });
            }
            if ( flagden == 1){
                const Real cf = coef[0];
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[1]}, {hi[0]+1, hi[1]+1}, {cb[0], cb[1]}),
                KOKKOS_LAMBDA(const int i, const int j){
                    denv(i,j,comp) = (maskv(i,j,comps)>0 ? cf : 0.);
                });
             }
        }
        else if ( bct == LO_REFLECT_ODD ){
            Kokkos::Experimental::md_parallel_for(mdpolicy<3>({lo[0], lo[1], 0}, {hi[0]+1, hi[1]+1, numcomp}, {cb[0], cb[1], numcomp}),
            KOKKOS_LAMBDA(const int i, const int j, const int n){
                const int nsrc = n+src_comp;
                phiv(i,j,comps,nsrc) = (maskv(i,j,comps) > 0 ? -phiv(i,j,comp,nsrc) : phiv(i,j,comps,nsrc) );
            });
            if ( flagden == 1){
                Kokkos::Experimental::md_parallel_for(mdpolicy<2>({lo[0], lo[1]}, {hi[0]+1, hi[1]+1}, {cb[0], cb[1]}),
                KOKKOS_LAMBDA(const int i, const int j){
                    denv(i,j,comp) = (maskv(i,j,comps)>0 ? -1. : 0.);
                });
             }
        }
        else{
            BoxLib::Error("UNKNOWN BC ON LEFT FACE IN APPLYBC");
        }
    }
}


void C_AVERAGECC (
    const Box& bx,
    const int nc,
FArrayBox& c,
const FArrayBox& f){
    
    //box extends:
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();
    
    //get fabs
    ViewFab<Real> cv = c.view_fab;
    ViewFab<Real> fv = f.view_fab;

    //execute
    Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
             cv(i,j,k,n) =  0.125 * (  fv(2*i+1,2*j+1,2*k  ,n)
                                     + fv(2*i  ,2*j+1,2*k  ,n)
                                     + fv(2*i+1,2*j  ,2*k  ,n)
                                     + fv(2*i  ,2*j  ,2*k  ,n)
                                     + fv(2*i+1,2*j+1,2*k+1,n)
                                     + fv(2*i  ,2*j+1,2*k+1,n)
                                     + fv(2*i+1,2*j  ,2*k+1,n)
                                     + fv(2*i  ,2*j  ,2*k+1,n) );
    });
}


void C_HARMONIC_AVERAGEEC (
    const Box& bx,
    const int nc,
    const int cdir,
FArrayBox& c,
const FArrayBox& f){
    
    //box extends:
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();
    
    //get fabs
    ViewFab<Real> cv = c.view_fab;
    ViewFab<Real> fv = f.view_fab;

    //execute
    switch(cdir){
        case 0:
        Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) =  4./(
                            + 1./fv(2*i,2*j  ,2*k  ,n)
                            + 1./fv(2*i,2*j+1,2*k  ,n)
                            + 1./fv(2*i,2*j  ,2*k+1,n)
                            + 1./fv(2*i,2*j+1,2*k+1,n) );
         });
         break;
         
         case 1:
         Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) = 4./(
                            + 1./fv(2*i  ,2*j,2*k  ,n)
                            + 1./fv(2*i+1,2*j,2*k  ,n)
                            + 1./fv(2*i  ,2*j,2*k+1,n)
                            + 1./fv(2*i+1,2*j,2*k+1,n) );
          });
         break;
         
         case 2:
         Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) = 4./(
                           + 1./fv(2*i  ,2*j  ,2*k,n)
                           + 1./fv(2*i+1,2*j  ,2*k,n)
                           + 1./fv(2*i  ,2*j+1,2*k,n)
                           + 1./fv(2*i+1,2*j+1,2*k,n) );
         });
         break;
     }
}


void C_AVERAGEEC (
    const Box& bx,
    const int nc,
    const int cdir,
FArrayBox& c,
const FArrayBox& f){
    
    //box extends:
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();
    
    //get fabs
    ViewFab<Real> cv = c.view_fab;
    ViewFab<Real> fv = f.view_fab;

    //execute
    switch(cdir){
        case 0:
        Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) = 0.25*(
                            + fv(2*i,2*j  ,2*k  ,n)
                            + fv(2*i,2*j+1,2*k  ,n)
                            + fv(2*i,2*j  ,2*k+1,n)
                            + fv(2*i,2*j+1,2*k+1,n) );
         });
         break;
         
         case 1:
         Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) = 0.25*(
                            + fv(2*i  ,2*j,2*k  ,n)
                            + fv(2*i+1,2*j,2*k  ,n)
                            + fv(2*i  ,2*j,2*k+1,n)
                            + fv(2*i+1,2*j,2*k+1,n) );
          });
         break;
         
         case 2:
         Kokkos::Experimental::md_parallel_for(mdpolicy<4>({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, nc}, {cb[0], cb[1], cb[2], nc}), KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
                cv(i,j,k,n) = 0.25*(
                            + fv(2*i  ,2*j  ,2*k,n)
                            + fv(2*i+1,2*j  ,2*k,n)
                            + fv(2*i  ,2*j+1,2*k,n)
                            + fv(2*i+1,2*j+1,2*k,n) );
         });
         break;
     }
}
