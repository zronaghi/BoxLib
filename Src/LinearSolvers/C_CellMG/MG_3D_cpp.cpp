#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>
#include <iostream>

//a small class for wrapping kokkos views nicely
template<>
class ViewFab<Real> {
public:
    
    //swap indices here to get "natural" layout
    KOKKOS_INLINE_FUNCTION
    Real& operator()(const int& i, const int& j, const int& k, const int& n = 0){
        return data[n](k-smallend[2], j-smallend[1], i-smallend[0]);
    }
    
    KOKKOS_INLINE_FUNCTION
    const Real& operator()(const int& i, const int& j, const int& k, const int& n = 0) const {
        return data[n](k-smallend[2], j-smallend[1], i-smallend[0]);
    }
    
    void init(const FArrayBox& rhs_, const std::string& name_){
        name=name_;
        smallend=rhs_.smallEnd();
        bigend=rhs_.bigEnd();
        length=IntVect(rhs_.length()[0],rhs_.length()[1],rhs_.length()[2]);
        numvars=rhs_.nComp();
        
        for(unsigned int n=0; n<numvars; n++){
            data.push_back(Kokkos::View<Real***>(name+"_comp_"+std::to_string(n),length[2],length[1],length[0]));
#pragma omp parallel for collapse(3)
            for(int k=smallend[2]; k<=bigend[2]; k++){
                for(int j=smallend[1]; j<=bigend[1]; j++){
                    for(int i=smallend[0]; i<=bigend[0]; i++){
                        (*this)(i,j,k,n) = rhs_(IntVect(i,j,k),n);
                    }
                }
            }
        }
    }
    
    ViewFab(){}
    
    ViewFab(const FArrayBox& rhs_, const std::string& name_){
        init(rhs_,name_);
    }
    
    ViewFab<Real>& operator=(const ViewFab<Real>& rhs_){
        data.clear();
        
        //copy stuff over
        name=rhs_.name;
        numvars=rhs_.numvars;
        smallend=rhs_.smallend;
        bigend=rhs_.bigend;
        length=rhs_.length;
        
        for(unsigned int n=0; n<numvars; n++){
            data.push_back(Kokkos::View<Real***>(name+"_comp_"+std::to_string(n),length[2],length[1],length[0]));
#pragma omp parallel for collapse(3)
            for(int k=smallend[2]; k<=bigend[2]; k++){
                for(int j=smallend[1]; j<=bigend[1]; j++){
                    for(int i=smallend[0]; i<=bigend[0]; i++){
                        (*this)(i,j,k,n) = rhs_(i,j,k,n);
                    }
                }
            }
        }
        
        return *this;
    }
    
    void fill(FArrayBox& lhs_) const{
        //do some sanity checks:
        bool is_ok=true;
        IntVect tmp_smallend=lhs_.smallEnd();
        IntVect tmp_bigend=lhs_.bigEnd();
        IntVect tmp_length=IntVect(lhs_.length()[0],lhs_.length()[1],lhs_.length()[2]);
        int tmp_numvars=lhs_.nComp();
        if(tmp_numvars!=numvars) is_ok=false;
        for(unsigned int d=0; d<3; d++){
            if(tmp_smallend[d]!=smallend[d]) is_ok=false;
            if(tmp_bigend[d]!=bigend[d]) is_ok=false;
            if(tmp_length[d]!=length[d]) is_ok=false;
        }
        if(is_ok){
#pragma omp parallel for collapse(4)
            for(unsigned int n=0; n<numvars; n++){
                for(int k=smallend[2]; k<=bigend[2]; k++){
                    for(int j=smallend[1]; j<=bigend[1]; j++){
                        for(int i=smallend[0]; i<=bigend[0]; i++){
                            lhs_(IntVect(i,j,k),n) = (*this)(i,j,k,n);
                        }
                    }
                }
            }
        }
    }
        
private:
    std::string name;
    int numvars;
    IntVect smallend, bigend, length;
    std::vector< Kokkos::View<Real***> > data;
};

template<>
class ViewFab<int> {
public:
    
    KOKKOS_INLINE_FUNCTION
    int& operator()(const int& i, const int& j, const int& k, const int& n=0){
        return data[n](k-smallend[2], j-smallend[1], i-smallend[0]);
    }
    
    KOKKOS_INLINE_FUNCTION
    const int& operator()(const int& i, const int& j, const int& k, const int& n=0) const{
        return data[n](k-smallend[2], j-smallend[1], i-smallend[0]);
    }
    
    void init(const Mask& rhs_, const std::string& name_){
        name=name_;
        smallend=rhs_.smallEnd();
        bigend=rhs_.bigEnd();
        length=IntVect(rhs_.length()[0],rhs_.length()[1],rhs_.length()[2]);
        numvars=rhs_.nComp();
        
        for(unsigned int n=0; n<numvars; n++){
            data.push_back(Kokkos::View<int***>(name+"_comp_"+std::to_string(n),length[2],length[1],length[0]));
#pragma omp parallel for collapse(3)
            for(int k=smallend[2]; k<=bigend[2]; k++){
                for(int j=smallend[1]; j<=bigend[1]; j++){
                    for(int i=smallend[0]; i<=bigend[0]; i++){
                        (*this)(i,j,k,n) = rhs_(IntVect(i,j,k),n);
                    }
                }
            }
        }
    }
    
    ViewFab(){}
    
    ViewFab<int>(const Mask& rhs_, const std::string name_){
        init(rhs_,name_);
    }
    
    ViewFab<int>& operator=(const ViewFab<int>& rhs_){
        //clear old
        data.clear();
        
        //copy stuff over
        name=rhs_.name;
        numvars=rhs_.numvars;
        smallend=rhs_.smallend;
        bigend=rhs_.bigend;
        length=rhs_.length;
        
        for(unsigned int n=0; n<numvars; n++){
            data.push_back(Kokkos::View<int***>(name+"_comp_"+std::to_string(n),length[2],length[1],length[0]));
#pragma omp parallel for collapse(3)
            for(int k=smallend[2]; k<=bigend[2]; k++){
                for(int j=smallend[1]; j<=bigend[1]; j++){
                    for(int i=smallend[0]; i<=bigend[0]; i++){
                        (*this)(i,j,k,n) = rhs_(i,j,k,n);
                    }
                }
            }
        }
        
        return *this;
    }
    
private:
    std::string name;
    int numvars;
    IntVect smallend, bigend, length;
    std::vector< Kokkos::View<int***> > data;
};


//Average Kernel
void C_AVERAGE(
    const Box& bx,
const int nc,
FArrayBox& c,
const FArrayBox& f){
	
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int* cb = bx.cbVect();
	
    //create views
    ViewFab<Real> cv(c,"c"), fv(f,"f");
    
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;

    //execute
    for(int n=0; n<nc; n++){
      Kokkos::Experimental::md_parallel_for(t_policy({lo[2],lo[1],lo[0]},{hi[2]+1,hi[1]+1,hi[0]+1},{cb[2],cb[1],cb[0]}),
					    [&](const int& k, const int& j, const int& i){
					      cv(i,j,k,n) =  (fv(2*i+1,2*j+1,2*k,n) + fv(2*i,2*j+1,2*k,n) + fv(2*i+1,2*j,2*k,n) + fv(2*i,2*j,2*k,n))*0.125;
					      cv(i,j,k,n) += (fv(2*i+1,2*j+1,2*k+1,n) + fv(2*i,2*j+1,2*k+1,n) + fv(2*i+1,2*j,2*k+1,n) + fv(2*i,2*j,2*k+1,n))*0.125;
					    });
	}
    //write back
    cv.fill(c);

}


//Interpolation Kernel
void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){
	
  const int *lo = bx.loVect();
  const int *hi = bx.hiVect();
  const int* cb = bx.cbVect();

  //create views                                                                                                                                                                                                  
  ViewFab<Real> cv(c,"c"), fv(f,"f");

  //define policy                                                                                                                                                                                                 
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;
	
  // Execute functor
  for(int n=0; n<nc; n++){
    Kokkos::Experimental::md_parallel_for(t_policy({lo[2],lo[1],lo[0]},{hi[2]+1,hi[1]+1,hi[0]+1},{cb[2],cb[1],cb[0]}),
					  [&](const int& k, const int& j, const int& i){
					    fv(2*i+1,2*j+1,2*k  ,n)       += cv(i,j,k,n);
					    fv(2*i  ,2*j+1,2*k  ,n)       += cv(i,j,k,n);
					    fv(2*i+1,2*j  ,2*k  ,n)       += cv(i,j,k,n);
					    fv(2*i  ,2*j  ,2*k  ,n)       += cv(i,j,k,n);
					    fv(2*i+1,2*j+1,2*k+1,n)       += cv(i,j,k,n);
					    fv(2*i  ,2*j+1,2*k+1,n)       += cv(i,j,k,n);
					    fv(2*i+1,2*j  ,2*k+1,n)       += cv(i,j,k,n);
					    fv(2*i  ,2*j  ,2*k+1,n)       += cv(i,j,k,n);
					  });
  }
  fv.fill(f);
  
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

    Real omega= 1.15;
    Real dhx = beta/(h[0]*h[0]);
    Real dhy = beta/(h[1]*h[1]);
    Real dhz = beta/(h[2]*h[2]);

    //#if USE_VIEWS
    //create views to do the magic:
    ViewFab<Real> phiv(phi,"phi"), rhsv(rhs,"rhs"), av(a,"a"), bxv(bX,"bx"), byv(bY,"by"), bzv(bZ,"bz");
    ViewFab<Real> f0v(f0,"f0"), f1v(f1,"f1"), f2v(f2,"f2"), f3v(f3,"f3"), f4v(f4,"f4"), f5v(f5,"f5");
    ViewFab<int> m0v(m0,"m0"), m1v(m1,"m1"), m2v(m2,"m2"), m3v(m3,"m3"), m4v(m4,"m4"), m5v(m5,"m5");
    
#if 0
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;
    //execute
    double start_time = omp_get_wtime();
    for(unsigned int n=0; n<nc; n++){
        Kokkos::Experimental::md_parallel_for(t_policy({lo[2],lo[1],lo[0]},{hi[2]+1,hi[1]+1,hi[0]+1},{cb[2],cb[1],cb[0]}),
            [&](const int& k, const int& j, const int& i){
                if ( (i + j + k + rb) % 2 ==0){
        
                    //BC terms
                    Real cf0 = ( (i==blo[0]) && (m0v(blo[0]-1,j,k)>0) ? f0v(blo[0],j,k) : 0. );
                    Real cf1 = ( (j==blo[1]) && (m1v(i,blo[1]-1,k)>0) ? f1v(i,blo[1],k) : 0. );
                    Real cf2 = ( (k==blo[2]) && (m2v(i,j,blo[2]-1)>0) ? f2v(i,j,blo[2]) : 0. );
                    Real cf3 = ( (i==bhi[0]) && (m3v(bhi[0]+1,j,k)>0) ? f3v(bhi[0],j,k) : 0. );
                    Real cf4 = ( (j==bhi[1]) && (m4v(i,bhi[1]+1,k)>0) ? f4v(i,bhi[1],k) : 0. );
                    Real cf5 = ( (k==bhi[2]) && (m5v(i,j,bhi[2]+1)>0) ? f5v(i,j,bhi[2]) : 0. );
		
                    //assign ORA constants
                    double gamma = alpha * av(i,j,k)
                                + dhx * (bxv(i,j,k) + bxv(i+1,j,k))
                                + dhy * (byv(i,j,k) + byv(i,j+1,k))
                                + dhz * (bzv(i,j,k) + bzv(i,j,k+1));
		
                    double g_m_d = gamma
                                - dhx * (bxv(i,j,k)*cf0 + bxv(i+1,j,k)*cf3)
                                - dhy * (byv(i,j,k)*cf1 + byv(i,j+1,k)*cf4)
                                - dhz * (bzv(i,j,k)*cf2 + bzv(i,j,k+1)*cf5);
		
                    double rho =  dhx * (bxv(i,j,k)*phiv(i-1,j,k,n) + bxv(i+1,j,k)*phiv(i+1,j,k,n))
                                + dhy * (byv(i,j,k)*phiv(i,j-1,k,n) + byv(i,j+1,k)*phiv(i,j+1,k,n))
                                + dhz * (bzv(i,j,k)*phiv(i,j,k-1,n) + bzv(i,j,k+1)*phiv(i,j,k+1,n));
		
                    double res = rhsv(i,j,k,n) - gamma * phiv(i,j,k,n) + rho;
                    phiv(i,j,k,n) += omega/g_m_d * res; 
                    
                }
            });
    }
    double end_time =  omp_get_wtime();
#else
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<2> > t_policy;
    //execute
    double start_time = omp_get_wtime();
    for(unsigned int n=0; n<nc; n++){
        Kokkos::Experimental::md_parallel_for(t_policy({lo[2],lo[1]},{hi[2]+1,hi[1]+1},{cb[2],cb[1]}),
            [&](const int& k, const int& j){
                int ioff = (lo[0] + j + k + rb) % 2;
                for(int i=ioff; i<=hi[0]; i+=2){
        
                    //BC terms
                    Real cf0 = ( (i==blo[0]) && (m0v(blo[0]-1,j,k)>0) ? f0v(blo[0],j,k) : 0. );
                    Real cf1 = ( (j==blo[1]) && (m1v(i,blo[1]-1,k)>0) ? f1v(i,blo[1],k) : 0. );
                    Real cf2 = ( (k==blo[2]) && (m2v(i,j,blo[2]-1)>0) ? f2v(i,j,blo[2]) : 0. );
                    Real cf3 = ( (i==bhi[0]) && (m3v(bhi[0]+1,j,k)>0) ? f3v(bhi[0],j,k) : 0. );
                    Real cf4 = ( (j==bhi[1]) && (m4v(i,bhi[1]+1,k)>0) ? f4v(i,bhi[1],k) : 0. );
                    Real cf5 = ( (k==bhi[2]) && (m5v(i,j,bhi[2]+1)>0) ? f5v(i,j,bhi[2]) : 0. );
		
                    //assign ORA constants
                    double gamma = alpha * av(i,j,k)
                                + dhx * (bxv(i,j,k) + bxv(i+1,j,k))
                                + dhy * (byv(i,j,k) + byv(i,j+1,k))
                                + dhz * (bzv(i,j,k) + bzv(i,j,k+1));
		
                    double g_m_d = gamma
                                - dhx * (bxv(i,j,k)*cf0 + bxv(i+1,j,k)*cf3)
                                - dhy * (byv(i,j,k)*cf1 + byv(i,j+1,k)*cf4)
                                - dhz * (bzv(i,j,k)*cf2 + bzv(i,j,k+1)*cf5);
		
                    double rho =  dhx * (bxv(i,j,k)*phiv(i-1,j,k,n) + bxv(i+1,j,k)*phiv(i+1,j,k,n))
                                + dhy * (byv(i,j,k)*phiv(i,j-1,k,n) + byv(i,j+1,k)*phiv(i,j+1,k,n))
                                + dhz * (bzv(i,j,k)*phiv(i,j,k-1,n) + bzv(i,j,k+1)*phiv(i,j,k+1,n));
		
                    double res = rhsv(i,j,k,n) - gamma * phiv(i,j,k,n) + rho;
                    phiv(i,j,k,n) += omega/g_m_d * res; 
                }
            });
    }
    double end_time =  omp_get_wtime();
#endif
    std::cout << "GSRB Elapsed time: " << end_time - start_time << std::endl;
    
    //#else
    //typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;
    //// Create a functor
    //C_GSRB_Functor gsrb_functor(rb,lo[0],hi[0],alpha,beta,&phi,&rhs,&a,&bX,&bY,&bZ,&f0,&m0,&f1,&m1,&f2,&m2,&f3,&m3,&f4,&m4,&f5,&m5,h,blo,bhi);
    //// Execute functor
    //Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1]},{nc,hi[2]+1,hi[1]+1},{1,4,4}),gsrb_functor);
    //#endif
        
    //copy data back from the views
    phiv.fill(phi);
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
    const int *cb = bx.cbVect();

    //some parameters                                                                                                                                                                                           
    Real dhx = beta/(h[0]*h[0]);
    Real dhy = beta/(h[1]*h[1]);
    Real dhz = beta/(h[2]*h[2]);

    //create views
    ViewFab<Real> yv(y,"y"), xv(x,"x"), av(a,"a"), bXv(bX,"bX"), bYv(bY,"bY"), bZv(bZ,"bZ");

    //create policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;
    
    //execute
    for(int n=0; n<nc; n++){
      Kokkos::Experimental::md_parallel_for(t_policy({lo[3],lo[2],lo[1]},{hi[3]+1,hi[2]+1,hi[1]+1},{cb[3],cb[2],cb[1]}),
					    [&](const int& k, const int& j, const int& i){
					      yv(i,j,k,n) = alpha * av(i,j,k) * xv(i,j,k,n)
						- dhx * (   bXv(i+1,j,  k  ) * ( xv(i+1,j,  k,  n) - xv(i,  j,  k  ,n) )
							  - bXv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i-1,j,  k  ,n) ) )
						- dhy * (   bYv(i,  j+1,k  ) * ( xv(i,  j+1,k,  n) - xv(i,  j  ,k  ,n) )
							  - bYv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i,  j-1,k  ,n) ) )
						- dhz * (   bZv(i,  j,  k+1) * ( xv(i,  j,  k+1,n) - xv(i,  j  ,k  ,n) )
	      					          - bZv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i,  j,  k-1,n) ) );
					    });
    }
    yv.fill(y);

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
