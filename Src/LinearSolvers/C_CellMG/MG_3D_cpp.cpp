#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>
#include <iostream>

//typedef Kokkos::Device<Kokkos::OpenMP,Kokkos::CudaUVMSpace>::memory_space  hostspace;
typedef Kokkos::HostSpace hostspace;
#ifdef KOKKOS_ENABLE_CUDA
typedef Kokkos::CudaSpace devspace;
#else
typedef Kokkos::HostSpace devspace;
#endif

//a small class for wrapping kokkos views nicely
template<>
class ViewFab<Real> {
public:
    
  //swap indices here to get "natural" layout
  KOKKOS_INLINE_FUNCTION
  Real& operator()(const int& i, const int& j, const int& k, const int& n = 0){
    return d_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }
    
  KOKKOS_INLINE_FUNCTION
  Real& operator()(const int& i, const int& j, const int& k, const int& n = 0) const {
    return d_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }

  const Real& getHostElem(const int& i, const int& j, const int& k, const int& n = 0) const {
    return h_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }

  Real& getHostElem(const int& i, const int& j, const int& k, const int& n = 0) {
    return h_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }
  
  void init(const FArrayBox& rhs_, const std::string& name_){
    name=name_;
    Kokkos::Profiling::pushRegion("Init ViewFab "+name);
    smallend=rhs_.smallEnd();
    bigend=rhs_.bigEnd();
    length=IntVect(rhs_.length()[0],rhs_.length()[1],rhs_.length()[2]);
    numvars=rhs_.nComp();
        
    //allocate views
    d_data=Kokkos::View<Real****,devspace>(Kokkos::ViewAllocateWithoutInitializing(name),numvars,length[2],length[1],length[0]);
    h_data=Kokkos::create_mirror(d_data);
	
#pragma omp parallel for collapse(4)
    for(unsigned int n=0; n<numvars; n++){
      for(int k=smallend[2]; k<=bigend[2]; k++){
        for(int j=smallend[1]; j<=bigend[1]; j++){
          for(int i=smallend[0]; i<=bigend[0]; i++){
            this->getHostElem(i,j,k,n) = rhs_(IntVect(i,j,k),n);
          }
        }
      }
    }
    Kokkos::Profiling::popRegion();
  }
    
  ViewFab(){}

  void syncH2D(){
    std::cout << "uploading view(" << name << ") ... " << std::flush;
    Kokkos::fence();
    Kokkos::deep_copy(d_data,h_data);
    std::cout << "done!" << std::endl;
  }

  void syncD2H(){
    std::cout << "downloading view(" << name << ") ... " << std::flush;
    Kokkos::fence();
    Kokkos::deep_copy(h_data,d_data);
    std::cout << "done!" << std::endl;
  }
  
  ViewFab(const FArrayBox& rhs_, const std::string& name_){
    init(rhs_,name_);
  }
    
  ViewFab<Real>& operator=(const ViewFab<Real>& rhs_){
    //copy stuff over
    name=rhs_.name;
    Kokkos::Profiling::pushRegion("Copy ViewFab "+name);
    numvars=rhs_.numvars;

    //we need that on the device too:
    smallend=rhs_.smallend;
    bigend=rhs_.bigend;
    //length of box
    length=rhs_.length;
    //realloc data
    d_data=Kokkos::View<Real****,devspace>(Kokkos::ViewAllocateWithoutInitializing(name),numvars,length[2],length[1],length[0]);
    h_data=Kokkos::create_mirror_view(d_data);
        
#pragma omp parallel for collapse(4)
    for(unsigned int n=0; n<numvars; n++){
      for(int k=smallend[2]; k<=bigend[2]; k++){
        for(int j=smallend[1]; j<=bigend[1]; j++){
          for(int i=smallend[0]; i<=bigend[0]; i++){
            this->getHostElem(i,j,k,n) = rhs_(i,j,k,n);
          }
        }
      }
    }
    Kokkos::Profiling::popRegion();
        
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
              lhs_(IntVect(i,j,k),n) = this->getHostElem(i,j,k,n);
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
  Kokkos::View<Real****,devspace> d_data;
  Kokkos::View<Real****,devspace>::HostMirror h_data;
};

template<>
class ViewFab<int> {
public:
    
  KOKKOS_INLINE_FUNCTION
  int& operator()(const int& i, const int& j, const int& k, const int& n=0){
    return d_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }
    
  KOKKOS_INLINE_FUNCTION
  const int& operator()(const int& i, const int& j, const int& k, const int& n=0) const{
    return d_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }

  int& getHostElem(const int& i, const int& j, const int& k, const int& n=0){
    return h_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }

  const int& getHostElem(const int& i, const int& j, const int& k, const int& n=0) const{
    return h_data(n, k-smallend[2], j-smallend[1], i-smallend[0]);
  }
    
  void init(const Mask& rhs_, const std::string& name_){
    name=name_;
    smallend=rhs_.smallEnd();
    bigend=rhs_.bigEnd();
    length=IntVect(rhs_.length()[0],rhs_.length()[1],rhs_.length()[2]);
    numvars=rhs_.nComp();
    //realloc data
    d_data=Kokkos::View<int****,devspace>(Kokkos::ViewAllocateWithoutInitializing(name),numvars,length[2],length[1],length[0]);
    h_data=Kokkos::create_mirror_view(d_data);
        
#pragma omp parallel for collapse(4)
    for(unsigned int n=0; n<numvars; n++){
      for(int k=smallend[2]; k<=bigend[2]; k++){
        for(int j=smallend[1]; j<=bigend[1]; j++){
          for(int i=smallend[0]; i<=bigend[0]; i++){
            this->getHostElem(i,j,k,n) = rhs_(IntVect(i,j,k),n);
          }
        }
      }
    }
  }
    
  ViewFab(){}

  void syncH2D(){
    std::cout << "uploading view(" << name << ") ... " << std::flush;
    Kokkos::fence();
    Kokkos::deep_copy(d_data,h_data);
    std::cout << "done!" << std::endl;
  }

  void syncD2H(){
    std::cout << "downloading view(" << name << ") ... " << std::flush;
    Kokkos::fence();
    Kokkos::deep_copy(h_data,d_data);
    std::cout << "done!" << std::endl;
  }
  
  ViewFab<int>(const Mask& rhs_, const std::string name_){
    init(rhs_,name_);
  }
    
  ViewFab<int>& operator=(const ViewFab<int>& rhs_){
    //copy stuff over
    name=rhs_.name;
    numvars=rhs_.numvars;
    smallend=rhs_.smallend;
    bigend=rhs_.bigend;
    length=rhs_.length;
    //realloc data
    d_data=Kokkos::View<int****,devspace>(Kokkos::ViewAllocateWithoutInitializing(name),numvars,length[2],length[1],length[0]);
    h_data=Kokkos::create_mirror_view(d_data);
        
#pragma omp parallel for collapse(4)
    for(unsigned int n=0; n<numvars; n++){
      for(int k=smallend[2]; k<=bigend[2]; k++){
        for(int j=smallend[1]; j<=bigend[1]; j++){
          for(int i=smallend[0]; i<=bigend[0]; i++){
            this->getHostElem(i,j,k,n) = rhs_(i,j,k,n);
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
  Kokkos::View<int****,devspace> d_data;
  Kokkos::View<int****,devspace>::HostMirror h_data;
};


//Average Functor
struct C_AVERAGE_FUNCTOR{
public:
  C_AVERAGE_FUNCTOR(const Box& bx_, const int& nc_, const FArrayBox& c_, const FArrayBox& f_) : cv(c_,"cv"), fv(f_,"fv"), nc(nc_), bx(bx_){
    cv.syncH2D();
    fv.syncH2D();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int k, const int j, const int i) const{
    cv(i,j,k,n) =  (fv(2*i+1,2*j+1,2*k,n) + fv(2*i,2*j+1,2*k,n) + fv(2*i+1,2*j,2*k,n) + fv(2*i,2*j,2*k,n))*0.125;
    cv(i,j,k,n) += (fv(2*i+1,2*j+1,2*k+1,n) + fv(2*i,2*j+1,2*k+1,n) + fv(2*i+1,2*j,2*k+1,n) + fv(2*i,2*j,2*k+1,n))*0.125;
  }

  void fill(FArrayBox& cfab){
    cv.syncD2H();
    cv.fill(cfab);
  }
private:
  ViewFab<Real> cv, fv;
  Box bx;
  int nc;
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
	
  //create functor
  C_AVERAGE_FUNCTOR cavfunc(bx,nc,c,f);
    
  //define policy
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4> > t_policy;

  //execute
  Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{nc,cb[2],cb[1],cb[0]}),cavfunc);

  //write back
  cavfunc.fill(c);
}


//Interpolation Functor
struct C_INTERP_FUNCTOR{
public:
  C_INTERP_FUNCTOR(const Box& bx_, const int& nc_, const FArrayBox& f_, const FArrayBox& c_) : fv(f_,"fv"), cv(c_,"cv"), nc(nc_), bx(bx_){
    fv.syncH2D();
    cv.syncH2D();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int k, const int j, const int i) const{
    fv(2*i+1,2*j+1,2*k  ,n)       += cv(i,j,k,n);
    fv(2*i  ,2*j+1,2*k  ,n)       += cv(i,j,k,n);
    fv(2*i+1,2*j  ,2*k  ,n)       += cv(i,j,k,n);
    fv(2*i  ,2*j  ,2*k  ,n)       += cv(i,j,k,n);
    fv(2*i+1,2*j+1,2*k+1,n)       += cv(i,j,k,n);
    fv(2*i  ,2*j+1,2*k+1,n)       += cv(i,j,k,n);
    fv(2*i+1,2*j  ,2*k+1,n)       += cv(i,j,k,n);
    fv(2*i  ,2*j  ,2*k+1,n)       += cv(i,j,k,n);
  }

  void fill(FArrayBox& ffab){
    fv.syncD2H();
    fv.fill(ffab);
  }
private:
  ViewFab<Real> fv, cv;
  Box bx;
  int nc;
};

void C_INTERP(const Box& bx,
const int nc,
FArrayBox& f,
const FArrayBox& c){
	
  const int *lo = bx.loVect();
  const int *hi = bx.hiVect();
  const int* cb = bx.cbVect();

  //create functor
  C_INTERP_FUNCTOR cintfunc(bx,nc,f,c);

  //define policy
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4> > t_policy;
	
  // Execute functor
  Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{nc,cb[2],cb[1],cb[0]}),cintfunc);
  
  //write back
  cintfunc.fill(f);
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
  const int& nc_,
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
  phiv(phi_,"phiv"), rhsv(rhs_,"rhsv"), av(a_,"av"), bXv(bX_,"bXv"), bYv(bY_,"bYv"), bZv(bZ_,"bZv"), 
  f0v(f0_,"f0v"), f1v(f1_,"f1v"), f2v(f2_,"f2v"), f3v(f3_,"f3v"), f4v(f4_,"f4v"), f5v(f5_,"f5v"),
  m0v(m0_,"m0v"), m1v(m1_,"m1v"), m2v(m2_,"m2v"), m3v(m3_,"m3v"), m4v(m4_,"m4v"), m5v(m5_,"m5v"),
  nc(nc_), rb(rb_), comp(0), bx(bx_), bbx(bbx_), alpha(alpha_), beta(beta_) {

    //sync to space:
    phiv.syncH2D();
    rhsv.syncH2D();
    av.syncH2D();
    bXv.syncH2D();
    bYv.syncH2D();
    bZv.syncH2D();
    f0v.syncH2D();
    f1v.syncH2D();
    f2v.syncH2D();
    f3v.syncH2D();
    f4v.syncH2D();
    f5v.syncH2D();
    m0v.syncH2D();
    m1v.syncH2D();
    m2v.syncH2D();
    m3v.syncH2D();
    m4v.syncH2D();
    m5v.syncH2D();
    
    //some parameters
    omega= 1.15;
    dhx = beta/(h[0]*h[0]);
    dhy = beta/(h[1]*h[1]);
    dhz = beta/(h[2]*h[2]);

    lo0=bx.loVect()[0];
    hi0=bx.hiVect()[0];
        
    blo=const_cast<int*>(bbx.loVect());
    bhi=const_cast<int*>(bbx.hiVect());

    d_blo=Kokkos::View<int[3], devspace>("blo");
    h_blo=Kokkos::create_mirror_view(d_blo);
    d_bhi=Kokkos::View<int[3], devspace>("bhi");
    h_bhi=Kokkos::create_mirror_view(d_bhi);
    for(unsigned int d=0; d<3; d++){
      h_blo(d)=bbx.loVect()[d];
      h_bhi(d)=bbx.hiVect()[d];
    }
    Kokkos::deep_copy(h_blo,d_blo);
    Kokkos::deep_copy(h_bhi,d_bhi);
  }
    
  void setComp(const int& n){
    comp=n;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int k, const int j, const int i) const{
    if ( (i + j + k + rb) % 2 ==0){

      //BC terms
      Real cf0 = ( (i==d_blo[0]) && (m0v(d_blo[0]-1,j,k)>0) ? f0v(d_blo[0],j,k) : 0. );
      Real cf1 = ( (j==d_blo[1]) && (m1v(i,d_blo[1]-1,k)>0) ? f1v(i,d_blo[1],k) : 0. );
      Real cf2 = ( (k==d_blo[2]) && (m2v(i,j,d_blo[2]-1)>0) ? f2v(i,j,d_blo[2]) : 0. );
      Real cf3 = ( (i==d_bhi[0]) && (m3v(d_bhi[0]+1,j,k)>0) ? f3v(d_bhi[0],j,k) : 0. );
      Real cf4 = ( (j==d_bhi[1]) && (m4v(i,d_bhi[1]+1,k)>0) ? f4v(i,d_bhi[1],k) : 0. );
      Real cf5 = ( (k==d_bhi[2]) && (m5v(i,j,d_bhi[2]+1)>0) ? f5v(i,j,d_bhi[2]) : 0. );

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

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int k, const int j) const{
    int ioff = (lo0 + j + k + rb) % 2;
    for(int i=ioff; i<=hi0; i+=2){

      //BC terms
      Real cf0 = ( (i==d_blo[0]) && (m0v(d_blo[0]-1,j,k)>0) ? f0v(d_blo[0],j,k) : 0. );
      Real cf1 = ( (j==d_blo[1]) && (m1v(i,d_blo[1]-1,k)>0) ? f1v(i,d_blo[1],k) : 0. );
      Real cf2 = ( (k==d_blo[2]) && (m2v(i,j,d_blo[2]-1)>0) ? f2v(i,j,d_blo[2]) : 0. );
      Real cf3 = ( (i==d_bhi[0]) && (m3v(d_bhi[0]+1,j,k)>0) ? f3v(d_bhi[0],j,k) : 0. );
      Real cf4 = ( (j==d_bhi[1]) && (m4v(i,d_bhi[1]+1,k)>0) ? f4v(i,d_bhi[1],k) : 0. );
      Real cf5 = ( (k==d_bhi[2]) && (m5v(i,j,d_bhi[2]+1)>0) ? f5v(i,j,d_bhi[2]) : 0. );

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
    
  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, const int j) const{
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
      
      double rho =  dhx * (bXv(i,j,k)*phiv(i-1,j,k,comp) + bXv(i+1,j,k)*phiv(i+1,j,k,comp))
        + dhy * (bYv(i,j,k)*phiv(i,j-1,k,comp) + bYv(i,j+1,k)*phiv(i,j+1,k,comp))
          + dhz * (bZv(i,j,k)*phiv(i,j,k-1,comp) + bZv(i,j,k+1)*phiv(i,j,k+1,comp));
      
      double res = rhsv(i,j,k,comp) - gamma * phiv(i,j,k,comp) + rho;
      phiv(i,j,k,comp) += omega/g_m_d * res;
    }
  }

  void fill(FArrayBox& phifab){
    phiv.syncD2H();
    phiv.fill(phifab);
  }

private:
  ViewFab<Real> phiv, rhsv, av, bXv, bYv, bZv;
  ViewFab<Real> f0v, f1v, f2v, f3v, f4v, f5v;
  ViewFab<int> m0v, m1v, m2v, m3v, m4v, m5v;
  Box bx, bbx;
  int nc, rb, comp;
  Real alpha, beta, dhx, dhy, dhz, omega;
  int lo0, hi0;
  int *blo, *bhi;
  Kokkos::View<int[3], devspace> d_blo, d_bhi;
  Kokkos::View<int[3], devspace>::HostMirror h_blo, h_bhi;
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
  C_GSRB_FUNCTOR cgsrbfunc(bx, bbx, nc, rb, alpha, beta, phi, rhs, a, bX, bY, bZ, f0, m0, f1, m1, f2, m2, f3, m3, f4, m4, f5, m5, h);
    
#if 0
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4> > t_policy;
  //execute
  double start_time = omp_get_wtime();
  Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{nc,cb[2],cb[1],cb[0]}),cgsrbfunc);
  double end_time =  omp_get_wtime();
#else
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3> > t_policy;
  //execute
  double start_time = omp_get_wtime();
  Kokkos::Experimental::md_parallel_for(t_policy({0,lo[2],lo[1]},{nc,hi[2]+1,hi[1]+1},{nc,cb[2],cb[1]}),cgsrbfunc);
  double end_time =  omp_get_wtime();
#endif
  std::cout << "GSRB Elapsed time: " << end_time - start_time << std::endl;
            
  //copy data back from the views
  cgsrbfunc.fill(phi);
}

//-----------------------------------------------------------------------
//
//     Fill in a matrix x vector operator here
//
//ADOTX Functor
struct C_ADOTX_FUNCTOR{
public:
  C_ADOTX_FUNCTOR(const Box& bx_, 
  const int& nc_, 
  const FArrayBox& y_, 
  const FArrayBox& x_,  
  const Real& alpha_, 
  const Real& beta_, 
  const FArrayBox& a_, 
  const FArrayBox& bX_,
  const FArrayBox& bY_,
  const FArrayBox& bZ_,
  const Real* h) :
  yv(y_,"yv"), xv(x_,"xv"), av(a_,"av"), bXv(bX_,"bXv"), bYv(bY_,"bYv"), bZv(bZ_,"bZv"), nc(nc_), bx(bx_), alpha(alpha_), beta(beta_) {

    //sync
    yv.syncH2D();
    xv.syncH2D();
    av.syncH2D();
    bXv.syncH2D();
    bYv.syncH2D();
    bZv.syncH2D();
    
    //helpers
    d_helpers=Kokkos::View<Real[4],devspace>("helpers");
    h_helpers=Kokkos::create_mirror_view(d_helpers);
    h_helpers(0) = beta/(h[0]*h[0]);
    h_helpers(1) = beta/(h[1]*h[1]);
    h_helpers(2) = beta/(h[2]*h[2]);
    h_helpers(3) = alpha;
    Kokkos::deep_copy(d_helpers,h_helpers);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int k, const int j, const int i) const{
    yv(i,j,k,n) = 0.5;
    /*yv(i,j,k,n) = d_helpers(3) * av(i,j,k) * xv(i,j,k,n)
    - d_helpers(0) * (   bXv(i+1,j,  k  ) * ( xv(i+1,j,  k,  n) - xv(i,  j,  k  ,n) )
    - bXv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i-1,j,  k  ,n) ) )
    - d_helpers(1) * (   bYv(i,  j+1,k  ) * ( xv(i,  j+1,k,  n) - xv(i,  j  ,k  ,n) )
    - bYv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i,  j-1,k  ,n) ) )
    - d_helpers(2) * (   bZv(i,  j,  k+1) * ( xv(i,  j,  k+1,n) - xv(i,  j  ,k  ,n) )
    - bZv(i,  j,  k  ) * ( xv(i,  j,  k,  n) - xv(i,  j,  k-1,n) ) );*/
  }

  void fill(FArrayBox& yfab){
    yv.syncD2H();
    yv.fill(yfab);
  }

private:
  ViewFab<Real> yv, xv, av, bXv, bYv, bZv;
  Box bx;
  int nc;
  Kokkos::View<Real[4],devspace> d_helpers;
  Kokkos::View<Real[4],devspace>::HostMirror h_helpers;
  Real alpha, beta; //, dhx, dhy, dhz;
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
  C_ADOTX_FUNCTOR cadxfunc(bx,nc,y,x,alpha,beta,a,bX,bY,bZ,h);

  //create policy
  typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4> > t_policy;

  std::cout << "BOX LO (" << lo[0] << "," << lo[1] << "," << lo[2] << ")" << std::endl;
  std::cout << "BOX HI (" << hi[0] << "," << hi[1] << "," << hi[2] << ")" << std::endl;
    
  //execute
  Kokkos::parallel_for(t_policy({0,lo[2],lo[1],lo[0]},{nc,hi[2]+1,hi[1]+1,hi[0]+1},{nc,cb[2],cb[1],cb[0]}),cadxfunc);
    
  //write back result
  cadxfunc.fill(y);

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
