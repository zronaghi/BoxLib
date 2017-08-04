#ifndef _MG_F_H_
#define _MG_F_H_

#include <REAL.H>

#if        defined(BL_LANG_FORT)

#if (BL_SPACEDIM == 1) 
#define FORT_AVERAGE   average1dgen
#define FORT_INTERP    interp1dgen
#endif

#if (BL_SPACEDIM == 2) 
#define FORT_AVERAGE   average2dgen
#define FORT_INTERP    interp2dgen
#endif

#if (BL_SPACEDIM == 3) 
#define FORT_AVERAGE   average3dgen
#define FORT_INTERP    interp3dgen
#endif

#else

#if (BL_SPACEDIM == 1)

#if    defined(BL_FORT_USE_UPPERCASE)
#define FORT_AVERAGE   AVERAGE1DGEN
#define FORT_INTERP    INTERP1DGEN
#elif  defined(BL_FORT_USE_LOWERCASE)
#define FORT_AVERAGE   average1dgen
#define FORT_INTERP    interp1dgen
#elif  defined(BL_FORT_USE_UNDERSCORE)
#define FORT_AVERAGE   average1dgen_
#define FORT_INTERP    interp1dgen_
#endif

#endif

#if (BL_SPACEDIM == 2)

#if    defined(BL_FORT_USE_UPPERCASE)
#define FORT_AVERAGE   AVERAGE2DGEN
#define FORT_INTERP    INTERP2DGEN
#elif  defined(BL_FORT_USE_LOWERCASE)
#define FORT_AVERAGE   average2dgen
#define FORT_INTERP    interp2dgen
#elif  defined(BL_FORT_USE_UNDERSCORE)
#define FORT_AVERAGE   average2dgen_
#define FORT_INTERP    interp2dgen_
#endif

#endif

#if (BL_SPACEDIM == 3)

#if    defined(BL_FORT_USE_UPPERCASE)
#define FORT_AVERAGE   AVERAGE3DGEN
#define FORT_INTERP    INTERP3DGEN
#elif  defined(BL_FORT_USE_LOWERCASE)
#define FORT_AVERAGE   average3dgen
#define FORT_INTERP    interp3dgen
#elif  defined(BL_FORT_USE_UNDERSCORE)
#define FORT_AVERAGE   average3dgen_
#define FORT_INTERP    interp3dgen_
#endif

#endif

#include <ArrayLim.H>
#include <Mask.H>
#include <Kokkos_Core.hpp>

void C_AVERAGE(
		const Box& bx,
        const int nc,
        FArrayBox& c,
        const FArrayBox& f);

void C_INTERP(
		const Box& bx,
        const int nc,
        FArrayBox& f,
        const FArrayBox& c);

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
		const Real* h);
		
void C_ADOTX(const Box& bx,
		const int nc,
		FArrayBox& y,
		const FArrayBox& x,
		Real alpha,
		Real beta,
		const FArrayBox& a,
		const FArrayBox& bX,
		const FArrayBox& bY,
		const FArrayBox& bZ,
		const Real* h);
		
void C_FLUX(const Box& xbx,
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
		const Real* h);

#if (BL_SPACEDIM == 3)
        template<class T>
        class ViewFab{
        public:
            
            T& operator()(const int& i, const int& j, const int& k, const int& n = 0){
                return data[n](i-smallend[0],j-smallend[1],k-smallend[2]);
            }
            
            const T& operator()(const int& i, const int& j, const int& k, const int& n = 0) const{
                return data[n](i-smallend[0],j-smallend[1],k-smallend[2]);
            }
            
            const T& operator()(int i, int j, int k, int n=0) const{
                return data[n](i-smallend[0],j-smallend[1],k-smallend[2]);
            }
            
            void init(const BaseFab<T>& rhs_, const std::string& name_){
                name=name_;
                smallend=rhs_.smallEnd();
                bigend=rhs_.bigEnd();
                length=IntVect(rhs_.length()[0],rhs_.length()[1],rhs_.length()[2]);
                numvars=rhs_.nComp();
                for(unsigned int n=0; n<numvars; n++){
                    data.push_back(Kokkos::View<T***>(name+"_comp_"+std::to_string(n),length[0],length[1],length[2]));
#pragma omp parallel for collapse(3)
                    for(unsigned int k_=smallend[2]; k_<=bigend[2]; k_++){
                        for(unsigned int j_=smallend[1]; j_<=bigend[1]; j_++){
                            for(unsigned int i_=smallend[0]; i_<bigend[0]; i_++){
                                const int k = k_ - smallend[2];
                                const int j = j_ - smallend[1];
                                const int i = i_ - smallend[0];
                                data[n](i,j,k) = rhs_(IntVect(i_,j_,k_),n);
                            }
                        }
                    }
                }
            }
            
            ViewFab(){}
            
            ViewFab(const BaseFab<T>& rhs_, const std::string& name_){
                init(rhs_,name_);
            }
            
            ViewFab<T>& operator=(const ViewFab<T>& rhs_){
                //clear old
                data.clear();
        
                //copy stuff over
                name=rhs_.name;
                numvars=rhs_.numvars;
                smallend=rhs_.smallend;
                bigend=rhs_.bigend;
                length=rhs_.length;
                for(unsigned int n=0; n<numvars; n++){
                    data.push_back(Kokkos::View<T***>(name+"_comp_"+std::to_string(n),length[0],length[1],length[2]));
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
            std::vector< Kokkos::View<T***> > data;
        };

void C_GSRB_3D(const Box& bx,
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
        const Real* h);
#endif

extern "C"
{
    void FORT_AVERAGE (
        Real* crse,       ARLIM_P(crse_lo), ARLIM_P(crse_hi),
        const Real* fine, ARLIM_P(fine_lo), ARLIM_P(fine_hi),
        const int *tlo, const int *thi,
        const int *nc);

    void FORT_INTERP (
        Real* fine,       ARLIM_P(fine_lo), ARLIM_P(fine_hi),
        const Real* crse, ARLIM_P(crse_lo), ARLIM_P(crse_hi),
        const int *tlo, const int *thi,
        const int *nc);
}
#endif

#endif /*_MG_F_H_*/
