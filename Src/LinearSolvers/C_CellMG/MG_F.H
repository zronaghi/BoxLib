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

template <typename T>
class oarray {
    
public:
    //the ranges are inclusive!
    oarray(int lo_, int hi_){
        lo = lo_;
        size = (hi_+1) - lo_;
        data = new T[size];
    }
    
    ~oarray(){
        delete [] data;
    }
    
    T& operator[](int index){
        return data[index-lo];
    }
    
    const T& operator[](int index) const{
        return data[index-lo];
    }
    
    const int getOffset() const{
        return lo;
    }
    
private:
    int lo;
    int size;
    T* data;
};

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


        ////4D
        //template <typename T>
        //struct KokkosOffsetView4 {
        //    
        //    KokkosOffsetView4(IntVect, )
        //    
        //    devview<T****> view;
        //    int offset[4] = {};
        //}
        //
        //template <typename itype, typename ViewType>
        //struct is_rank_4 {
        //    static constexpr bool value = ViewType::rank == 4u;
        //}
        //
        //template <typename itype, typename ViewType>
        //struct is_rank_3 {
        //    static constexpr bool value = ViewType::rank == 3u;
        //}
        //
        //template <typename... Args>
        //struct KokkosOffsetView
        //{
        //    using view_type =  Kokkos::View<Args...>;
        //    static constexpr unsigned rank = view_type::rank;
        //    using reference = typename view_type::reference;
        //    
        //    //4D
        //    template <typename itype>
        //    typename std::enable_if< is_rank4<itype, view_type>, reference >::type
        //    KOKKOS_FORCEINLINE_FUNCTION
        //    operator()(itype i, int j, int k, int n){
        //        return view(i-offset[0], j-offset[1], k-offset[2], n-offset[3]);
        //    }
        //    
        //    template <typename itype>
        //    typename std::enable_if< is_rank4<itype, view_type>, reference >::type
        //    setOffset(const Intvect& offset_){
        //        for(unsigned int d=0; d<4; d++) offset[d]=offset_(d);
        //    }
        //    
        //    //3D
        //    template <typename itype>
        //    typename std::enable_if< is_rank3<itype, view_type>, reference >::type
        //    KOKKOS_FORCEINLINE_FUNCTION
        //    operator()(itype i, int j, int k){
        //        return view(i-offset[0], j-offset[1], k-offset[2]);
        //    }
        //    
        //    template <typename itype>
        //    typename std::enable_if< is_rank3<itype, view_type>, reference >::type
        //    setOffset(const Intvect& offset_){
        //        for(unsigned int d=0; d<3; d++) offset[d]=offset_(d);
        //    }
        //    
        //    template <typename... A>
        //    KokkosOffsetView(A &&... a) : view( std::forward<A>(a) ){}
        //    
        //    //members
        //    view_type view;
        //    int offset[rank] = {};
        //};

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
        const Real* h);

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
