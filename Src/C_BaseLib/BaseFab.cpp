
#include <winstd.H>

#include <cstring>
#include <cstdlib>

#include <BaseFab.H>
#include <BArena.H>
#include <CArena.H>

#if !(defined(BL_NO_FORT) || defined(WIN32))
#include <BaseFab_f.H>
#endif

#ifdef BL_MEM_PROFILING
#include <MemProfiler.H>
#endif

long BoxLib::private_total_bytes_allocated_in_fabs     = 0L;
long BoxLib::private_total_bytes_allocated_in_fabs_hwm = 0L;
long BoxLib::private_total_cells_allocated_in_fabs     = 0L;
long BoxLib::private_total_cells_allocated_in_fabs_hwm = 0L;

int BoxLib::BF_init::m_cnt = 0;

namespace
{
    Arena* the_arena = 0;
}

BoxLib::BF_init::BF_init ()
{
    if (m_cnt++ == 0)
    {
        BL_ASSERT(the_arena == 0);

#if defined(BL_COALESCE_FABS)
        the_arena = new CArena;
#else
        the_arena = new BArena;
#endif

#ifdef _OPENMP
#pragma omp parallel
	{
	    BoxLib::private_total_bytes_allocated_in_fabs     = 0;
	    BoxLib::private_total_bytes_allocated_in_fabs_hwm = 0;
	    BoxLib::private_total_cells_allocated_in_fabs     = 0;
	    BoxLib::private_total_cells_allocated_in_fabs_hwm = 0;
	}
#endif

#ifdef BL_MEM_PROFILING
	MemProfiler::add("Fab", std::function<MemProfiler::MemInfo()>
			 ([] () -> MemProfiler::MemInfo {
			     return {BoxLib::TotalBytesAllocatedInFabs(),
				     BoxLib::TotalBytesAllocatedInFabsHWM()};
			 }));
#endif
    }
}

BoxLib::BF_init::~BF_init ()
{
    if (--m_cnt == 0)
        delete the_arena;
}

long 
BoxLib::TotalBytesAllocatedInFabs()
{
#ifdef _OPENMP
    long r=0;
#pragma omp parallel reduction(+:r)
    {
	r += private_total_bytes_allocated_in_fabs;
    }
    return r;
#else
    return private_total_bytes_allocated_in_fabs;
#endif
}

long 
BoxLib::TotalBytesAllocatedInFabsHWM()
{
#ifdef _OPENMP
    long r=0;
#pragma omp parallel reduction(+:r)
    {
	r += private_total_bytes_allocated_in_fabs_hwm;
    }
    return r;
#else
    return private_total_bytes_allocated_in_fabs_hwm;
#endif
}

long 
BoxLib::TotalCellsAllocatedInFabs()
{
#ifdef _OPENMP
    long r=0;
#pragma omp parallel reduction(+:r)
    {
	r += private_total_cells_allocated_in_fabs;
    }
    return r;
#else
    return private_total_cells_allocated_in_fabs;
#endif
}

long 
BoxLib::TotalCellsAllocatedInFabsHWM()
{
#ifdef _OPENMP
    long r=0;
#pragma omp parallel reduction(+:r)
    {
	r += private_total_cells_allocated_in_fabs_hwm;
    }
    return r;
#else
    return private_total_cells_allocated_in_fabs_hwm;
#endif
}

void 
BoxLib::ResetTotalBytesAllocatedInFabsHWM()
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
	private_total_bytes_allocated_in_fabs_hwm = 0;
    }
}

void
BoxLib::update_fab_stats (long n, long s, size_t szt)
{
    long tst = s*szt;
    BoxLib::private_total_bytes_allocated_in_fabs += tst;
    BoxLib::private_total_bytes_allocated_in_fabs_hwm 
	= std::max(BoxLib::private_total_bytes_allocated_in_fabs_hwm,
		   BoxLib::private_total_bytes_allocated_in_fabs);
	
    if(szt == sizeof(Real)) {
	BoxLib::private_total_cells_allocated_in_fabs += n;
	BoxLib::private_total_cells_allocated_in_fabs_hwm 
	    = std::max(BoxLib::private_total_cells_allocated_in_fabs_hwm,
		       BoxLib::private_total_cells_allocated_in_fabs);
    }
}

Arena*
BoxLib::The_Arena ()
{
    BL_ASSERT(the_arena != 0);

    return the_arena;
}

#if !(defined(BL_NO_FORT) || defined(WIN32))
template<>
void
BaseFab<Real>::performCopy (const BaseFab<Real>& src,
                            const Box&           srcbox,
                            int                  srccomp,
                            const Box&           destbox,
                            int                  destcomp,
                            int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    //fort_fab_copy(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
	//	  BL_TO_FORTRAN_N_3D(*this,destcomp),
	//	  BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
	//	  &numcomp);
#if (BL_SPACEDIM == 3)
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *destbox_lo = destbox.loVect();
    const int *srcbox_lo = srcbox.loVect();
    const int *cb = destbox.cbVect();
        
        
    ViewFab<Real> fab = this->view_fab; 
    ViewFab<Real> srcfab = src.view_fab; 
    
    Kokkos::Experimental::md_parallel_for(t_policy({0, 0, 0, 0}, {destbox.length(0), destbox.length(1), destbox.length(2), numcomp}, {cb[0], cb[1], cb[2], numcomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
        fab(i+destbox_lo[0], j+destbox_lo[1], k+destbox_lo[2], n+destcomp) = srcfab(i+srcbox_lo[0], j+srcbox_lo[1], k+srcbox_lo[2], n+srccomp);
    });
#else
#error "BaseFab<Real>::performCopy not implemented!"
#endif
}

template <>
void
BaseFab<Real>::copyToMem (const Box& srcbox,
                          int        srccomp,
                          int        numcomp,
                          Real*      dst) const
{
    BL_ASSERT(box().contains(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= nComp());

    if (srcbox.ok())
    {
	fort_fab_copytomem(ARLIM_3D(srcbox.loVect()), ARLIM_3D(srcbox.hiVect()),
			   dst,
			   BL_TO_FORTRAN_N_3D(*this,srccomp),
			   &numcomp);
    }
}

template <>
void
BaseFab<Real>::copyFromMem (const Box&  dstbox,
                            int         dstcomp,
                            int         numcomp,
                            const Real* src)
{
    BL_ASSERT(box().contains(dstbox));
    BL_ASSERT(dstcomp >= 0 && dstcomp+numcomp <= nComp());

    if (dstbox.ok()) 
    {
	fort_fab_copyfrommem(ARLIM_3D(dstbox.loVect()), ARLIM_3D(dstbox.hiVect()),
			     BL_TO_FORTRAN_N_3D(*this,dstcomp), &numcomp,
			     src);
    }
}

template<>
void
BaseFab<Real>::performSetVal (Real       val,
                              const Box& bx,
                              int        comp,
                              int        ncomp)
{
    BL_ASSERT(domain.contains(bx));
    BL_ASSERT(comp >= 0 && comp + ncomp <= nvar);

#if BL_SPACEDIM == 3
    //the asserts are not sufficient, check length
    for(unsigned int d=0; d<3; d++){
        if(bx.length(d)<=0) return;
    }
    
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();
        
    ViewFab<Real> fab = this->view_fab; 
    
    Kokkos::Experimental::md_parallel_for(t_policy({lo[0], lo[1], lo[2], comp}, {hi[0]+1, hi[1]+1, hi[2]+1, comp+ncomp}, {cb[0], cb[1], cb[2], ncomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
        fab(i,j,k,n) = val;
    });
#else
#error "BaseFab::performSetVal needs to be implemented for other spacedims"
#endif

    //fort_fab_setval(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
	//	    BL_TO_FORTRAN_N_3D(*this,comp), &ncomp,
	//	    &val);
}

template<>
BaseFab<Real>&
BaseFab<Real>::invert (Real       val,
                       const Box& bx,
                       int        comp,
                       int        ncomp)
{
    BL_ASSERT(domain.contains(bx));
    BL_ASSERT(comp >= 0 && comp + ncomp <= nvar);

    fort_fab_invert(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
		    BL_TO_FORTRAN_N_3D(*this,comp), &ncomp,
		    &val);
    return *this;
}

template<>
Real
BaseFab<Real>::norm (const Box& bx,
                     int        p,
                     int        comp,
                     int        ncomp) const
{
    BL_ASSERT(domain.contains(bx));
    BL_ASSERT(comp >= 0 && comp + ncomp <= nvar);

    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();

    Real nrm = 0.;

    if (p == 0)
    {
#if BL_SPACEDIM == 3
        //define policy
        typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
        ViewFab<Real> fab = this->view_fab; 
        
        Kokkos::Experimental::md_parallel_reduce(t_policy({lo[0], lo[1], lo[2], comp}, {hi[0]+1, hi[1]+1, hi[2]+1, comp+ncomp}, {cb[0], cb[1], cb[2], ncomp}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int n, Real& tmpnrm){
            tmpnrm = std::max(tmpnrm, static_cast<Real>(std::abs(fab(i,j,k,n))));
        }, nrm);
#else
#error "BaseFab::norm needs to be implemented for other spacedims"
#endif
    }
    else if (p == 1)
    {
#if BL_SPACEDIM == 3
        //define policy
        typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
        ViewFab<Real> fab = this->view_fab; 
        
        Kokkos::Experimental::md_parallel_reduce(t_policy({lo[0], lo[1], lo[2], comp}, {hi[0]+1, hi[1]+1, hi[2]+1, comp+ncomp}, {cb[0], cb[1], cb[2], ncomp}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int n, Real& tmpnrm){
            tmpnrm += static_cast<Real>(std::abs(fab(i,j,k,n)));
        }, nrm);
#else
#error "BaseFab::norm needs to be implemented for other spacedims"
#endif
    }

//    if (p == 0 || p == 1)
//    {
//	nrm = fort_fab_norm(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
   //			    BL_TO_FORTRAN_N_3D(*this,comp), &ncomp,
   //			    &p);
//    }
//    else
//    {
//        BoxLib::Error("BaseFab<Real>::norm(): only p == 0 or p == 1 are supported");
//    }

    return nrm;
}

template<>
Real
BaseFab<Real>::sum (const Box& bx,
                    int        comp,
                    int        ncomp) const
{
    BL_ASSERT(domain.contains(bx));
    BL_ASSERT(comp >= 0 && comp + ncomp <= nvar);

    Real sum = 0;

#if BL_SPACEDIM == 3
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *lo = bx.loVect();
    const int *hi = bx.hiVect();
    const int *cb = bx.cbVect();
        
    ViewFab<Real> fab = this->view_fab; 
        
    Kokkos::Experimental::md_parallel_reduce(t_policy({lo[0], lo[1], lo[2], comp}, {hi[0]+1, hi[1]+1, hi[2]+1, comp+ncomp}, {cb[0], cb[1], cb[2], ncomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n, Real& tmpsum){
        tmpsum += fab(i,j,k,n);
    },sum);
#else
#error "BaseFab::norm needs to be implemented for other spacedims"
#endif
    
    return sum;

    //return fort_fab_sum(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
	//		BL_TO_FORTRAN_N_3D(*this,comp), &ncomp);
}

template<>
BaseFab<Real>&
BaseFab<Real>::plus (const BaseFab<Real>& src,
                     const Box&           srcbox,
                     const Box&           destbox,
                     int                  srccomp,
                     int                  destcomp,
                     int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    //fort_fab_plus(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
	//	  BL_TO_FORTRAN_N_3D(*this,destcomp),
	//	  BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
	//	  &numcomp);
    
#if BL_SPACEDIM == 3    
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *lo = destbox.loVect();
    const int *hi = destbox.hiVect();
    const int *cb = destbox.cbVect();
    const int *sblo = srcbox.loVect();
        
    ViewFab<Real> destfab = this->view_fab; 
    ViewFab<Real> srcfab = src.view_fab; 
    
    Kokkos::Experimental::md_parallel_for(t_policy({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, numcomp}, {cb[0], cb[1], cb[2], numcomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
        const int ioff = sblo[0] - lo[0];
        const int joff = sblo[1] - lo[1];
        const int koff = sblo[2] - lo[2];
        destfab(i,j,k,n+destcomp) += srcfab(i+ioff,j+joff,k+koff,n+srccomp);
    });
#else
#error "BaseFab::performSetVal needs to be implemented for other spacedims"
#endif

    return *this;
}

template<>
BaseFab<Real>&
BaseFab<Real>::mult (const BaseFab<Real>& src,
                     const Box&           srcbox,
                     const Box&           destbox,
                     int                  srccomp,
                     int                  destcomp,
                     int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    fort_fab_mult(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
		  BL_TO_FORTRAN_N_3D(*this,destcomp),
		  BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
		  &numcomp);
    return *this;
}

template <>
BaseFab<Real>&
BaseFab<Real>::saxpy (Real a, const BaseFab<Real>& src,
                      const Box&        srcbox,
                      const Box&        destbox,
                      int               srccomp,
                      int               destcomp,
                      int               numcomp)
{
    BL_ASSERT(srcbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(destbox.ok());
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT( srccomp >= 0 &&  srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <=     nComp());

    fort_fab_saxpy(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
		   BL_TO_FORTRAN_N_3D(*this,destcomp),
		   &a,
		   BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
		   &numcomp);
    return *this;
}

template <>
BaseFab<Real>&
BaseFab<Real>::xpay (Real a, const BaseFab<Real>& src,
		     const Box&        srcbox,
		     const Box&        destbox,
		     int               srccomp,
		     int               destcomp,
		     int               numcomp)
{
    BL_ASSERT(srcbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(destbox.ok());
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT( srccomp >= 0 &&  srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <=     nComp());

    //fort_fab_xpay(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
	//	  BL_TO_FORTRAN_N_3D(*this,destcomp),
	//	  &a,
	//	  BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
	//	  &numcomp);
    
#if BL_SPACEDIM == 3    
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *lo = destbox.loVect();
    const int *hi = destbox.hiVect();
    const int *cb = destbox.cbVect();
    const int *sblo = srcbox.loVect();
        
    ViewFab<Real> destfab = this->view_fab; 
    ViewFab<Real> srcfab = src.view_fab; 
    
    Kokkos::Experimental::md_parallel_for(t_policy({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, numcomp}, {cb[0], cb[1], cb[2], numcomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
        const int ioff = sblo[0] - lo[0];
        const int joff = sblo[1] - lo[1];
        const int koff = sblo[2] - lo[2];
        destfab(i,j,k,n+destcomp) = a * destfab(i,j,k,n+destcomp) + srcfab(i+ioff,j+joff,k+koff,n+srccomp);
    });
#else
#error "BaseFab::performSetVal needs to be implemented for other spacedims"
#endif
    
    return *this;
}

template <>
BaseFab<Real>&
BaseFab<Real>::addproduct (const Box&           destbox,
			   int                  destcomp,
			   int                  numcomp,
			   const BaseFab<Real>& src1,
			   int                  comp1,
			   const BaseFab<Real>& src2,
			   int                  comp2)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(   comp1 >= 0 &&    comp1+numcomp <= src1.nComp());
    BL_ASSERT(   comp2 >= 0 &&    comp2+numcomp <= src2.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <=      nComp());

    fort_fab_addproduct(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
			BL_TO_FORTRAN_N_3D(*this,destcomp),
			BL_TO_FORTRAN_N_3D(src1,comp1),
			BL_TO_FORTRAN_N_3D(src2,comp2),
			&numcomp);
    return *this;
}

template<>
BaseFab<Real>&
BaseFab<Real>::minus (const BaseFab<Real>& src,
                      const Box&           srcbox,
                      const Box&           destbox,
                      int                  srccomp,
                      int                  destcomp,
                      int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    fort_fab_minus(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
		   BL_TO_FORTRAN_N_3D(*this,destcomp),
		   BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
		   &numcomp);
    return *this;
}

template<>
BaseFab<Real>&
BaseFab<Real>::divide (const BaseFab<Real>& src,
                       const Box&           srcbox,
                       const Box&           destbox,
                       int                  srccomp,
                       int                  destcomp,
                       int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    fort_fab_divide(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
		    BL_TO_FORTRAN_N_3D(*this,destcomp),
		    BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
		    &numcomp);
    return *this;
}

template<>
BaseFab<Real>&
BaseFab<Real>::protected_divide (const BaseFab<Real>& src,
                                 const Box&           srcbox,
                                 const Box&           destbox,
                                 int                  srccomp,
                                 int                  destcomp,
                                 int                  numcomp)
{
    BL_ASSERT(destbox.ok());
    BL_ASSERT(src.box().contains(srcbox));
    BL_ASSERT(box().contains(destbox));
    BL_ASSERT(destbox.sameSize(srcbox));
    BL_ASSERT(srccomp >= 0 && srccomp+numcomp <= src.nComp());
    BL_ASSERT(destcomp >= 0 && destcomp+numcomp <= nComp());

    fort_fab_protdivide(ARLIM_3D(destbox.loVect()), ARLIM_3D(destbox.hiVect()),
			BL_TO_FORTRAN_N_3D(*this,destcomp),
			BL_TO_FORTRAN_N_3D(src,srccomp), ARLIM_3D(srcbox.loVect()),
			&numcomp);
    return *this;
}

template <>
BaseFab<Real>&
BaseFab<Real>::linComb (const BaseFab<Real>& f1,
			const Box&           b1,
			int                  comp1,
			const BaseFab<Real>& f2,
			const Box&           b2,
			int                  comp2,
			Real                 alpha,
			Real                 beta,
			const Box&           b,
			int                  comp,
			int                  numcomp)
{
    BL_ASSERT(b1.ok());
    BL_ASSERT(f1.box().contains(b1));
    BL_ASSERT(b2.ok());
    BL_ASSERT(f2.box().contains(b2));
    BL_ASSERT(b.ok());
    BL_ASSERT(box().contains(b));
    BL_ASSERT(b.sameSize(b1));
    BL_ASSERT(b.sameSize(b2));
    BL_ASSERT(comp1 >= 0 && comp1+numcomp <= f1.nComp());
    BL_ASSERT(comp2 >= 0 && comp2+numcomp <= f2.nComp());
    BL_ASSERT(comp  >= 0 && comp +numcomp <=    nComp());

    //fort_fab_lincomb(ARLIM_3D(b.loVect()), ARLIM_3D(b.hiVect()),
	//	     BL_TO_FORTRAN_N_3D(*this,comp),
	//	     &alpha, BL_TO_FORTRAN_N_3D(f1,comp1), ARLIM_3D(b1.loVect()),
	//	     &beta,  BL_TO_FORTRAN_N_3D(f2,comp2), ARLIM_3D(b2.loVect()),
	//	     &numcomp);
    
#if BL_SPACEDIM == 3    
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
        
    const int *lo = b.loVect();
    const int *hi = b.hiVect();
    const int *cb = b.cbVect();
    const int *f1lo = b1.loVect();
    const int *f2lo = b2.loVect();
        
    ViewFab<Real> destv = this->view_fab; 
    ViewFab<Real> f1v = f1.view_fab; 
    ViewFab<Real> f2v = f2.view_fab; 
    
    Kokkos::Experimental::md_parallel_for(t_policy({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, numcomp}, {cb[0], cb[1], cb[2], numcomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n){
        const int ioff1 = f1lo[0] - lo[0];
        const int joff1 = f1lo[1] - lo[1];
        const int koff1 = f1lo[2] - lo[2];
        const int ioff2 = f2lo[0] - lo[0];
        const int joff2 = f2lo[1] - lo[1];
        const int koff2 = f2lo[2] - lo[2];
        destv(i,j,k,n+comp) = alpha*f1v(i+ioff1,j+joff1,k+koff1,n+comp1) + beta*f2v(i+ioff2,j+joff2,k+koff2,n+comp2);
    });
#else
#error "BaseFab::performSetVal needs to be implemented for other spacedims"
#endif
    
    return *this;
}

template <>
Real
BaseFab<Real>::dot (const Box& xbx, int xcomp, 
		    const BaseFab<Real>& y, const Box& ybx, int ycomp,
		    int numcomp) const
{
    BL_ASSERT(xbx.ok());
    BL_ASSERT(box().contains(xbx));
    BL_ASSERT(y.box().contains(ybx));
    BL_ASSERT(xbx.sameSize(ybx));
    BL_ASSERT(xcomp >= 0 && xcomp+numcomp <=   nComp());
    BL_ASSERT(ycomp >= 0 && ycomp+numcomp <= y.nComp());

    //return fort_fab_dot(ARLIM_3D(xbx.loVect()), ARLIM_3D(xbx.hiVect()),
	//		BL_TO_FORTRAN_N_3D(*this,xcomp),
	//		BL_TO_FORTRAN_N_3D(y,ycomp), ARLIM_3D(ybx.loVect()),
	//		&numcomp);
    Real res = 0.;
    
#if BL_SPACEDIM == 3
    //define policy
    typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4, outer_iter_policy, inner_iter_policy> > t_policy;
    
    const int *lo = xbx.loVect();
    const int *hi = xbx.hiVect();
    const int *cb = xbx.cbVect();
    const int *ylo = ybx.loVect();
        
    ViewFab<Real> xv = this->view_fab;
    ViewFab<Real> yv = y.view_fab;
    
    Kokkos::Experimental::md_parallel_reduce(t_policy({lo[0], lo[1], lo[2], 0}, {hi[0]+1, hi[1]+1, hi[2]+1, numcomp}, {cb[0], cb[1], cb[2], numcomp}), 
    KOKKOS_LAMBDA(const int i, const int j, const int k, const int n, Real& tmpres){
        const int ioff = ylo[0] - lo[0];
        const int joff = ylo[1] - lo[1];
        const int koff = ylo[2] - lo[2];
        tmpres += xv(i,j,k,n+xcomp)*yv(i+ioff,j+joff,k+koff,n+ycomp);
    }, res);
#else
#error "BaseFab::norm needs to be implemented for other spacedims"
#endif
    return res;
}

#endif
