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

void C_AVERAGE(const Box* bx,
        const int ng,
        const int nc,
        Real* c,
        const Real* f);

void C_INTERP(const Box* bx,
        const int ng,
        const int nc,
        Real* f,
        const Real* c);

#if (BL_SPACEDIM == 3)
void C_GSRB_3D(const Box* bx,
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
