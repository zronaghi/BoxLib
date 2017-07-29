
#include <winstd.H>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

#include <IArrayBox.H>
#include <ParmParse.H>
#include <FPC.H>

#include <BLassert.H>
#include <BoxLib.H>
#include <Looping.H>
#include <Utility.H>

namespace
{
    bool initialized = false;
}

IArrayBox::IArrayBox () {}

IArrayBox::IArrayBox (const Box& b,
                      int        n,
		      bool       alloc,
		      bool       shared)
    :
    BaseFab<int>(b,n,alloc,shared)
{
    //
    // For debugging purposes set values to QNAN when possible.
    //
    if ( alloc && do_initval )
	setVal(0);
}

IArrayBox&
IArrayBox::operator= (const int& v)
{
    BaseFab<int>::operator=(v);
    return *this;
}

void
IArrayBox::resize (const Box& b,
                   int        N)
{
    BaseFab<int>::resize(b,N);
    //
    // For debugging purposes set values to QNAN when possible.
    //
    if ( do_initval )
        setVal(0);
}

int
IArrayBox::norm (int p,
                 int comp,
                 int numcomp) const
{
    return norm(domain,p,comp,numcomp);
}

IArrayBox::~IArrayBox () {}

#if !defined(NDEBUG)
bool IArrayBox::do_initval = true;
#else
bool IArrayBox::do_initval = false;
#endif

void
IArrayBox::Initialize ()
{
    if (initialized) return;
//    ParmParse pp("iab");
    BoxLib::ExecOnFinalize(IArrayBox::Finalize);
    initialized = true;
}

void
IArrayBox::Finalize ()
{
    initialized = false;
}

int
IArrayBox::norm (const Box& subbox,
                 int        p,
                 int        comp,
                 int        ncomp) const
{
    BL_ASSERT(p >= 0);
    BL_ASSERT(comp >= 0 && comp+ncomp <= nComp());

    int  nrm    = 0;

    if (p == 0 || p == 1)
    {
        nrm = BaseFab<int>::norm(subbox,p,comp,ncomp);
    }
    else if (p == 2)
    {
        ForAllThisCPencilAdd(int,subbox,comp,ncomp,nrm)
        {
            redR += thisR*thisR;

        } EndForPencil(nrm)
        nrm = std::sqrt(double(nrm));
    }
    else
    {
        ForAllThisCPencilAdd(int,subbox,comp,ncomp,nrm)
        {
            redR += std::pow(static_cast<double>(thisR), p);
        } EndForPencil(nrm)
        nrm = std::pow(static_cast<double>(nrm),1./static_cast<double>(p));
    }

    return nrm;
}
