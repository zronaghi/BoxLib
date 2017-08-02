
#include <winstd.H>
#include <BndryData.H>
#include <Utility.H>
#include <LO_BCTYPES.H>
#include <ParallelDescriptor.H>

//
// Mask info required for this many cells past grid edge
//  (here, e.g. ref=4, crse stencil width=3, and let stencil slide 2 past grid edge)
//
int BndryData::NTangHalfWidth = 5;  // ref_ratio + 1, so won't work if ref_ratio > 4

BndryData::BndryData ()
    :
m_ncomp(-1), m_defined(false) {}

BndryData::BndryData (const BoxArray& _grids,
int             _ncomp, 
const Geometry& _geom,
ParallelDescriptor::Color color)
    :
geom(_geom),
m_ncomp(_ncomp),
m_defined(false)
{
    define(_grids,_ncomp,_geom,color);
}

void
    BndryData::setBoundCond (Orientation     _face,
int              _n,
int              _comp,
const BoundCond& _bcn)
{
    bcond[_n][_face][_comp] = _bcn;
}

void
    BndryData::setBoundLoc (Orientation _face,
int         _n,
Real        _val)
{
    bcloc[_n][_face] = _val;
}

const Array< Array<BoundCond> >&
    BndryData::bndryConds (int igrid) const
{
    std::map< int, Array< Array<BoundCond> > >::const_iterator it = bcond.find(igrid);
    BL_ASSERT(it != bcond.end());
    return it->second;
}

const BndryData::RealTuple&
    BndryData::bndryLocs (int igrid) const
{
    std::map<int,RealTuple>::const_iterator it = bcloc.find(igrid);
    BL_ASSERT(it != bcloc.end());
    return it->second;
}

void
    BndryData::init (const BndryData& src)
{
    geom      = src.geom;
    m_ncomp   = src.m_ncomp;
    m_defined = src.m_defined;
    bcloc     = src.bcloc;
    bcond     = src.bcond;

    masks.clear();
    masks.resize(2*BL_SPACEDIM, PArrayManage);
    for (int i = 0; i < 2*BL_SPACEDIM; i++)
    {
        const MultiMask& smasks = src.masks[i];
        masks.set(i, new MultiMask(smasks.boxArray(), smasks.DistributionMap(), smasks.nComp()));
        Copy(masks[i], smasks);
    }
}

BndryData::BndryData (const BndryData& src)
    :
BndryRegister(src),
m_ncomp(src.m_ncomp)
{
    init(src);
}

BndryData&
    BndryData::operator= (const BndryData& src)
{
    if (this != &src)
    {
        BndryRegister::operator=(src);
        for (int i = 0; i < 2*BL_SPACEDIM; i++) {
            bndry[i].clear();
        }
        init(src);
    }
    return *this;
}

BndryData::~BndryData ()
{
}

void
    BndryData::define (const BoxArray& _grids,
int             _ncomp,
const Geometry& _geom,
ParallelDescriptor::Color color)
{
    BL_PROFILE("BndryData::define()");

    if (m_defined)
    {
        if (_grids == boxes() && m_ncomp == _ncomp && _geom.Domain() == geom.Domain())
            //
            // We want to allow reuse of BndryData objects that were define()d exactly as a previous call.
            //
            return;
        //
        // Otherwise we'll just abort.  We could make this work but it's just as easy to start with a fresh Bndrydata object.
        //
        BoxLib::Abort("BndryData::define(): object already built");
    }
    geom    = _geom;
    m_ncomp = _ncomp;
    
    BndryRegister::setBoxes(_grids);

    masks.clear();
    masks.resize(2*BL_SPACEDIM, PArrayManage);

    for (OrientationIter fi; fi; ++fi)
    {
        Orientation face = fi();
        
        BndryRegister::define(face,IndexType::TheCellType(),0,1,1,_ncomp,color);
        
        masks.set(face, new MultiMask(grids, bndry[face].DistributionMap(), geom, face, 0, 2, NTangHalfWidth, 1, true));
    }
    
    //
    // Define "bcond" and "bcloc".
    //
    // We note that all orientations of the FabSets have the same distribution.
    // We'll use the low 0 side as the model.
    //
    //
    for (FabSetIter bfsi(bndry[Orientation(0,Orientation::low)]);
    bfsi.isValid();
    ++bfsi)
    {
        const int idx = bfsi.index();
        //
        // Insert with a hint since we know the indices are increasing.
        //
        bcloc.insert(bcloc.end(),std::map<int,RealTuple>::value_type(idx,RealTuple()));

        std::map< int, Array< Array<BoundCond> > >::value_type v(idx,Array< Array<BoundCond> >());

        Array< Array<BoundCond> >& abc = bcond.insert(bcond.end(),v)->second;

        abc.resize(2*BL_SPACEDIM);

        for (OrientationIter fi; fi; ++fi)
        {
            abc[fi()].resize(_ncomp);
        }
    }

    m_defined = true;
}

