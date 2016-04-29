#ifndef _AMR_REGRID_H_
#define _AMR_REGRID_H_

#include <BoxArray.H>
#include <Geometry.H>

class AmrRegrid
{
public:

    AmrRegrid (int max_level_, 
	       Array<int>& blocking_factor_,
	       Array<int>& max_grid_size_, 
	       const Array<Geometry>& geom_);

    ~AmrRegrid ();

    void grid_places (Real             time,
		      int              lbase,
		      int              finest_level,
		      int&             new_finest,
		      Array<BoxArray>& new_grids);

    void impose_refine_grid_layout (int lbase, int new_finest, Array<BoxArray>& new_grids);

private:
    int             max_level;
    Array<int>      blocking_factor;
    Array<int>      max_grid_size;
    Array<Geometry> geom;
};

#endif
