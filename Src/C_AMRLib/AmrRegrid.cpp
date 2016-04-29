
#include <iostream>

#include <AmrRegrid.H>
#include <ParallelDescriptor.H>
#include <ParmParse.H>

namespace
{
    bool initialized        = false;
    bool refine_grid_layout = true;
}

AmrRegrid::AmrRegrid (int max_level_,
		      Array<int>& blocking_factor_,
		      Array<int>& max_grid_size_, 
		      const Array<Geometry>& geom_)
    : max_level(max_level_),
      blocking_factor(blocking_factor_),
      max_grid_size(max_grid_size_),
      geom(geom_)
{ 
    if (!initialized) {
	initialized = true;
	ParmParse pp("amr");
	pp.query("refine_grid_layout", refine_grid_layout);
    }
}

AmrRegrid::~AmrRegrid ()
{ }

void
AmrRegrid::grid_places (Real             time,
			int              lbase,
			int              finest_level,
			int&             new_finest,
			Array<BoxArray>& new_grids)
{
    BL_PROFILE("AmrRegrid::grid_places()");

    int i, max_crse = std::min(finest_level,max_level-1);

    const Real strttime = ParallelDescriptor::second();

    if (lbase == 0)
    {
        //
        // Recalculate level 0 BoxArray in case max_grid_size[0] has changed.
        // This is done exactly as in defBaseLev().
        //
        BoxArray lev0(1);

        lev0.set(0,BoxLib::coarsen(geom[0].Domain(),2));
        //
        // Now split up into list of grids within max_grid_size[0] limit.
        //
        lev0.maxSize(max_grid_size[0]/2);
        //
        // Now refine these boxes back to level 0.
        //
        lev0.refine(2);

        new_grids[0] = lev0;

       // If Nprocs > Ngrids and refine_grid_layout == 1 then break up the grids
       //    into smaller chunks for better load balancing
       // We need to impose this here in the event that we return with fixed_grids
       //    and never have a chance to impose it later
       impose_refine_grid_layout(lbase,lbase,new_grids);
    }

    if ( time == 0. && !initial_grids_file.empty() && !useFixedCoarseGrids)
    {
        new_finest = std::min(max_level,(finest_level+1));
        new_finest = std::min(new_finest,initial_ba.size());

        for (int lev = 1; lev <= new_finest; lev++)
        {
            BoxList bl;
            int ngrid = initial_ba[lev-1].size();
            for (i = 0; i < ngrid; i++)
            {
                Box bx(initial_ba[lev-1][i]);
                if (lev > lbase)
                    bl.push_back(bx);
            }
            if (lev > lbase)
                new_grids[lev].define(bl);
        }
        return;
    }

#if 0

    // Use grids in initial_grids_file as fixed coarse grids.
    if ( !initial_grids_file.empty() && useFixedCoarseGrids)
    {
        new_finest = std::min(max_level,(finest_level+1));
        new_finest = std::min(new_finest,initial_ba.size());

        for (int lev = lbase+1; lev <= new_finest; lev++)
        {
            BoxList bl;
            int ngrid = initial_ba[lev-1].size();
            for (i = 0; i < ngrid; i++)
            {
                Box bx(initial_ba[lev-1][i]);

                if (lev > lbase)
                    bl.push_back(bx);

            }
            if (lev > lbase)
                new_grids[lev].define(bl);
            new_grids[lev].maxSize(max_grid_size[lev]);
        }
    }

    // Use grids in regrid_grids_file 
    else if ( !regrid_grids_file.empty() )
    {
        new_finest = std::min(max_level,(finest_level+1));
        new_finest = std::min(new_finest,regrid_ba.size());
        for (int lev = 1; lev <= new_finest; lev++)
        {
            BoxList bl;
            int ngrid = regrid_ba[lev-1].size();
            for (i = 0; i < ngrid; i++)
            {
                Box bx(regrid_ba[lev-1][i]);
                if (lev > lbase)
                    bl.push_back(bx);
            }
            if (lev > lbase)
                new_grids[lev].define(bl);
        }
        return;
    }

    //
    // Construct problem domain at each level.
    //
    Array<IntVect> bf_lev(max_level); // Blocking factor at each level.
    Array<IntVect> rr_lev(max_level);
    Array<Box>     pc_domain(max_level);  // Coarsened problem domain.
    for (i = 0; i <= max_crse; i++)
    {
        for (int n=0; n<BL_SPACEDIM; n++)
            bf_lev[i][n] = std::max(1,blocking_factor[i+1]/ref_ratio[i][n]);
    }
    for (i = lbase; i < max_crse; i++)
    {
        for (int n=0; n<BL_SPACEDIM; n++)
            rr_lev[i][n] = (ref_ratio[i][n]*bf_lev[i][n])/bf_lev[i+1][n];
    }
    for (i = lbase; i <= max_crse; i++)
        pc_domain[i] = BoxLib::coarsen(geom[i].Domain(),bf_lev[i]);
    //
    // Construct proper nesting domains.
    //
    Array<BoxList> p_n(max_level);      // Proper nesting domain.
    Array<BoxList> p_n_comp(max_level); // Complement proper nesting domain.

    BoxList bl(amr_level[lbase].boxArray());
    bl.simplify();
    bl.coarsen(bf_lev[lbase]);
    p_n_comp[lbase].complementIn(pc_domain[lbase],bl);
    p_n_comp[lbase].simplify();
    p_n_comp[lbase].accrete(n_proper);
    Amr::ProjPeriodic(p_n_comp[lbase], Geometry(pc_domain[lbase]));
    p_n[lbase].complementIn(pc_domain[lbase],p_n_comp[lbase]);
    p_n[lbase].simplify();
    bl.clear();

    for (i = lbase+1; i <= max_crse; i++)
    {
        p_n_comp[i] = p_n_comp[i-1];

        // Need to simplify p_n_comp or the number of grids can too large for many levels.
        p_n_comp[i].simplify();

        p_n_comp[i].refine(rr_lev[i-1]);
        p_n_comp[i].accrete(n_proper);

        Amr::ProjPeriodic(p_n_comp[i], Geometry(pc_domain[i]));

        p_n[i].complementIn(pc_domain[i],p_n_comp[i]);
        p_n[i].simplify();
    }
    //
    // Now generate grids from finest level down.
    //
    new_finest = lbase;

    for (int levc = max_crse; levc >= lbase; levc--)
    {
        int levf = levc+1;
        //
        // Construct TagBoxArray with sufficient grow factor to contain
        // new levels projected down to this level.
        //
        int ngrow = 0;

        if (levf < new_finest)
        {
            BoxArray ba_proj(new_grids[levf+1]);

            ba_proj.coarsen(ref_ratio[levf]);
            ba_proj.grow(n_proper);
            ba_proj.coarsen(ref_ratio[levc]);

            BoxArray levcBA = amr_level[levc].boxArray();

            while (!levcBA.contains(ba_proj))
            {
                BoxArray tmp = levcBA;
                tmp.grow(1);
                levcBA = tmp;
                ngrow++;
            }
        }
        TagBoxArray tags(amr_level[levc].boxArray(),n_error_buf[levc]+ngrow);
    
        //
        // Only use error estimation to tag cells for the creation of new grids
        //      if the grids at that level aren't already fixed.
        //
        if ( !(useFixedCoarseGrids && levc < useFixedUpToLevel) )
            amr_level[levc].errorEst(tags,
                                     TagBox::CLEAR,TagBox::SET,time,
                                     n_error_buf[levc],ngrow);
        //
        // If new grids have been constructed above this level, project
        // those grids down and tag cells on intersections to ensure
        // proper nesting.
        //
        // NOTE: this loop replaces the previous code:
        //      if (levf < new_finest) 
        //          tags.setVal(ba_proj,TagBox::SET);
        // The problem with this code is that it effectively 
        // "buffered the buffer cells",  i.e., the grids at level
        // levf+1 which were created by buffering with n_error_buf[levf]
        // are then coarsened down twice to define tagging at
        // level levc, which will then also be buffered.  This can
        // create grids which are larger than necessary.
        //
        if (levf < new_finest)
        {
            int nerr = n_error_buf[levf];

            BoxList bl_tagged(new_grids[levf+1]);
            bl_tagged.simplify();
            bl_tagged.coarsen(ref_ratio[levf]);
            //
            // This grows the boxes by nerr if they touch the edge of the
            // domain in preparation for them being shrunk by nerr later.
            // We want the net effect to be that grids are NOT shrunk away
            // from the edges of the domain.
            //
            for (BoxList::iterator blt = bl_tagged.begin(), End = bl_tagged.end();
                 blt != End;
                 ++blt)
            {
                for (int idir = 0; idir < BL_SPACEDIM; idir++)
                {
                    if (blt->smallEnd(idir) == geom[levf].Domain().smallEnd(idir))
                        blt->growLo(idir,nerr);
                    if (blt->bigEnd(idir) == geom[levf].Domain().bigEnd(idir))
                        blt->growHi(idir,nerr);
                }
            }
            Box mboxF = BoxLib::grow(bl_tagged.minimalBox(),1);
            BoxList blFcomp;
            blFcomp.complementIn(mboxF,bl_tagged);
            blFcomp.simplify();
            bl_tagged.clear();

            const IntVect& iv = IntVect(D_DECL(nerr/ref_ratio[levf][0],
                                               nerr/ref_ratio[levf][1],
                                               nerr/ref_ratio[levf][2]));
            blFcomp.accrete(iv);
            BoxList blF;
            blF.complementIn(mboxF,blFcomp);
            BoxArray baF(blF);
            blF.clear();
            baF.grow(n_proper);
            //
            // We need to do this in case the error buffering at
            // levc will not be enough to cover the error buffering
            // at levf which was just subtracted off.
            //
            for (int idir = 0; idir < BL_SPACEDIM; idir++) 
            {
                if (nerr > n_error_buf[levc]*ref_ratio[levc][idir]) 
                    baF.grow(idir,nerr-n_error_buf[levc]*ref_ratio[levc][idir]);
            }

            baF.coarsen(ref_ratio[levc]);

            tags.setVal(baF,TagBox::SET);
        }
        //
        // Buffer error cells.
        //
        tags.buffer(n_error_buf[levc]+ngrow);

        if (useFixedCoarseGrids)
        {
           if (levc>=useFixedUpToLevel)
           {
               BoxArray bla(amr_level[levc].getAreaNotToTag());
               tags.setVal(bla,TagBox::CLEAR);
           }
           if (levc<useFixedUpToLevel)
              new_finest = std::max(new_finest,levf);
        }

        //
        // Coarsen the taglist by blocking_factor/ref_ratio.
        //
        int bl_max = 0;
        for (int n=0; n<BL_SPACEDIM; n++)
            bl_max = std::max(bl_max,bf_lev[levc][n]);
        if (bl_max > 1) 
            tags.coarsen(bf_lev[levc]);
        //
        // Remove or add tagged points which violate/satisfy additional 
        // user-specified criteria.
        //
        amr_level[levc].manual_tags_placement(tags, bf_lev);
        //
        // Map tagged points through periodic boundaries, if any.
        //
        tags.mapPeriodic(Geometry(pc_domain[levc]));
        //
        // Remove cells outside proper nesting domain for this level.
        //
        tags.setVal(p_n_comp[levc],TagBox::CLEAR);
        //
        // Create initial cluster containing all tagged points.
        //
        long     len = 0;
        IntVect* pts = tags.collate(len);

        tags.clear();

        if (len > 0)
        {
            //
            // Created new level, now generate efficient grids.
            //
            if ( !(useFixedCoarseGrids && levc<useFixedUpToLevel) )
                new_finest = std::max(new_finest,levf);
            //
            // Construct initial cluster.
            //
            ClusterList clist(pts,len);
            clist.chop(grid_eff);
            BoxDomain bd;
            bd.add(p_n[levc]);
            clist.intersect(bd);
            bd.clear();
            //
            // Efficient properly nested Clusters have been constructed
            // now generate list of grids at level levf.
            //
            BoxList new_bx;
            clist.boxList(new_bx);
            new_bx.refine(bf_lev[levc]);
            new_bx.simplify();
            BL_ASSERT(new_bx.isDisjoint());
	    
	    if (new_bx.size()>0) {
	      if ( !(geom[levc].Domain().contains(BoxArray(new_bx).minimalBox())) ) {
		// Chop new grids outside domain, note that this is likely to result in
		//  new grids that violate blocking_factor....see warning checking below
		new_bx = BoxLib::intersect(new_bx,geom[levc].Domain());
	      }
	    }

            IntVect largest_grid_size;
            for (int n = 0; n < BL_SPACEDIM; n++)
                largest_grid_size[n] = max_grid_size[levf] / ref_ratio[levc][n];
            //
            // Ensure new grid boxes are at most max_grid_size in index dirs.
            //
            new_bx.maxSize(largest_grid_size);

#ifdef BL_FIX_GATHERV_ERROR
	      int wcount = 0, iLGS = largest_grid_size[0];

              while (new_bx.size() < 64 && wcount++ < 4)
              {
                  iLGS /= 2;
                  if (ParallelDescriptor::IOProcessor())
                  {
                      std::cout << "BL_FIX_GATHERV_ERROR:  using iLGS = " << iLGS
                                << "   largest_grid_size was:  " << largest_grid_size[0]
                                << '\n';
                      std::cout << "BL_FIX_GATHERV_ERROR:  new_bx.size() was:   "
                                << new_bx.size() << '\n';
                  }

                  new_bx.maxSize(iLGS);

                  if (ParallelDescriptor::IOProcessor())
                  {
                      std::cout << "BL_FIX_GATHERV_ERROR:  new_bx.size() now:   "
                                << new_bx.size() << '\n';
                  }
	      }
#endif
            //
            // Refine up to levf.
            //
            new_bx.refine(ref_ratio[levc]);
            BL_ASSERT(new_bx.isDisjoint());

	    if (new_bx.size()>0) {
	      if ( !(geom[levf].Domain().contains(BoxArray(new_bx).minimalBox())) ) {
		new_bx = BoxLib::intersect(new_bx,geom[levf].Domain());
	      }
	      if (ParallelDescriptor::IOProcessor()) {
		for (int d=0; d<BL_SPACEDIM; ++d) {
		  bool ok = true;
		  for (BoxList::const_iterator bli = new_bx.begin(); bli != new_bx.end(); ++bli) {
		    int len = bli->length(d);
		    int bf = blocking_factor[levf];
		    ok &= (len/bf) * bf == len;
		  }
		  if (!ok) {
		    BoxLib::Warning("WARNING: New grids violate blocking factor near upper boundary");
		  }
		}
	      }
	    }

            if (levf > useFixedUpToLevel)
                new_grids[levf].define(new_bx);
        }
        //
        // Don't forget to get rid of space used for collate()ing.
        //
        delete [] pts;
    }

    // If Nprocs > Ngrids and refine_grid_layout == 1 then break up the grids
    //    into smaller chunks for better load balancing
    impose_refine_grid_layout(lbase,new_finest,new_grids);

    if (verbose > 0)
    {
        Real stoptime = ParallelDescriptor::second() - strttime;

#ifdef BL_LAZY
	Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(stoptime,ParallelDescriptor::IOProcessorNumber());
        if (ParallelDescriptor::IOProcessor())
            std::cout << "grid_places() time: " << stoptime << " new finest: " << new_finest<< '\n';
#ifdef BL_LAZY
	});
#endif
    }


#endif
}

void 
AmrRegrid::impose_refine_grid_layout (int lbase, int new_finest, Array<BoxArray>& new_grids)
{
    const int NProcs = ParallelDescriptor::NProcs();
    if (NProcs > 1 && refine_grid_layout)
    {
        //
        // Chop up grids if fewer grids at level than CPUs.
        // The idea here is to make more grids on a given level
        // to spread the work around.
        //
        for (int cnt = 1; cnt <= 4; cnt *= 2)
        {
            for (int i = lbase; i <= new_finest; i++)
            {
                const int ChunkSize = max_grid_size[i]/cnt;

                IntVect chunk(D_DECL(ChunkSize,ChunkSize,ChunkSize));
                //
                // We go from Z -> Y -> X to promote cache-efficiency.
                //
                for (int j = BL_SPACEDIM-1; j >= 0 ; j--)
                {
                    chunk[j] /= 2;

                    if ( (new_grids[i].size() < NProcs) && (chunk[j]%blocking_factor[i] == 0) )
                    {
                        new_grids[i].maxSize(chunk);
                    }
                }
            }
        }
    }
}
