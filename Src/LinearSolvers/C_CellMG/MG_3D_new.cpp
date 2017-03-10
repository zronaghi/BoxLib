
//#undef  BL_LANG_CC
//#ifndef BL_LANG_FORT
//#define BL_LANG_FORT
//#endif

#include <REAL.H>
#include <cmath>
#include <Box.H>
#include "CONSTANTS.H"
#include "MG_F.H"
#include <ArrayLim.H>


void MG_3D(const Box* bx,
	const int ng,
	const int nc,
        Real *cp,
        Real *fp){

 
#ifdef USE_CPP_KERNELS
    //const Box* bx;
    //Real *cp, *fp;
    
    int i2, j2, k2;
    int ijkn;
    int i2p1_j2p1_k2_n, i2_j2p1_k2_n, i2p1_j2_k2_n, i2_j2_k2_n;
    int i2p1_j2p1_k2p1_n, i2_j2p1_k2p1_n, i2p1_j2_k2p1_n, i2_j2_k2p1_n;     

    //const int ng;
    const int BL_jStride = bx->length(0) + 2*ng;
    const int BL_j2Stride = 2*(bx->length(0)) + 2*ng;
    const int BL_kStride = BL_jStride * (bx->length(1) + 2*ng);
    const int BL_k2Stride = BL_j2Stride * (2*(bx->length(1)) + 2*ng);
    const int BL_nStride = BL_kStride * (bx->length(2) + 2*ng);
    const int BL_n2Stride = BL_k2Stride * (2*(bx->length(2)) + 2*ng);
    const int *lo = bx->loVect();
    const int *hi = bx->hiVect();

    //const int offset = 
    int abs_i, abs_j, abs_k;

      for (int n = 0; n<nc; n++){
         for (int k = 0; k < bx->length(2); ++k) {
            k2 = 2*k;
            //k2p1 = k2 + 1
	    for (int j = 0; j < bx->length(1); ++j) {
               j2 = 2*j;
               //j2p1 = j2 + 1
               for (int i = 0; i < bx->length(0); ++i) {
                  i2 = 2*i;
                  //i2p1 = i2 + 1
                    
                    abs_i = lo[0] + i;
                    abs_j = lo[1] + j;
                    abs_k = lo[2] + k;
  


		    ijkn = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride + n*BL_nStride;
    

//add k2Stide, n2Stride, et
                    i2_j2_k2_n =       (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2p1_k2_n =   (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2_j2p1_k2_n =     (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2_k2_n =     (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2p1_k2p1_n = (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2_j2p1_k2p1_n =   (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2p1_j2_k2p1_n =   (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2_j2_k2p1_n =     (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;

                    cp[ijkn] =  (fp[i2p1_j2p1_k2_n] + fp[i2_j2p1_k2_n] + fp[i2p1_j2_k2_n] + fp[i2_j2_k2_n])*eighth;
                    cp[ijkn] += (fp[i2p1_j2p1_k2p1_n] + fp[i2_j2p1_k2p1_n] + fp[i2p1_j2_k2p1_n] + fp[i2_j2_k2p1_n])*eighth;
               }
            }
         }
      }
     
#else

      FORT_AVERAGE (
          c, DIMS(c),
          f, DIMS(f),
          lo, hi, nc)
      implicit none
      integer nc
      integer DIMDEC(c)
      integer DIMDEC(f)
      integer lo(BL_SPACEDIM)
      integer hi(BL_SPACEDIM)
      REAL_T f(DIMV(f),nc)
      REAL_T c(DIMV(c),nc)

      integer i, i2, i2p1, j, j2, j2p1, k, k2, k2p1, n

      do n = 1, nc
         do k = lo(3), hi(3)
            k2 = 2*k
            k2p1 = k2 + 1
	    do j = lo(2), hi(2)
               j2 = 2*j
               j2p1 = j2 + 1
               do i = lo(1), hi(1)
                  i2 = 2*i
                  i2p1 = i2 + 1
                  c(i,j,k,n) =  (
                      + f(i2p1,j2p1,k2  ,n) + f(i2,j2p1,k2  ,n)
                      + f(i2p1,j2  ,k2  ,n) + f(i2,j2  ,k2  ,n)
                      + f(i2p1,j2p1,k2p1,n) + f(i2,j2p1,k2p1,n)
                      + f(i2p1,j2  ,k2p1,n) + f(i2,j2  ,k2p1,n)
                      )*eighth
               end do
            end do
         end do
      end do

      end

#endif


#ifdef USE_CPP_KERNELS

      for (int n = 0; n<nc; n++){
         for (int k = 0; k < bx->length(2); ++k) {
            k2 = 2*k;
            //k2p1 = k2 + 1
            for (int j = 0; j < bx->length(1); ++j) {
               j2 = 2*j;
               //j2p1 = j2 + 1
               for (int i = 0; i < bx->length(0); ++i) {
                  i2 = 2*i;
                  //i2p1 = i2 + 1

                    abs_i = lo[0] + i;
                    abs_j = lo[1] + j;
                    abs_k = lo[2] + k;



                    ijkn = (i + ng) + (j + ng)*BL_jStride + (k + ng)*BL_kStride + n*BL_nStride;


//add k2Stide, n2Stride, et
                    i2_j2_k2_n =       (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2p1_k2_n =   (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2_j2p1_k2_n =     (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2_k2_n =     (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + ng)*BL_k2Stride +     n*BL_n2Stride;
                    i2p1_j2p1_k2p1_n = (i2 + 1 + ng) + (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2_j2p1_k2p1_n =   (i2 + ng) +     (j2 + 1 + ng)*BL_j2Stride + (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2p1_j2_k2p1_n =   (i2 + 1 + ng) + (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;
                    i2_j2_k2p1_n =     (i2 + ng) +     (j2 + ng)*BL_j2Stride +     (k2 + 1 + ng)*BL_k2Stride + n*BL_n2Stride;

                    cp[ijkn] =  (fp[i2p1_j2p1_k2_n] + fp[i2_j2p1_k2_n] + fp[i2p1_j2_k2_n] + fp[i2_j2_k2_n])*eighth;
                    cp[ijkn] += (fp[i2p1_j2p1_k2p1_n] + fp[i2_j2p1_k2p1_n] + fp[i2p1_j2_k2p1_n] + fp[i2_j2_k2p1_n])*eighth;
                   
                    fp[i2p1_j2p1_k2_n] = cp[i_j_k_n] + f[i2p1_j2p1_k2_n]
                    f[i2_j2p1_k2_n] = cp[i_j_k_n] + f[i2_j2p1_k2_n]
                    f[i2p1_j2_k2_n] = cp[i_j_k_n] + f[i2p1_j2_k2_n]
                    f[i2_j2_k2_n] = cp[i_j_k_n] + f[i2_j2_k2_n]
                    f[i2p1_j2p1_k2p1_n] = cp[i_j_k_n] + f[i2p1_j2p1_k2p1_n]
                    f[i2_j2p1_k2p1_n] = cp[i_j_k_n] + f[i2_j2p1_k2p1_n]
                    f[i2p1_j2_k2p1_n] = cp[i_j_k_n] + f[i2p1_j2_k2p1_n]
                    f[i2_j2_k2p1_n] = cp[i_j_k_n] + f[i2_j2_k2p1_n]

              }
            }
         }
      }

#else
      FORT_INTERP(
          f, DIMS(f),
          c, DIMS(c),
          lo, hi, nc)
      implicit none
      integer nc
      integer DIMDEC(f)
      integer DIMDEC(c)
      integer lo(BL_SPACEDIM)
      integer hi(BL_SPACEDIM)
      REAL_T f(DIMV(f),nc)
      REAL_T c(DIMV(c),nc)

      integer i, i2, i2p1, j, j2, j2p1, k, k2, k2p1, n
       
!     MultiGrid::relax(...) does only V-cycles (not F-cycles), and for V-cycles, 
!     piecewise-constant interpolation performs better than linear interpolation,
!     as measured both by run-time and number of V-cycles for convergence.

      do n = 1, nc
         do k = lo(3), hi(3)
            k2 = 2*k
            k2p1 = k2 + 1
	    do j = lo(2), hi(2)
               j2 = 2*j
               j2p1 = j2 + 1

               do i = lo(1), hi(1)
                  i2 = 2*i
                  i2p1 = i2 + 1

                  f(i2p1,j2p1,k2  ,n) = c(i,j,k,n) + f(i2p1,j2p1,k2  ,n)
                  f(i2  ,j2p1,k2  ,n) = c(i,j,k,n) + f(i2  ,j2p1,k2  ,n)
                  f(i2p1,j2  ,k2  ,n) = c(i,j,k,n) + f(i2p1,j2  ,k2  ,n)
                  f(i2  ,j2  ,k2  ,n) = c(i,j,k,n) + f(i2  ,j2  ,k2  ,n)
                  f(i2p1,j2p1,k2p1,n) = c(i,j,k,n) + f(i2p1,j2p1,k2p1,n)
                  f(i2  ,j2p1,k2p1,n) = c(i,j,k,n) + f(i2  ,j2p1,k2p1,n)
                  f(i2p1,j2  ,k2p1,n) = c(i,j,k,n) + f(i2p1,j2  ,k2p1,n)
                  f(i2  ,j2  ,k2p1,n) = c(i,j,k,n) + f(i2  ,j2  ,k2p1,n)

               end do
            end do
         end do
      end do

      end
#endif
}

