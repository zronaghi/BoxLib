module basefab_nd_module

  use bl_fort_module, only : c_real

  implicit none

contains

  ! dst = src
  subroutine fort_fab_copy(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_copy')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo
    
    !$omp target update to(src)
    
    !$omp target map(to: src) map(from: dst) map(to: hi, lo, off)
    !$omp teams distribute parallel do collapse(4)
    !&omp private(n,k,j,i)
    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
    !$omp end teams distribute parallel do
    !$omp end target
    
    !$omp target update from(dst)
  end subroutine fort_fab_copy
    
  
  ! copy from multi-d array to 1d array
  subroutine fort_fab_copytomem (lo, hi, dst, src, slo, shi, ncomp) &
       bind(c,name='fort_fab_copytomem')
    integer, intent(in) :: lo(3), hi(3), slo(3), shi(3), ncomp
    real(c_real)             :: dst( (hi(1)-lo(1)+1) * (hi(2)-lo(2)+1) * (hi(3)-lo(3)+1) * ncomp)
    real(c_real), intent(in) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)

    integer :: i, j, k, n, nx, ny, nz, offset

    nx = hi(1)-lo(1)+1
    ny = hi(2)-lo(2)+1
    nz = hi(3)-lo(3)+1
    
    !$omp target update to(src)

    !$omp target map(to: src) map(from: dst) map(to: hi, lo, nx, ny, nz)
    !$omp teams distribute parallel do collapse(3)
    !&omp private(n,k,j,i, offset) firstprivate(nx, ny)
    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             offset = 1-lo(1) + nx * ( (j-lo(2)) + ny * ( (k-lo(3)) + nz * (ncomp-1) ) )
             !$omp do simd private(i)
             do i = lo(1), hi(1)
                dst(offset+i) = src(i,j,k,n) 
             end do
             !$omp end do simd
             !offset = offset + nx
          end do
       end do
    end do
    !$omp end teams distribute parallel do
    !$omp end target
    
    !$omp target update from(dst)
  end subroutine fort_fab_copytomem


  ! copy from 1d array to multi-d array
  subroutine fort_fab_copyfrommem (lo, hi, dst, dlo, dhi, ncomp, src) &
       bind(c,name='fort_fab_copyfrommem')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), ncomp
    real(c_real), intent(in   ) :: src(*)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)

    integer :: i, j, k, n, nx, offset

    nx = hi(1)-lo(1)+1
    offset = 1-lo(1)
    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n)  = src(offset+i)
             end do
             offset = offset + nx
          end do
       end do
    end do    
  end subroutine fort_fab_copyfrommem


  subroutine fort_fab_setval(lo, hi, dst, dlo, dhi, ncomp, val) &
       bind(c,name='fort_fab_setval')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), ncomp
    real(c_real), intent(in) :: val
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)

    integer :: i, j, k, n

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = val
             end do
          end do
       end do
    end do
  end subroutine fort_fab_setval


  function fort_fab_norm (lo, hi, src, slo, shi, ncomp, p) result(nrm) &
       bind(c,name='fort_fab_norm')
    integer, intent(in) :: lo(3), hi(3), slo(3), shi(3), ncomp, p
    real(c_real), intent(in) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real) :: nrm

    integer :: i,j,k,n

    nrm = 0.0_c_real
    if (p .eq. 0) then ! max norm
       do n = 1, ncomp
          do       k = lo(3), hi(3)
             do    j = lo(2), hi(2)
                do i = lo(1), hi(1)
                   nrm = max(nrm, abs(src(i,j,k,n)))
                end do
             end do
          end do
       end do
    else if (p .eq. 1) then
       do n = 1, ncomp
          do       k = lo(3), hi(3)
             do    j = lo(2), hi(2)
                do i = lo(1), hi(1)
                   nrm = nrm + abs(src(i,j,k,n))
                end do
             end do
          end do
       end do
    end if
  end function fort_fab_norm


  function fort_fab_sum (lo, hi, src, slo, shi, ncomp) result(sm) &
       bind(c,name='fort_fab_sum')
    integer, intent(in) :: lo(3), hi(3), slo(3), shi(3), ncomp
    real(c_real), intent(in) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real) :: sm

    integer :: i,j,k,n

    sm = 0.0_c_real
    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                sm = sm + src(i,j,k,n)
             end do
          end do
       end do
    end do
  end function fort_fab_sum


  subroutine fort_fab_plus(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_plus')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = dst(i,j,k,n) + src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_plus


  subroutine fort_fab_minus(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_minus')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = dst(i,j,k,n) - src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_minus


  subroutine fort_fab_mult(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_mult')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = dst(i,j,k,n) * src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_mult


  subroutine fort_fab_divide(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_divide')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = dst(i,j,k,n) / src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_divide


  subroutine fort_fab_protdivide(lo, hi, dst, dlo, dhi, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_protdivide')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                if (src(i+off(1),j+off(2),k+off(3),n) .ne. 0._c_real) then
                   dst(i,j,k,n) = dst(i,j,k,n) / src(i+off(1),j+off(2),k+off(3),n)
                end if
             end do
          end do
       end do
    end do
  end subroutine fort_fab_protdivide


  ! dst = a/src
  subroutine fort_fab_invert(lo, hi, dst, dlo, dhi, ncomp, a) &
       bind(c,name='fort_fab_invert')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), ncomp
    real(c_real), intent(in   ) :: a
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = a / dst(i,j,k,n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_invert


  ! dst += a*src
  subroutine fort_fab_saxpy(lo, hi, dst, dlo, dhi, a, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_saxpy')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: a
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = dst(i,j,k,n) + a * src(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_saxpy


  ! dst = src + a*dst
  subroutine fort_fab_xpay(lo, hi, dst, dlo, dhi, a, src, slo, shi, sblo, ncomp) &
       bind(c,name='fort_fab_xpay')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), slo(3), shi(3), sblo(3), ncomp
    real(c_real), intent(in   ) :: a
    real(c_real), intent(in   ) :: src(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),ncomp)
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    
    integer :: i,j,k,n,off(3)

    off = sblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = src(i+off(1),j+off(2),k+off(3),n) + a * dst(i,j,k,n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_xpay


  ! dst = a*x + b*y
  subroutine fort_fab_lincomb(lo, hi, dst, dlo, dhi, a, x, xlo, xhi, xblo, &
       b, y, ylo, yhi, yblo, ncomp) bind(c,name='fort_fab_lincomb')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), xlo(3), xhi(3), xblo(3), &
         ylo(3), yhi(3), yblo(3), ncomp
    real(c_real), intent(in   ) :: a, b
    real(c_real), intent(inout) :: dst(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3),ncomp)
    real(c_real), intent(in   ) ::   x(xlo(1):xhi(1),xlo(2):xhi(2),xlo(3):xhi(3),ncomp)
    real(c_real), intent(in   ) ::   y(ylo(1):yhi(1),ylo(2):yhi(2),ylo(3):yhi(3),ncomp)
    
    integer :: i,j,k,n,xoff(3),yoff(3)

    xoff = xblo - lo
    yoff = yblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = a * x(i+xoff(1),j+xoff(2),k+xoff(3),n) &
                     +         b * y(i+yoff(1),j+yoff(2),k+yoff(3),n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_lincomb

  ! dst = dst + src1*src2
  subroutine fort_fab_addproduct(lo, hi, dst, dlo, dhi, src1, s1lo, s1hi, src2, s2lo, s2hi,ncomp) &
       bind(c,name='fort_fab_addproduct')
    integer, intent(in) :: lo(3), hi(3), dlo(3), dhi(3), s1lo(3), s1hi(3), s2lo(3), s2hi(3), ncomp
    real(c_real), intent(in   ) :: src1(s1lo(1):s1hi(1),s1lo(2):s1hi(2),s1lo(3):s1hi(3),ncomp)
    real(c_real), intent(in   ) :: src2(s2lo(1):s2hi(1),s2lo(2):s2hi(2),s2lo(3):s2hi(3),ncomp)
    real(c_real), intent(inout) ::  dst( dlo(1): dhi(1), dlo(2): dhi(2), dlo(3): dhi(3),ncomp)
    
    integer :: i,j,k,n

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dst(i,j,k,n) = src1(i,j,k,n) * src2(i,j,k,n) + dst(i,j,k,n)
             end do
          end do
       end do
    end do
  end subroutine fort_fab_addproduct
  
  ! dot_product
  function fort_fab_dot(lo, hi, x, xlo, xhi, y, ylo, yhi, yblo, ncomp) result(dp) &
       bind(c,name='fort_fab_dot')
    integer, intent(in) :: lo(3), hi(3), xlo(3), xhi(3), ylo(3), yhi(3), yblo(3), ncomp
    real(c_real), intent(in) :: x(xlo(1):xhi(1),xlo(2):xhi(2),xlo(3):xhi(3),ncomp)
    real(c_real), intent(in) :: y(ylo(1):yhi(1),ylo(2):yhi(2),ylo(3):yhi(3),ncomp)
    real(c_real) :: dp

    integer :: i,j,k,n, off(3)

    dp = 0.0_c_real

    off = yblo - lo

    do n = 1, ncomp
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
                dp = dp + x(i,j,k,n)*y(i+off(1),j+off(2),k+off(3),n)
             end do
          end do
       end do
    end do
  end function fort_fab_dot

end module basefab_nd_module
