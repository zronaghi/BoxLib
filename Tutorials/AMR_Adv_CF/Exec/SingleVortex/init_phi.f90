
module init_phi_module

  use boxlib_module
  
  implicit none

  private

  public :: init_phi_on_level

contains

  subroutine init_phi_on_level(phi,geom)

    type(multifab), intent(inout) :: phi
    type(geometry), intent(in   ) :: geom

    ! local
    real(double) :: dx, prob_lo(bl_num_dims)
    integer :: dlo(4), dhi(4)
    type(box) :: bx
    type(mfiter) :: mfi
    real(double), pointer :: p(:,:,:,:)

    dx = geom%dx(1)
    prob_lo = geometry_get_prob_lo()

    !$omp parallel private(dlo,dhi,bx,mfi,p)
    call mfiter_build(mfi, phi, tiling=.true.)
    do while(mfi%next())
       bx  = mfi%tilebox()
       p   => phi%dataptr(mfi)
       dlo = lbound(p)
       dhi = ubound(p)
       select case (bl_num_dims)
       case (2)
          call init_phi_2d(bx%lo, bx%hi, p, dlo, dhi, dx, prob_lo)
       case (3)
          call init_phi_3d(bx%lo, bx%hi, p, dlo, dhi, dx, prob_lo)
       end select
    end do
    !$omp end parallel

  end subroutine init_phi_on_level
  
  subroutine init_phi_2d(lo, hi, phi, dlo, dhi, dx, prob_lo)
    integer     , intent(in   ) :: lo(2), hi(2), dlo(2), dhi(2)
    real(double), intent(inout) :: phi(dlo(1):dhi(1),dlo(2):dhi(2))
    real(double), intent(in   ) :: dx, prob_lo(2)
 
    ! local varables
    integer      :: i,j
    real(double) :: x,y,r1

    do j=lo(2),hi(2)
       y = prob_lo(2) + (dble(j)+0.5d0) * dx
       do i=lo(1),hi(1)
          x = prob_lo(1) + (dble(i)+0.5d0) * dx
          r1 = ((x-0.5d0)**2 + (y-0.75d0)**2) / 0.01d0
          phi(i,j) = 1.d0 + exp(-r1)
       end do
    end do    
  end subroutine init_phi_2d

  subroutine init_phi_3d(lo, hi, phi, dlo, dhi, dx, prob_lo)
    integer     , intent(in   ) :: lo(3), hi(3), dlo(3), dhi(3)
    real(double), intent(inout) :: phi(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3))
    real(double), intent(in   ) :: dx, prob_lo(3)
 
    ! local varables
    integer      :: i,j,k
    real(double) :: x,y,z,r1

    do k=lo(3),hi(3)
       z = prob_lo(3) + (dble(k)+0.5d0) * dx
       do j=lo(2),hi(2)
          y = prob_lo(2) + (dble(j)+0.5d0) * dx
          do i=lo(1),hi(1)
             x = prob_lo(1) + (dble(i)+0.5d0) * dx
             r1 = ((x-0.5d0)**2 + (y-0.75d0)**2 + (z-0.5d0)**2) / 0.01d0
             phi(i,j,k) = 1.d0 + exp(-r1)
          end do
       end do
    end do
  end subroutine init_phi_3d

end module init_phi_module

