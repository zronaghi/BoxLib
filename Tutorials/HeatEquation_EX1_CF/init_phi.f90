
module init_phi_module

  use boxlib_module

  implicit none

  private

  public :: init_phi

contains
  
  subroutine init_phi(phi,geom)
    
    type(multifab), intent(inout) :: phi
    type(geometry), intent(in   ) :: geom

    ! local variables
    integer :: lo(4), hi(4)
    type(Box) :: bx
    type(mfiter) :: mfi
    real(double), pointer :: p(:,:,:,:)
    real(double) :: problo(bl_num_dims)

    problo = geometry_get_prob_lo()

    !$omp parallel private(bx,p,lo,hi,mfi)
    call mfiter_build(mfi, phi, tiling=.true.)
    do while(mfi%next())
       bx = mfi%tilebox()
       p => phi%dataPtr(mfi)
       lo = lbound(p)
       hi = ubound(p)
       select case (bl_num_dims)
       case (2)
          call init_phi_2d(bx%lo(1:2), bx%hi(1:2), p(:,:,1,1), lo(1:2), hi(1:2), &
               problo, geom%dx(1:2))
       case (3)
          call init_phi_3d(bx%lo, bx%hi, p(:,:,:,1), lo(1:3), hi(1:3), &
               problo, geom%dx)
       end select
    end do
    !$omp end parallel

  end subroutine init_phi

  subroutine init_phi_2d(lo, hi, phi, dlo, dhi, prob_lo, dx)
    integer          :: lo(2), hi(2), dlo(2), dhi(2)
    real(double) :: phi(dlo(1):dhi(1),dlo(2):dhi(2))
    real(double) :: prob_lo(2)
    real(double) :: dx(2)
 
    ! local varables
    integer          :: i,j
    real(double) :: x,y,r2

    do j=lo(2),hi(2)
       y = prob_lo(2) + (dble(j)+0.5d0) * dx(2)
       do i=lo(1),hi(1)
          x = prob_lo(1) + (dble(i)+0.5d0) * dx(1)
          
          r2 = ((x-0.25d0)**2 + (y-0.25d0)**2) / 0.01d0
          phi(i,j) = 1.d0 + exp(-r2)
          
       end do
    end do
    
  end subroutine init_phi_2d

  subroutine init_phi_3d(lo, hi, phi, dlo, dhi, prob_lo, dx)
    integer          :: lo(3), hi(3), dlo(3), dhi(3)
    real(double) :: phi(dlo(1):dhi(1),dlo(2):dhi(2),dlo(3):dhi(3))
    real(double) :: prob_lo(3)
    real(double) :: dx(3)

    ! local varables
    integer          :: i,j,k
    real(double) :: x,y,z,r2

    do k=lo(3),hi(3)
       z = prob_lo(3) + (dble(k)+0.5d0) * dx(3)
       do j=lo(2),hi(2)
          y = prob_lo(2) + (dble(j)+0.5d0) * dx(2)
          do i=lo(1),hi(1)
             x = prob_lo(1) + (dble(i)+0.5d0) * dx(1)

             r2 = ((x-0.25d0)**2 + (y-0.25d0)**2 + (z-0.25d0)**2) / 0.01d0
             phi(i,j,k) = 1.d0 + exp(-r2)

          end do
       end do
    end do

  end subroutine init_phi_3d
  
end module init_phi_module
