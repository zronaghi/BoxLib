
module box_module

  use bl_space_module, only : bl_num_dims

  implicit none

  private

  type, public :: Box
     logical, dimension(3) :: nodal = .false.
     integer, dimension(3) :: lo    = 1
     integer, dimension(3) :: hi    = 1
   contains
     procedure :: refine
  end type Box

  interface Box
     module procedure build_box
  end interface Box

contains

  function build_box(lo, hi, nodal) result(bx)
    integer, intent(in) :: lo(:), hi(:)
    logical, intent(in), optional :: nodal(:)
    type(Box) :: bx
    bx%lo(1:bl_num_dims) = lo(1:bl_num_dims)
    bx%hi(1:bl_num_dims) = hi(1:bl_num_dims)
    if (present(nodal)) then
       bx%nodal(1:bl_num_dims) = nodal(1:bl_num_dims)
    end if
  end function build_box

  subroutine refine(this, rr)
    class(Box), intent(inout) :: this
    integer, intent(in) :: rr
    integer :: i
    do i = 1, bl_num_dims
       this%lo(i) = this%lo(i) * rr
       if (this%nodal(i)) then
          this%hi(i) = this%hi(i) * rr
       else
          this%hi(i) = (this%hi(i)+1) * rr - 1 
       end if
    end do
  end subroutine refine

end module box_module

