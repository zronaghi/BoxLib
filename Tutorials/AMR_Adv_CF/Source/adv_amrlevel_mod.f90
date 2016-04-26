
module adv_amrlevel_module

  use boxlib_module
  use amr_module
  use amrlevel_module

  implicit none
  
  private

  public :: adv_amrlevel, amrlevel_build, amrlevel_init

  type, extends(amrlevel) :: adv_amrlevel
     type(multifab) :: phi_old
     type(multifab) :: phi_new
     ! There appears to be a gfortran bug that causes the destruction of the 
     ! explicit shape array crash.  So we use allocatable instead.
     ! type(multifab) :: velocity(bl_num_dims)  
     type(multifab), allocatable :: velocity(:)
   contains
     procedure, nopass :: amrlevel_build
     final :: adv_amrlevel_destroy
  end type adv_amrlevel

contains

  impure elemental subroutine adv_amrlevel_destroy (this)
    type(adv_amrlevel), intent(inout) :: this
  end subroutine adv_amrlevel_destroy

  ! We need to let amr_moudle know out extended amrlevel type
  subroutine amrlevel_init ()
    type(adv_amrlevel) :: a_level
    call amr_level_type_init(a_level)
  end subroutine amrlevel_init

  subroutine amrlevel_build (this_amrlevel, ba, level)
    class(amrlevel), pointer :: this_amrlevel
    type(boxarray), intent(in) :: ba
    integer, intent(in) :: level

    ! local variables
    integer :: ncomp, ngrow, i
    logical :: nodal(bl_num_dims)
    class(adv_amrlevel), pointer :: a_level

    ! <MUST HAVE> the following block
    allocate(a_level)
    this_amrlevel => a_level
    call a_level%amrlevel_ctor(level)
    ! </MUST HAVE>

    ncomp = 1
    ngrow = 3
    call multifab_build(a_level%phi_old, ba, ncomp, ngrow)
    call multifab_build(a_level%phi_new, ba, ncomp, ngrow)

    allocate(a_level%velocity(bl_num_dims))
    do i = 1, bl_num_dims
       nodal = .false.
       nodal(i) = .true.
       call multifab_build(a_level%velocity(i), ba, 1, 1, nodal)
    end do

  end subroutine amrlevel_build

end module adv_amrlevel_module
