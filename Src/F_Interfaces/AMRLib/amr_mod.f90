
module amr_module

  use amrlevel_module
  use boxlib_module

  implicit none

  private

  public :: amr_parm_init, amr_init, amr_finalize, amr_level_type_init

  type amrlevel_ptr
     class(amrlevel), pointer :: p =>null()
  end type amrlevel_ptr

  type(geometry), allocatable :: geoms(:)
  type(amrlevel_ptr), allocatable :: amrlevels(:)

  ! These will become runtime parameters
  logical, save :: parm_initialized = .false.
  integer, save :: max_level = -1
  integer, save :: max_grid_size = 64  ! this could be level dependent
  integer, save :: blocking_factor = 8
  integer, save :: regrid_int = 2
  integer, save :: n_cell(3) = 1
  integer, allocatable, save :: ref_ratio(:)

  class(amrlevel), pointer :: amrlevel_builder => null()

contains

  subroutine amr_parm_init ()
    type(parmparse) :: pp
    call parmparse_build(pp, "amr")
    call pp%get("max_level", max_level)
    call pp%query("max_grid_size", max_grid_size)
    call pp%query("blocking_factor", blocking_factor)
    call pp%query("regrid_int", regrid_int)
    call pp%getarr("n_cell", n_cell,bl_num_dims)
    allocate(ref_ratio(0:max_level-1))
    ref_ratio = 2
    call pp%queryarr("ref_ratio",ref_ratio,max_level)
    parm_initialized = .true.
  end subroutine amr_parm_init

  subroutine amr_level_type_init (level)
    class(amrlevel), intent(in) :: level
    allocate(amrlevel_builder, source=level)
  end subroutine amr_level_type_init

  subroutine amr_init ()
    
    integer :: i
    type(Box) :: domain
    type(BoxArray) :: ba0

    if (.not.associated(amrlevel_builder)) then
       call bl_error("amr_level_type_init has not been called to let amr module get hold of a concrete type")
    end if

    if (.not.parm_initialized) then
       call bl_error("amr_parm_init has not been called to initialize amr runtime parameters")
    end if

    allocate(geoms(0:max_level))

    do i = 0, max_level
       ! domain.refine()
       call geometry_build(geoms(i), domain)
    end do

    allocate(amrlevels(0:max_level))

    ! Level 0 boxarray
    domain = Box((/0,0,0/), n_cell-1)
    call boxarray_build(ba0, domain)
    call ba0%maxSize(max_grid_size)
    
    ! build level 0 data
    call amrlevel_builder%amrlevel_build(amrlevels(0)%p, ba0, level=0)

  end subroutine amr_init

  subroutine amr_finalize ()
    print *, "TODO: need to deallocate resources in amr_finalize()"
  end subroutine amr_finalize

end module amr_module

