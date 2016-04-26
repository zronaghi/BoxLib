
module geometry_module

  use iso_c_binding
  use bl_types_module
  use bl_space_module, only : bl_num_dims
  use box_module
  use parmparse_module

  implicit none

  private

  public :: geometry_build, geometry_parm_init, geometry_get_coord_sys, &
       geometry_get_periodic, geometry_get_prob_lo, geometry_get_prob_hi

  type, public :: Geometry
     real(double) :: dx(3)     = 0.0d0
     type(Box)    :: domain
     type(c_ptr)  :: p         = c_null_ptr
   contains
     final :: geometry_destroy
  end type Geometry

  logical, save :: parm_initialized = .false.
  integer, save :: coord_sys = 0
  logical, save :: is_periodic(3) = .false.
  real(double), save :: prob_lo(3) = 0.0d0
  real(double), save :: prob_hi(3) = 0.0d0

  ! interfaces to c++ functions

  interface
     subroutine fi_new_geometry (geom,lo,hi) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: geom
       integer, intent(in) :: lo(3), hi(3)
     end subroutine fi_new_geometry

     subroutine fi_delete_geometry (geom) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value, intent(in) :: geom
     end subroutine fi_delete_geometry

     subroutine fi_geometry_get_pmask (geom,pmask) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value, intent(in) :: geom
       integer(c_int) :: pmask(3)
     end subroutine fi_geometry_get_pmask

     subroutine fi_geometry_get_probdomain (geom,problo,probhi) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value, intent(in) :: geom
       real(c_double) :: problo(3), probhi(3)
     end subroutine fi_geometry_get_probdomain
  end interface

contains

  subroutine geometry_parm_init ()
    type(parmparse) :: pp
    integer :: isp(3)
    call parmparse_build(pp, "geometry")
    call pp%query("coord_sys", coord_sys)
    isp = 0
    call pp%queryarr("is_periodic", isp, bl_num_dims)
    where (isp .eq. 1) is_periodic = .true.
    call pp%getarr("prob_lo", prob_lo, bl_num_dims);
    call pp%getarr("prob_hi", prob_hi, bl_num_dims);
  end subroutine geometry_parm_init

  subroutine geometry_build (geom, domain)
    type(Geometry) :: geom
    type(Box), intent(in) :: domain
    integer :: i
    if (.not.parm_initialized) then
       call geometry_parm_init()
    end if
    call fi_new_geometry(geom%p, domain%lo, domain%hi)
    geom%domain = domain
    do i = 1, bl_num_dims
       geom%dx(i) = (prob_hi(i)-prob_lo(i)) / dble(domain%hi(i)-domain%lo(i)+1)
    end do
  end subroutine geometry_build

  impure elemental subroutine geometry_destroy (geom)
    type(Geometry), intent(inout) :: geom
    if (c_associated(geom%p)) then
       call fi_delete_geometry(geom%p)
    end if
    geom%p = c_null_ptr
  end subroutine geometry_destroy

  integer function geometry_get_coord_sys ()
    geometry_get_coord_sys = coord_sys
  end function geometry_get_coord_sys

  function geometry_get_periodic () result(r)
    logical :: r(bl_num_dims)
    r = is_periodic(1:bl_num_dims)
  end function geometry_get_periodic

  function geometry_get_prob_lo () result(r)
    real(double) :: r(bl_num_dims)
    r = prob_lo(1:bl_num_dims)
  end function geometry_get_prob_lo

  function geometry_get_prob_hi () result(r)
    real(double) :: r(bl_num_dims)
    r = prob_hi(1:bl_num_dims)
  end function geometry_get_prob_hi

end module geometry_module
