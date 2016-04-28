
module amrlevel_module

  use boxlib_module

  implicit none

  private

  public :: amrlevel

  type, abstract :: amrlevel
     integer :: level  = -1
   contains
     procedure :: amrlevel_ctor
     procedure(levelbuild), nopass, deferred :: amrlevel_build
     procedure(initdata)  ,         deferred :: amrlevel_init_data
  end type amrlevel

  abstract interface
     subroutine levelbuild(this_amrlevel, ba, level)
       import amrlevel
       import boxarray
       implicit none
       class(amrlevel), pointer :: this_amrlevel
       type(boxarray), intent(in) :: ba
       integer, intent(in) :: level
     end subroutine levelbuild

     subroutine initdata(this)
       import amrlevel
       implicit none
       class(amrlevel), intent(inout) :: this
     end subroutine initdata
  end interface

contains
  
  subroutine amrlevel_ctor (this, level)
    class(amrlevel), intent(inout) :: this
    integer, intent(in) :: level
    this%level = level
  end subroutine amrlevel_ctor

end module amrlevel_module
