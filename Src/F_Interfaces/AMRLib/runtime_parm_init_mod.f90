
module runtime_parm_init_module

  implicit none

  private

  public :: runtime_parm_init

contains

  subroutine runtime_parm_init
    use amr_module, only : amr_parm_init
    use geometry_module, only : geometry_parm_init
    call amr_parm_init()
    call geometry_parm_init()
  end subroutine runtime_parm_init

end module runtime_parm_init_module

