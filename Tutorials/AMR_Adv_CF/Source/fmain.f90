
subroutine fmain () bind(c)

  use boxlib_module
  use amr_module, only : amr_init, amr_finalize
  use adv_amrlevel_module, only : amrlevel_init
  use runtime_parm_init_module, only : runtime_parm_init

  implicit none

  call runtime_parm_init()

  call amrlevel_init()

  call amr_init()

!  call advance()

  call amr_finalize()

end subroutine fmain
