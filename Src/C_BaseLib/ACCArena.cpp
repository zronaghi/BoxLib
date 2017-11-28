#include <ACCArena.H>

void*
ACCArena::alloc (std::size_t _sz)
{
  void* pt=::operator new(_sz);
  char* ptr=reinterpret_cast<char*>(pt);
#pragma acc enter data create(ptr[0:_sz])
  return pt;
}

void
ACCArena::free (void* pt)
{
  char* ptr=reinterpret_cast<char*>(pt);
#pragma acc exit data delete(ptr[:0])
    ::operator delete(pt);
}
