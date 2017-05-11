#include <OMPArena.H>

void*
OMPArena::alloc (std::size_t _sz)
{
  void* pt=::operator new(_sz);
  char* ptr=reinterpret_cast<char*>(pt);
#pragma omp target enter data map(alloc:ptr[0:_sz])
  return pt;
}

void
OMPArena::free (void* pt)
{
  char* ptr=reinterpret_cast<char*>(pt);
#pragma omp target exit data map(release:ptr[:0])
    ::operator delete(pt);
}
