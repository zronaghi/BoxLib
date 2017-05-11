#ifndef BL_OMPARENA_H
#define BL_OMPARENA_H

#include <Arena.H>
#include <BL_CXX11.H>
#include <omp.h>

//
// A Concrete Class for Dynamic Memory Management
//
// It allocates and maps data to the target device
//
// Makes calls to ::operator new() and ::operator delete().
//

class OMPArena
    :
    public Arena
{
public:
    //
    // Allocates a dynamic memory arena of size sz.
    // Returns a pointer to this memory.
    //
  virtual void* alloc (std::size_t sz) BL_OVERRIDE;
    //
    // Deletes the arena pointed to by pt.
    //
  virtual void free (void* pt) BL_OVERRIDE;
};

#endif /*BL_OMPARENA_H*/