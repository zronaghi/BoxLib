#ifndef BL_ACCARENA_H
#define BL_ACCARENA_H

#include <Arena.H>

//
// A Concrete Class for Dynamic Memory Management
//
// It allocates and maps data to the target device
//
// Makes calls to ::operator new() and ::operator delete().
//

class ACCArena
    :
    public Arena
{
public:
    //
    // Allocates a dynamic memory arena of size sz.
    // Returns a pointer to this memory.
    //
  virtual void* alloc (std::size_t sz);
    //
    // Deletes the arena pointed to by pt.
    //
  virtual void free (void* pt);
};

#endif /*BL_ACCARENA_H*/
