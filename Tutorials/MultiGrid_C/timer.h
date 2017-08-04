#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <string>
#include <omp.h>

class timer {
private:
    double tstart, tend, elapsed;
    std::string name;
public:
    timer(const std::string& name_);
    
    void reset();
    void start();
    void stop();
    void print();
};

#endif