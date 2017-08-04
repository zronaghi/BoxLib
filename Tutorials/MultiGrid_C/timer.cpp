#include "timer.h"

using namespace std;

timer::timer(const std::string& name_) : elapsed(0.), name(name_) {}
    
void timer::reset(){ elapsed=0.; }

void timer::start(){ 
    tstart = omp_get_wtime(); 
}

void timer::stop(){ 
    tend = omp_get_wtime(); 
    elapsed += tend-tstart;
}
    
void timer::print(){
    double s = elapsed;
    int h = static_cast<int>(s/3600.0);
    s = s - h*3600;
    int m = static_cast<int>(s/60.0);
    s = s - m*60;
    std::cout << "   Execution time " << name << " ";
    if( h > 1 )
        std::cout << h << " hours ";
    else if( h > 0 )
        cout << h << " hour  ";
    
    if( m > 1 )
        std::cout << m << " minutes ";
    else if( m > 0 )
        cout << m << " minute  ";
    
    if( s > 0 )
        std::cout << s << " seconds " ;
    std::cout << endl;
}