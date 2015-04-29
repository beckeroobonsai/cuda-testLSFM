#ifndef included_LevelSet
#define included_LevelSet

///////////////////////////////// INCLUDES /////////////////////////////////

//C++ STDLIB INCLUDES
//#include <cmath>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
//#include <cstdio>
#include <algorithm>
#include <queue>
#include "fastmarching.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifndef PI
#define PI           3.1415926535897932384626433832795028841971693
#endif

#define EPSILON 1e-8
#define LARGE 1.0e10

//using namespace std;


///////////////////////////////// LevelSet DEFINITION /////////////////////////////////


typedef struct LevelSet
//class Levelset 
{
//public:	

    int c2g(float x);
    float g2c(int j);

    int Nx;			// No. pts in x-dirn
    int Ny;			// No. pts in y-dirn
    float dx;			// Grid spacing in x-dirn
    float dy;			// Grid spacing in y-dirn
    float Xmin;
    float Ymin;
    

    float* PhiGPU; 		
    float* PhiCPU;
    float* Dist;
    int *status, *signs;    
    
    float* Fext;
    int* Accept;
    float* xi;		// for points on implicit curve
    float* yj;		// for points on implicit curve
    int points;


//public:
    LevelSet(int Nx, int Ny, int example);//Constructor
    ~LevelSet();  //Destructor	

    int _Reinitialize();    // Reinitialize on CPU by FastMarching 
    
    void _TestOutput(float * A, int nx, int ny); // Output results
    void _TestOutputInt(int * A, int nx, int ny);     // Output shortened to integer
    void _SavePhiToFile();     // Save Phi matrix to file
    float _getDifference();    //calculates the 1 norm of PhiGPU-PhiCPU

		


} Levelset ;

//////////////////////////////////////////////////////////////////

#endif //#ifndef included_LevelSet
