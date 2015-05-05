#ifndef included_LevelSetGPU
#define included_LevelSetGPU

///////////////////////////////// INCLUDES /////////////////////////////////


#include <stdio.h>
#include <float.h>
#include <math.h> 




#ifdef CPPCODE
#define GLOBAL
#define DEVICE 
#else
#define GLOBAL __global__
#define DEVICE __device__
#endif


#define BLOCK_SIZE          32



#include "handleerror.h"
#include "LevelSet.h"




/////////////////////////////  GPU KERNEL FUNCTION PROTOTYPES ///////////////////////////////////


GLOBAL void testkernel1(float* phi, int Nx, int Ny, int pitch);

GLOBAL void SetPhi(float* phi, int Nx, int Ny, int pitch, int points, float * fx, float * fy, 
		       float Xmin, float Ymin, float dx, float dy);

// For reinialization via FastMarching method by brute force using bicubic interpolation and Newton solver
GLOBAL void getVoxels(float* psi, int pitch, int* voxelList, float* alphaList, int Nx, int Ny);
GLOBAL void reinitPhi(float* phi, int pitch, float* psi, int* voxelList, float* alphaList, int Nx, int Ny, float dx, float thres);


GLOBAL void FastMarchInit(float* phi, int pitch, int* Accept, int Apitch, float* Fext, int Nx, int Ny, float dx, float dy, float Xmin, float Ymin);
GLOBAL void FastMarchVelocity(int count, float* phi, int pitch, int* Accept, int Apitch, float* Fext, int* AcceptNew, float* FextNew, int Nx, int Ny, float dx, float dy);



///////////////////////////////// STRUCT DEFINITION /////////////////////////////////

class LevelSetCUDA
{
public:
    int Nx;		// No. pts in x-dirn
    int Ny;		// No. pts in y-dirn
    float dx;		// Grid spacing in x-dirn
    float dy;         // Grid spacing in x-dirn
    float Xmin;
    float Ymin;

 
   dim3 dimGrid;
   dim3 dimBlock;
 
 
     float* Psi; 	        // Initial lvl set fn
    float* Phi; 		// Least distance fn


    int * Accept;
    float * Fspeed;
    int * Atemp;
    float * Ftemp;
 
    float* xi;    	// for points on implicit curve
    float* yj;	// for points on implicit curve




	// METHODS
    void _setup(LevelSet* hostLS);
    void _transferPhiToHost(LevelSet* hostLS);
    void _launchSimpleSignKernel( );
    void _getSignedDistanceFunction();
    void _reinitFastMarchNewton(LevelSet* hostLS);
    void _extendVelocityF(LevelSet* hostLS);
    void _teardown(LevelSet* hostLS);


};
	

//////////////////////////////////////////////////////////////////


#endif //#ifndef included_LevelSetGPU
