/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 */



////////////////////////////////////////////////////////////////////////////////
// C++ library includes
////////////////////////////////////////////////////////////////////////////////
#include <cstdlib>
#include <time.h>
//#define CPPCODE


////////////////////////////////////////////////////////////////////////////////
// Host library includes
////////////////////////////////////////////////////////////////////////////////
#include "LevelSet.h"

////////////////////////////////////////////////////////////////////////////////
// Device library includes
////////////////////////////////////////////////////////////////////////////////
#include "LevelSetGPU.cuh"


////////////////////////////////////////////////////////////////////////////////
// MAIN PROGRAM
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
 
     if(argc != 4)
    {
    printf("ERROR: Usage: ./main <nX> <nY> <curveType>\n");
    exit(0);
    }   

    // parameters set on host
    int h_Nx = atoi(argv[1]);
    int h_Ny = atoi(argv[2]);

    // create structure on host
    int curveType = atoi(argv[3]);
    LevelSet hostLS(h_Nx, h_Ny, curveType); // third argument is 1 for default circle. 

    // create structure on device and move data from host to device memory
    LevelSetCUDA devLS;
    devLS._setup(&hostLS);

    // set up implicit curve on device memory
    if (curveType==3) {
	devLS._getSignedDistanceFunction();
	devLS._transferPhiToHost(&hostLS); //only for testing output to debug
     }

    // Check initial values
    //hostLS._TestOutput(hostLS.PhiGPU, hostLS.Nx, hostLS.Ny);


    clock_t t1,t2;
    float diff, secondsG, secondsC;
    
    t1=clock();
    devLS._reinitFastMarchNewton(&hostLS); // Launch reinitialization of LS on GPU by FastMarching Newton solve method
    t2=clock();
    diff = ((float)t2-(float)t1);
    secondsG = diff / CLOCKS_PER_SEC;
    //cout<<"\nGPU reinitialization by FastMarching Newton solver method took "<< secondsG << " seconds.\n"<<endl;
    devLS._transferPhiToHost(&hostLS);
    if(h_Nx<30)
    {
        cout<<"\nPress RETURN to continue\n "<<endl;
        cin.get();
	hostLS._TestOutput(hostLS.PhiGPU, hostLS.Nx, hostLS.Ny);     // Check results on host from GPU
    }
	

    t1=clock();
    hostLS._Reinitialize(); // Launch reinitialization of LS on CPU by FastMarching narrowband heap method
    t2=clock();
    diff = ((float)t2-(float)t1);
    secondsC = diff / CLOCKS_PER_SEC;
    //cout<<"\nGPU reinitialization by FastMarching Newton solver method took "<< secondsC << " seconds.\n"<<endl;
    if(h_Nx<30)
    {
	cout<<"\nPress RETURN to continue\n "<<endl;
	cin.get();
	hostLS._TestOutput(hostLS.PhiCPU, hostLS.Nx, hostLS.Ny);   // Check results on host from CPU  
    }    
  

	float errorDiff = hostLS._getDifference();
	cout<<"\nAvgerage difference between CPU and GPU Phi values: "<< errorDiff << " \n"<<endl;
	cout<<"\nCPU computation time: "<< secondsC << " \n"<<endl;
	cout<<"\nGPU computation time: "<< secondsG << " \n"<<endl;

    devLS._extendVelocityF(&hostLS); 


    // Pass data back to host and delete device memory
    devLS._teardown(&hostLS);

    
    hostLS._TestOutputInt(hostLS.Accept, hostLS.Nx, hostLS.Ny);

    


    return 0;
}

