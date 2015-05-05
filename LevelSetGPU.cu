


///////////////////////////////// INCLUDES /////////////////////////////////
#include "LevelSetGPU.cuh"




/////////////////  FUNCTION IMPLEMENTATIONS : Wrappers around GPU Kernel functions //////////


void LevelSetCUDA::_setup(LevelSet* hostLS)
{

	 // copy parameters from one structure to another
	 // Ideally, these parameters would be in constant memory on GPU rather than on host.
	 //  For now, still just a copy from host to host 
	 Nx = hostLS->Nx;
	 Ny = hostLS->Ny;
	 dx = hostLS->dx;
	 dy = hostLS->dy;
	Xmin = hostLS->Xmin;
	Ymin = hostLS->Ymin;


    // Determine block and thread amounts for 2D kernel function
    int grid_dimX = ((Nx + BLOCK_SIZE-1) / BLOCK_SIZE) ;
    int grid_dimY = ((Ny + BLOCK_SIZE-1) / BLOCK_SIZE) ;
    dim3 grid(grid_dimX , grid_dimY);
    dim3 block(BLOCK_SIZE , BLOCK_SIZE);
	dimGrid.x=grid.x;
	dimGrid.y=grid.y;
	dimGrid.z=grid.z;
	dimBlock.x=block.x;
	dimBlock.y=block.y;
	dimBlock.z=block.z;



	HANDLE_ERROR( cudaMalloc((void**)&Phi,  Ny*Nx*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc((void**)&Psi,  Ny*Nx*sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy(Phi, hostLS->PhiGPU,  Ny*Nx*sizeof(float), cudaMemcpyHostToDevice) );

//    	size_t PhiPitchInbytes;
//    	size_t PsiPitchInbytes;

//	// Allocate memory on device 
//	HANDLE_ERROR( cudaMallocPitch((void**)&Phi, &PhiPitchInbytes, Ny, Nx) );
//	HANDLE_ERROR( cudaMallocPitch((void**)&Psi, &PsiPitchInbytes, Ny , Nx));
//	PhiPitch = (int)(PhiPitchInbytes / sizeof(float));
//	PsiPitch = (int)(PsiPitchInbytes / sizeof(float));
//	
//    	// Copy data from host to device
//	HANDLE_ERROR( cudaMemcpy2D(Phi, PhiPitchInbytes, hostLS->PhiGPU, 
//				Ny*sizeof(float), Ny*sizeof(float), Nx,
//					cudaMemcpyHostToDevice) );



    // Allocate memory on device for vectors of points on implicit curve
	HANDLE_ERROR( cudaMalloc((void**)&xi, (Nx*4+1)*sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&yj, (Nx*4+1)*sizeof(float)) );
	// Copy data from host to device
	HANDLE_ERROR( cudaMemcpy(xi, hostLS->xi, (Nx*4+1)*sizeof(float), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(yj, hostLS->yj, (Nx*4+1)*sizeof(float), cudaMemcpyHostToDevice) );




	printf("\nSetup complete. dx= %3.3f\n", dx );

 
}//_setup



void LevelSetCUDA::_transferPhiToHost(LevelSet* hostLS)
{
	//size_t PhiPitchInbytes = (size_t)(PhiPitch * sizeof(float));

    // Copy data back to host
	//HANDLE_ERROR( cudaMemcpy2D(hostLS->PhiGPU, Ny*sizeof(float), Phi, PhiPitchInbytes, 
	//		Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) );
			
	HANDLE_ERROR( cudaMemcpy(hostLS->PhiGPU, Phi, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost) );		

}//_transferPhiToHost



void LevelSetCUDA::_teardown(LevelSet* hostLS)
{
	//size_t PhiPitchInbytes = (size_t)(PhiPitch * sizeof(float));

    // Copy data back to host
    //HANDLE_ERROR( cudaMemcpy2D(hostLS->PhiGPU, Ny*sizeof(float), Phi, PhiPitchInbytes, 
		//	Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) );
			
   HANDLE_ERROR( cudaMemcpy(hostLS->PhiGPU, Phi, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );						

    // Deallocate memory on device 
    HANDLE_ERROR(cudaFree(Phi));
    HANDLE_ERROR(cudaFree(Psi));



	printf("\nTeardown complete. \n " );

}//_tearDown




void LevelSetCUDA::_launchSimpleSignKernel( )
{
	
	printf("\nLaunching sign function kernel ...\n");

	int PhiPitch = Nx;

	testkernel1<<<dimGrid, dimBlock>>>(Phi, Nx, Ny, PhiPitch);//marks values -1 if <0 and 1 if >0
	HANDLE_ERROR( cudaDeviceSynchronize() );


}//_launchSimpleSignKernel



void LevelSetCUDA::_getSignedDistanceFunction( )
{
	
	printf("Initializing levelset data from implicit curve\n");

	int PhiPitch = Nx;

	SetPhi<<<dimGrid, dimBlock>>>(Phi, Nx, Ny, PhiPitch, Nx*4, xi, yj, Xmin, Ymin, dx, dy);
	HANDLE_ERROR( cudaDeviceSynchronize() );


}//_getSignedDistanceFunction





void LevelSetCUDA::_reinitFastMarchNewton(LevelSet* hostLS )
{
	
	printf("Reinitizing levelset using Fast Marching with Newton solver\n");


	//dim3 dimBlock2(8,16);
	//dim3 dimGrid2((Nx-1)/8 + 1,(Ny-1)/16 + 1);

    //int* ListVoxels = new int[Nx*Ny];     // use for debugging
    //float* ListCoeff = new float[16*Nx*Ny]; //only need these on gpu
    int* devListVoxels;
    float* devListCoeff;
    int numberOfVox = Nx*Ny;//4096;
    HANDLE_ERROR( cudaMalloc((void**)&devListVoxels, numberOfVox*sizeof(int)) );
    HANDLE_ERROR( cudaMemset(devListVoxels,0,sizeof(int)));    
    HANDLE_ERROR( cudaMalloc((void**)&devListCoeff, 16*numberOfVox*sizeof(float) ) );

    //copy shape from host_phi temporarily to dev_psi before reinitializing on GPU
    //size_t PsiPitchInbytes = (size_t)(PsiPitch * sizeof(float));
    //HANDLE_ERROR( cudaMemcpy2D(Psi, PsiPitchInbytes, hostLS->PhiGPU, Ny*sizeof(float),
			//	 Ny*sizeof(float), Nx, cudaMemcpyHostToDevice) ); 
			

   HANDLE_ERROR( cudaMemcpy(Psi, hostLS->PhiGPU, Ny*Nx*sizeof(float), cudaMemcpyHostToDevice)  );

    printf("Starting to get voxels and coefficients\n");
    
    int PsiPitch = Nx;
    int PhiPitch = Nx;

    // get interpolating polynomial coefficients and the active set voxels
    getVoxels<<<dimGrid, dimBlock>>>(Psi, PsiPitch, devListVoxels, devListCoeff, Nx, Ny);   
    cudaDeviceSynchronize();

    printf("Finished getting voxels and coefficients\n");

    // re-initialize each grid point
    float thres = 0.0001;
    
    reinitPhi<<<dimGrid, dimBlock>>>(Phi, PhiPitch, Psi, devListVoxels, devListCoeff, Nx, Ny, dx, thres);
    cudaDeviceSynchronize() ;

    HANDLE_ERROR(cudaFree(devListVoxels));
    HANDLE_ERROR(cudaFree(devListCoeff));

}//_reinitFastMarchNewton



void LevelSetCUDA::_extendVelocityF(LevelSet* hostLS )
{
	
	printf("Extending velocity field F(x,y) from levelset outward.\n");

    	//size_t FextPitchInbytes;
    	//size_t AccptPitchInbytes;


 	HANDLE_ERROR( cudaMalloc((void**)&Fspeed, Ny*Nx*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc((void**)&Accept,  Ny*Nx*sizeof(float) ));
	HANDLE_ERROR( cudaMemset(Accept,0,sizeof(int)));    
	HANDLE_ERROR( cudaMalloc((void**)&Ftemp,  Ny*Nx*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc((void**)&Atemp,  Ny*Nx*sizeof(float) ));
	HANDLE_ERROR( cudaMemset(Atemp,0,sizeof(int))); 
	int PhiPitch = Nx;
	int AccptPitch = Nx;
	

	// Allocate memory on device 
//	HANDLE_ERROR( cudaMallocPitch((void**)&Fspeed, &FextPitchInbytes, Ny, Nx) );
//	HANDLE_ERROR( cudaMallocPitch((void**)&Accept, &AccptPitchInbytes, Ny , Nx));
//	HANDLE_ERROR( cudaMemset(Accept,0,sizeof(int)));    
//	HANDLE_ERROR( cudaMallocPitch((void**)&Ftemp, &FextPitchInbytes, Ny, Nx) );
//	HANDLE_ERROR( cudaMallocPitch((void**)&Atemp, &AccptPitchInbytes, Ny , Nx));
//	HANDLE_ERROR( cudaMemset(Atemp,0,sizeof(int))); 
//	int AccptPitch = (int)(AccptPitchInbytes / sizeof(int));

	FastMarchInit<<<dimGrid, dimBlock>>>(Phi, PhiPitch, Accept, AccptPitch, Fspeed, Nx, Ny, dx, dy, Xmin, Ymin);
	HANDLE_ERROR( cudaDeviceSynchronize() );
	cudaThreadSynchronize();
	
	
	//printf("\n%d,  %d\t", PhiPitch, AccptPitch);
	

    	int   * newA, * oldA;
     	float * newF, * oldF;  
	
	int count;
	bool toggle=true;
	int countMax = Nx*floor(log(Nx)); //Not sure this is the upperbound. Best guess based on algorithm.
	for (count=1 ; count<countMax ; ++count){
		if (toggle){
		toggle = false;
		oldA = Accept;
		newA = Atemp;
		oldF = Fspeed;
		newF = Ftemp;
		} else {
		toggle = true;
		oldA = Atemp;
		newA = Accept;
		oldF = Ftemp;
		newF = Fspeed;		
		}
	
		FastMarchVelocity<<<dimGrid, dimBlock>>>(count, Phi, PhiPitch, oldA, AccptPitch, oldF, newA, newF, Nx, Ny, dx, dy);
		HANDLE_ERROR( cudaDeviceSynchronize() );
		cudaThreadSynchronize(); //needed this to synchronize correctly between iterations		
	} 
	cudaDeviceSynchronize() ;
	
	


      HANDLE_ERROR( cudaMemcpy(hostLS->PhiGPU, Phi, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );
      HANDLE_ERROR( cudaMemcpy(hostLS->Fext, newF, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );
      HANDLE_ERROR( cudaMemcpy(hostLS->Accept, newA, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );

//     HANDLE_ERROR( cudaMemcpy(hostLS->PhiGPU, Phi, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );
//     HANDLE_ERROR( cudaMemcpy(hostLS->Fext, Fspeed, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );
//     HANDLE_ERROR( cudaMemcpy(hostLS->Accept, Accept, Ny*Nx*sizeof(float), cudaMemcpyDeviceToHost)  );
	
	
//    size_t PhiPitchInbytes = (size_t)(PhiPitch * sizeof(float));

//    // Copy data back to host
//    HANDLE_ERROR( cudaMemcpy2D(hostLS->PhiGPU, Ny*sizeof(float), Phi, PhiPitchInbytes, 
//			Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) );
//			
//    // Copy data back to host
//    HANDLE_ERROR( cudaMemcpy2D(hostLS->Fext, Ny*sizeof(float), Fspeed, FextPitchInbytes, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) );
//    //HANDLE_ERROR( cudaMemcpy2D(hostLS->h_Fext, Ny*sizeof(float), newF, FextPitchInbytes, Ny*sizeof(float), Nx, cudaMemcpyDeviceToHost) );
//    
//			
//    // Copy data back to host
//    HANDLE_ERROR( cudaMemcpy2D(hostLS->Accept, Ny*sizeof(int), Accept, AccptPitchInbytes, Ny*sizeof(int), Nx, cudaMemcpyDeviceToHost) );		
//    //HANDLE_ERROR( cudaMemcpy2D(hostLS->h_Accept, Ny*sizeof(int), newA, AccptPitchInbytes, Ny*sizeof(int), Nx, cudaMemcpyDeviceToHost) );
//    	
	
    HANDLE_ERROR(cudaFree(Fspeed));
    HANDLE_ERROR(cudaFree(Accept));	
    HANDLE_ERROR(cudaFree(Ftemp));
    HANDLE_ERROR(cudaFree(Atemp));	
	


}//_extendVelocityF











