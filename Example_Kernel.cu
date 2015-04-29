


///////////////////////////////// DEVICE FUNCTIONS  /////////////////////////////////


//__device__ int sgn(float val) {
//    return ((0.0 < val) ? 1 : -1 );
//}



///////////////////////////////// GLOBAL GPU FUNCTIONS  /////////////////////////////////

__global__ void testkernel1(float* phi, int Nx, int Ny, int pitch)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx =  index_x * pitch +  index_y; 

	int row = idx/pitch;
	int col = idx%pitch;

	if (col<Ny && row<Nx) 
	{
		phi[idx] = (0.0 < phi[idx]) ? 1 : -1 ;//sgn(phi[idx]);		
	}
}






