


#include <stdio.h>
#include <math.h>



///////////////////////////////// DEVICE FUNCTIONS  /////////////////////////////////


__device__ void _bicubicCoeff(float* alpha, float* psi, int pitch, int i, int j);
__device__ void _newtonStep(float* A, float& x, float& y, float& error, int xi, int yi);

__device__ int sgn(float val) {
    return ((0.0 < val) ? 1 : -1 );
}

///////////////////////////////// GLOBAL GPU FUNCTIONS  /////////////////////////////////


// Determines which voxels are active. 
// Gets coefficients for interpolating polynomial for each active voxel.
__global__ void getVoxels(float* psi, int pitch, int* voxelList, float* alphaList, int Nx, int Ny)
{
	int row = blockDim.x*blockIdx.x + threadIdx.x;
	int col = blockDim.y*blockIdx.y + threadIdx.y;

	if (row<Nx-1 && col<Ny-1)
	{
		int idx = row * pitch + col;
		int pairity = sgn(psi[idx]) + sgn(psi[idx+pitch]) + sgn(psi[idx+pitch+1]) + sgn(psi[idx+1]);

		if (-3<pairity && pairity<3)
		{
			int old = atomicAdd(voxelList,1);
			*(voxelList + old + 1) = idx;

			_bicubicCoeff(alphaList+16*old, psi, pitch, row, col);
		}

	}
}

__global__ void reinitPhi(float* phi, int pitch, float* psi, int* voxelList, float* alphaList, 
						int Nx, int Ny, float dx, float thres)
{

    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;

    if (row<Nx && col<Ny)
    {

		float* alpha;
		int idx, r ,c;

		float minDist = 7770000.0; //for error checking
		float xO, yO;
		float error;		

		for(int k = 0; k < voxelList[0]; ++k)
		{
		   idx = voxelList[k+1];
		   alpha = alphaList + 16*k;

		   r = idx/pitch;
		   c = idx%pitch;
		   xO = .5;
		   yO = .5;
		   bool inVoxel = true;

		   do
		   {
			   _newtonStep(alpha, xO, yO, error, row-r, col-c);
			   inVoxel =  (yO>=-0.1f) && (yO<=1.1f) && (xO>=-0.1f) && (xO<=1.1f);
			   
			   
		   } while (error>thres && inVoxel);

		   if (inVoxel){
			float xdist = (row-r-xO);
			float ydist = (col-c-yO);
			minDist = min(minDist, dx*sqrt(ydist*ydist + xdist*xdist));
		   }

			   
		}
		
//		if (minDist>100){
//			printf("\nWHAT IS UP HERE? err=%3.3f, (x,y)=(%2.2f,%2.2f)  [r,c]=%d,%d\n", error, xO,yO,row,col );
//		}

		phi[row*pitch+col] = sgn(psi[row*pitch+col]) * minDist;		

    
	}

}

///////////////////////////////// DEVICE FUNCTION IMPLEMENTATIONS /////////////////////////////////

__device__ void _bicubicCoeff(float* alpha, float* psi, int pitch, int i, int j)
{
	int idx = i*pitch + j;

	float    f00,   f10,   f01,   f11;
	float   fx00,  fx10,  fx01,  fx11;
	float   fy00,  fy10,  fy01,  fy11;
	float  fxy00, fxy10, fxy01, fxy11;

	//f00 = psi[idx];
	//f01 = psi[idx+pitch];//psi[idx+1];
	//f10 = psi[idx+1];//psi[idx+N];
	//f11 = psi[idx+pitch+1];//psi[idx+N+1];

	//fy00 = (psi[idx+pitch]-psi[idx-pitch])/2.0;//(psi[idx+1]-psi[idx-1])/2.0;
	//fy01 = (psi[idx+2*pitch]-psi[idx])/2.0;//(psi[idx+2]-psi[idx])/2.0;
	//fy10 = (psi[idx+pitch+1]-psi[idx-pitch+1])/2.0;//(psi[idx+N+1]-psi[idx+N-1])/2.0;
	//fy11 = (psi[idx+2*pitch+1]-psi[idx+1])/2.0;//(psi[idx+N+2]-psi[idx+N])/2.0;

	//fx00 = (psi[idx+1]-psi[idx-1])/2.0;//(psi[idx+N]-psi[idx-N])/2.0;
	//fx01 = (psi[idx+pitch+1]-psi[idx+pitch-1])/2.0;//(psi[idx+N+1]-psi[idx-N+1])/2.0;
	//fx10 = (psi[idx+2]-psi[idx])/2.0;//(psi[idx+2*N]-psi[idx])/2.0;
	//fx11 = (psi[idx+pitch+2]-psi[idx+pitch])/2.0;//(psi[idx+2*N+1]-psi[idx+1])/2.0;

	//fxy00 = (psi[idx+pitch+1]-psi[idx+pitch-1]-psi[idx-pitch+1]+psi[idx-pitch-1])/4.0;
	//fxy01 = (psi[idx+2*pitch+1]-psi[idx+2*pitch-1]-psi[idx+1]+psi[idx-1])/4.0;
	//fxy10 = (psi[idx+pitch+2]-psi[idx+pitch]-psi[idx-pitch+2]+psi[idx-pitch])/4.0;
	//fxy11 = (psi[idx+2*pitch+2]-psi[idx+2*pitch]-psi[idx+2]+psi[idx])/4.0;



	f00 = psi[idx];
	f01 = psi[idx+1];
	f10 = psi[idx+pitch];
	f11 = psi[idx+pitch+1];

	fy00 = (psi[idx+1]-psi[idx-1])/2.0;
	fy01 = (psi[idx+2]-psi[idx])/2.0;
	fy10 = (psi[idx+pitch+1]-psi[idx+pitch-1])/2.0;
	fy11 = (psi[idx+pitch+2]-psi[idx+pitch])/2.0;

	fx00 = (psi[idx+pitch]-psi[idx-pitch])/2.0;
	fx01 = (psi[idx+pitch+1]-psi[idx-pitch+1])/2.0;
	fx10 = (psi[idx+2*pitch]-psi[idx])/2.0;
	fx11 = (psi[idx+2*pitch+1]-psi[idx+1])/2.0;

	fxy00 = (psi[idx+pitch+1]-psi[idx+1-pitch]-psi[idx-1+pitch]+psi[idx-pitch-1])/4.0;
	fxy01 = (psi[idx+pitch+2]-psi[idx-pitch+2]-psi[idx+pitch]+psi[idx-pitch])/4.0;
	fxy10 = (psi[idx+2*pitch+1]-psi[idx+1]-psi[idx+2*pitch-1]+psi[idx-1])/4.0;
	fxy11 = (psi[idx+2*pitch+2]-psi[idx+2]-psi[idx+2*pitch]+psi[idx])/4.0;


	alpha[0] = f00;
	alpha[1] = fy00;
	alpha[2] = -3*f00 + 3*f01 - 2*fy00 - fy01;
	alpha[3] =  2*f00 - 2*f01 + fy00 + fy01;
	alpha[4] = fx00;
	alpha[5] = fxy00;
	alpha[6] = -3*fx00 + 3*fx01 - 2*fxy00 - fxy10;
	alpha[7] =  2*fx00 - 2*fx01 + fxy00 + fxy01;
	alpha[8] = -3*f00 + 3*f10 - 2*fx00 - fx10;
	alpha[9] = -3*fy00 + 3*fy10 - 2*fxy00 -fxy10;
	alpha[10] = 9*f00 -9*f01 -9*f10 +9*f11 +6*fy00 +3*fy01 -6*fy10 -3*fy11 
			   +6*fx00 -6*fx01 +3*fx10 -3*fx11 +4*fxy00 +2*fxy01 +2*fxy10 + fxy11;
	alpha[11] = -6*f00 +6*f01 +6*f10 -6*f11 -3*fy00 -3*fy01 +3*fy10 +3*fy11 
			    -4*fx00 +4*fx01 -2*fx10 +2*fx11 -2*fxy00 -2*fxy01 - fxy10 - fxy11;
	alpha[12] = 2*f00 - 2*f10 + fx00 + fx10;
	alpha[13] = 2*fy00 - 2*fy10 + fxy00 + fxy10;
	alpha[14] = -6*f00 +6*f01 +6*f10 -6*f11 -4*fy00 -2*fy01 +4*fy10 +2*fy11 
			    -3*fx00 +3*fx01 -3*fx10 +3*fx11 -2*fxy00 -fxy01 -2*fxy10 - fxy11;
	alpha[15] = 4*f00 -4*f01 -4*f10 +4*f11 +2*fy00 +2*fy01 -2*fy10 -2*fy11 
			    +2*fx00 -2*fx01 +2*fx10 -2*fx11 + fxy00 + fxy01 + fxy10 + fxy11;


}//_bicubicCoeff


__device__ void _newtonStep(float* A, float& x, float& y, float& error, int xi, int yi)
{
	float p, px, py, pxx, pyy, pxy;
	float d1, d2, d3, D;

	float y2 = y*y;
	float y3 = y2*y;
	float x2 = x*x;
	float x3 = x2*x;

	p =    A[0]  + A[1]*y  + A[2]*y2  + A[3]*y3
		+ (A[4]  + A[5]*y  + A[6]*y2  + A[7]*y3)*x
		+ (A[8]  + A[9]*y  + A[10]*y2 + A[11]*y3)*x2
		+ (A[12] + A[13]*y + A[14]*y2 + A[15]*y3)*x3;

	py =   A[1]  + 2*A[2]*y  + 3*A[3]*y2
		+ (A[5]  + 2*A[6]*y  + 3*A[7]*y2)*x
		+ (A[9]  + 2*A[10]*y + 3*A[11]*y2)*x2
		+ (A[13] + 2*A[14]*y + 3*A[15]*y2)*x3;

	px =   A[4] + 2*A[8]*x  + 3*A[12]*x2 
		+ (A[5] + 2*A[9]*x  + 3*A[13]*x2)*y
		+ (A[6] + 2*A[10]*x + 3*A[14]*x2)*y2
		+ (A[7] + 2*A[11]*x + 3*A[15]*x2)*y3;	

	pyy = 2*A[2] + 6*A[3]*y + (2*A[6]  + 6*A[7]*y)*x 
			   				+ (2*A[10] + 6*A[11]*y)*x2 
			   				+ (2*A[14] + 6*A[15]*y)*x3;

	pxx = 2*A[8] + 6*A[12]*x + (2*A[9]  + 6*A[13]*x)*y
							 + (2*A[10] + 6*A[14]*x)*y2  
							 + (2*A[11] + 6*A[15]*x)*y3;

	pxy =  A[5]  + 2*A[6]*y  + 3*A[7]*y2 +
	 	  (A[9]  + 2*A[10]*y + 3*A[11]*y2)*2*x +
		  (A[13] + 2*A[14]*y + 3*A[15]*y2)*3*x2;

	d1 =  py*(x-xi) -  px*(y-yi);
	d2 = pyy*(x-xi) - pxy*(y-yi) - px;
	d3 = pxy*(x-xi) - pxx*(y-yi) + py ;
	D  = py*d3 - px*d2;

	error = p*p + d1*d1;

	y -= ( p*d3  - px*d1) / D;
	x -= ( py*d1 - p*d2 ) / D;

}//_newtonStep


