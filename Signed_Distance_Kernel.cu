
/*
This function takes the set of points (xj,yj) defining a closed curve
and populates the signed distance function Phi. 

The time for this should be of order Nx*Ny*points. 
Each of the Nx*Ny grid point independently loops through all points to determine 
its minDist from curve and if it is located inside or outside of the closed curve.

*/

#include <stdio.h>
#include <math.h>



__global__ void SetPhi(float* phi, int Nx, int Ny, int pitch, int points, float * fx, float * fy, 
		       float Xmin, float Ymin, float dx, float dy)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx =  index_x * pitch +  index_y; 
	int row = idx/pitch;
	int col = idx%pitch;


  float sign = 1;
  float area = 0;
  float minDist = 1000;

  float x = Xmin + col * dx; 
  float y = Ymin + row * dy;


  for (int i = 0; i < points ; ++i) 
  {
    // Sign-determine part.
    float fx_i = fx[i];
    float fy_i = fy[i];
    float fx_ipp = fx[i+1];
    float fy_ipp = fy[i+1];
    float cross = (fx_i - x) * (fy_ipp - y) - (fy_i - y) * (fx_ipp - x);
    float dot = (fx_i - x) * (fx_ipp - x) + (fy_i - y) * (fy_ipp - y);
    area += atan2(cross, dot);
    
    // Distance-determine part.
    float top = (x - fx_ipp) * (fx_i - fx_ipp) + (y - fy_ipp) * (fy_i - fy_ipp);
    float bot = (fx_i - fx_ipp) * (fx_i - fx_ipp) + (fy_i - fy_ipp) * (fy_i - fy_ipp);
    float ratio = top / bot;
    float distSq = 0;
    if (ratio > 1) {
      float xdiff = x - fx_i;
      float ydiff = y - fy_i;

      distSq = hypotf(xdiff, ydiff);

    } else if (ratio < 0) {
      float xdiff = x - fx_ipp;
      float ydiff = y - fy_ipp;

      distSq = hypotf(xdiff, ydiff);

    } else {
      float ux = (1 - ratio) * fx_ipp + ratio * fx_i;
      float uy = (1 - ratio) * fy_ipp + ratio * fy_i;
      ux = x - ux;
      uy = y - uy;
      distSq = hypotf(ux, uy);  
    }

    if (distSq < minDist) {
      minDist = distSq;

    }
  }

  if (area < 0.1) {  sign = -1; }

  float xx = sign * (minDist);

  if (col<Ny && row<Nx ){
    phi[idx] = xx ;

    printf("%d \t [%d][%d]=(%3.3f,%3.3f) \t %3.4f  \t %3.3f \n",idx, row, col, x,y, minDist, area);    
  }

}
