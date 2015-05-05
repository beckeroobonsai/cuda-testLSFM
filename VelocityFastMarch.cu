

#include <stdio.h>
#include <math.h>
#include <float.h>


__device__ int sign(float val) {
    return ((0.0 < val) ? 1 : -1 );
}




__global__ void FastMarchInit(float* phi, int pitch, int* Accept, int Apitch, float* Fext, int Nx, int Ny, float dx, float dy, float Xmin, float Ymin)
							   //float* XXo, float* YYo) // testing that the grid coord calculated right
{

	// This will initialize the Velocity field near the interface
	// ASSUMES THAT THE SIGNED DISTANCE FUNCTION DOES NOT NEED REINITIALIZATION 
	// Phi assumed accurate throughout domain. Only calculating Velocity extension Fext
	
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  =  index_x * pitch +  index_y; 
	int idxA =  index_x * Apitch +  index_y; 

//	int row = idx/pitch;
//	int col = idx%pitch;

	int row = idxA/Apitch;
	int col = idxA%Apitch;


	float dd = sqrt(dx*dx/4+dy*dy/4); //to check if interface is nearby

	int   Aval  = 0;
	float Fval  = 0;
	
	if (col<Ny-1 && row<Nx-1 && col>1 && row>1){
		if (abs(phi[idx])<dd){
			Aval = 1;
			Fval = Xmin + row * dx; 
			//Accept[idxA] = 1;// mark as on interface
			//Fext[idx] = Xmin + row * dx; // NOTE TO DO LATER: build more flexible function than F(x,y)=x
		
		}
		else if ( abs(phi[idx - pitch])<dd || abs(phi[idx + pitch])<dd || abs(phi[idx - 1])<dd || abs(phi[idx + 1])<dd  ){
			Aval = -1;
			Fval = Xmin + row * dx; 
			//Accept[idxA] = -11; // mark as tentative because it is a neigbor of interface
			//Fext[idx] =  Xmin + row * dx; 
			//printf("\nWHY!!!??? \t (r,c)=%d,%d\t , %d, %d, \t F=%3.3f, A=%d  \n", row,col, idxA, idx, Fval, Aval );
		}

	
		Accept[idxA] = Aval;
		Fext[idx]    = Fval;
	
		//if (Aval>2){
		//	printf("\nWHY!!!??? \t (r,c)=%d,%d\t , %d, %d, \t F=%3.3f, A=%d  \n", row,col, idxA, idx, Fval, Aval );
		//}	
	}

}





__global__ void FastMarchVelocity(int count, float* phi, int pitch, int* Accept, int Apitch, float* Fext, int* AcceptNew, float* FextNew, int Nx, int Ny, float dx, float dy)
{

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx =  index_x * pitch +  index_y;
	int idxA =  index_x * Apitch +  index_y; 

	int row = idx/pitch;
	int col = idx%pitch;
	
	float Dxp, Dxm, Dyp, Dym;
	float Dx, Dy, FDx, FDy;

	// put into local memory from global {center, top, right, bottom, left}
	int   A[5]  = { Accept[idxA], Accept[idxA - Apitch], Accept[idxA + 1], Accept[idxA + Apitch], Accept[idxA - 1] } ;
	float F[5]  = { Fext[idx], Fext[idx-pitch], Fext[idx+1], Fext[idx+pitch], Fext[idx-1] } ;
	float P[5]  = { phi[idx], phi[idx-pitch], phi[idx+1], phi[idx+pitch], phi[idx-1] } ;
			

	//Dxm = (row==0)    ? 0 : -F[1] * abs(A[1]) * (P[0] - P[1])/dx ;
	//Dxp = (row==Nx-1) ? 0 : F[3] * abs(A[3]) * (P[3] - P[0])/dx ;
	//Dym = (col==0)    ? 0 : -F[4] * abs(A[4]) * (P[0] - P[4])/dy ;
	//Dyp = (col==Ny-1) ? 0 : F[2] * abs(A[2]) * (P[2] - P[0])/dy ;

	
	Dxm = (row==0)    ? 0 : abs(A[1]) * (P[0] - P[1])/dx ;
	Dxp = (row==Nx-1) ? 0 : abs(A[3]) * (P[3] - P[0])/dx ;
	Dym = (col==0)    ? 0 : abs(A[4]) * (P[0] - P[4])/dy ;
	Dyp = (col==Ny-1) ? 0 : abs(A[2]) * (P[2] - P[0])/dy ;

	
	int tentative = 0;	
	tentative += (row==0)    ? 0 : abs(A[1]);
	tentative += (row==Nx-1) ? 0 : abs(A[3]);
	tentative += (col==0)    ? 0 : abs(A[4]);
	tentative += (row==Nx-1) ? 0 : abs(A[2]);
	
	

	if ((-Dxp>Dxm) && (-Dxp>0)){
		FDx = F[3] * (-Dxp);
		Dx  = -Dxp;
	}else if ((Dxm>-Dxp) &&(Dxm>0)){
		FDx = F[1] * (Dxm);
		Dx  = Dxm;	
	}else{
		FDx = 0;
		Dx  = 0;		
	}
	
	if ((-Dyp>Dym) && (-Dyp>0)){
		FDy = F[2] * (-Dyp);
		Dy  = -Dyp;
	}else if ((Dym>-Dyp) &&(Dym>0)){
		FDy = F[4] * (Dym);
		Dy  = Dym;	
	}else{
		FDy = 0;
		Dy  = 0;		
	}			
	



	
	//if ( row<Nx && col<Ny && A[0]!=1  && tentative>0 && (Dx+Dy)>0 ) {	
	if ( row<Nx && col<Ny && A[0]!=1  && tentative>0  ) {
	//if ( row<Nx && col<Ny && A[0]!=1  ) {		
		F[0] = (FDx + FDy)/(Dx + Dy + DBL_EPSILON);
		A[0] = -1; 
	} 

	//__syncthreads();

//	if (row<Nx && col<Ny && tentative==0 && count>49){
//		printf("\n count= %d\t, (r,c)=(%d,%d)\t A= {%d,%d,%d,%d} \n ", count, row, col, A[1], A[2], A[3], A[4] );
//	}

	
//	if (A[0]>2){
//		printf("\nWHY!!!??? \t (r,c)=%d,%d\t , %d, %d, \t F=%3.3f, A=%d  \n", row,col, idxA, idx, F[0], A[0] );
//	}
	
	AcceptNew[idxA] = A[0];
	FextNew[idx]    = F[0];
	//__syncthreads();


}






