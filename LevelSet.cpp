///////////////////////////////// INCLUDES /////////////////////////////////

//Headers for Application
#include "LevelSet.h"





inline int Levelset::c2g(float x) 
{
  return static_cast<int>(floor((x-Xmin)/dx));  //NOTE THAT THIS IS ASSUMING THAT Xmin=Ymin AND dx=dy
}

inline float Levelset::g2c(int i)
{
  return (Xmin+i*dx);  //NEEDS TO BE ADAPTED LATER FOR NON SYMMETRIC GRID
}



void LevelSet::_TestOutput(float * A, int nx, int ny)
{
    printf("\nOUTPUT OF PHI:\n");
    for(int i = 0; i < nx; ++i)
    {
        for(int j = 0; j < ny; ++j)
        {
            printf("%3.3f\t", A[i*ny+j]);
        }
        printf("\n");
    }
    printf("\n");

}//_TestOutput


void LevelSet::_TestOutputInt(int * A, int nx, int ny)
{
    printf("\nOUTPUT OF PHI:\n");
    for(int i = 0; i < nx; ++i)
    {
        for(int j = 0; j < ny; ++j)
        {
            printf("%d\t", (int)A[i*ny+j]);
        }
        printf("\n");
    }
    printf("\n");

}//_TestOutputInt





float LevelSet::_getDifference()
{
	float diff = 0;

    for(int i = 0; i < Nx; ++i)
    {
        for(int j = 0; j < Ny; ++j)
        {
            diff += sqrt( pow( PhiGPU[i*Ny+j]-PhiCPU[i*Ny+j] ,2) );
        }

    }

    return diff/(float)(Nx*Ny);
	
}






void LevelSet::_SavePhiToFile()
{
    //write data to file
    char filename[256];
    char str_t[128];
    sprintf(str_t, "%d_%d", Nx, Ny );
    sprintf(filename, "PHI_%s.bin", str_t);
    FILE* fileID = fopen(filename, "wb");

    fwrite(PhiGPU, sizeof(float), Nx*Ny , fileID);
  
    fclose(fileID);

}//_SavePhiToFile

















Coord Newton1x(const float a[], float y)
{
  float x = 0.5, y2 = y*y, y3 = y2*y;
  for(int it=0; it<10; ++it) {
    float x2 = x*x, x3 = x2*x;
    float p = a[0]+a[1]*x+a[2]*x2+a[3]*x3+a[4]*y+a[5]*x*y+a[6]*x2*y+a[7]*x3*y+a[8]*y2+a[9]*x*y2+a[10]*x2*y2+a[11]*x3*y2+a[12]*y3+a[13]*x*y3+a[14]*x2*y3+a[15]*x3*y3;
    float pp = a[1]+2*a[2]*x+3*a[3]*x2+a[5]*y+2*a[6]*x*y+3*a[7]*x2*y+a[9]*y2+2*a[10]*x*y2+3*a[11]*x2*y2+a[13]*y3+2*a[14]*x*y3+3*a[15]*x2*y3;
    x -= p/pp;
  }
  return Coord(x, y);
}

Coord Newton1y(const float a[], float x)
{
  float y = 0.5, x2 = x*x, x3 = x2*x;
  for(int it=0; it<10; ++it) {
    float y2 = y*y, y3 = y2*y;
    float p = a[0]+a[1]*x+a[2]*x2+a[3]*x3+a[4]*y+a[5]*x*y+a[6]*x2*y+a[7]*x3*y+a[8]*y2+a[9]*x*y2+a[10]*x2*y2+a[11]*x3*y2+a[12]*y3+a[13]*x*y3+a[14]*x2*y3+a[15]*x3*y3;
    float pp = a[4]+a[5]*x+a[6]*x2+a[7]*x3+2*a[8]*y+2*a[9]*x*y+2*a[10]*x2*y+2*a[11]*x3*y+3*a[12]*y2+3*a[13]*x*y2+3*a[14]*x2*y2+3*a[15]*x3*y2;
    y -= p/pp;
  }
  return Coord(x, y);
}


// Newton's Method to determine closet point on the interface 
Coord Newton2(const float a[], float xi, float yi)
{
  float x=0.5, y=0.5;
  float p, px, pxx, py, pyy, pxy;
  float c, d, D, P;
  
  for(int it=0; it<10; ++it){
    float x2 = x*x, x3 = x2*x, y2 = y*y, y3 = y2*y;    
    p=a[0]+a[1]*x+a[2]*x2+a[3]*x3+a[4]*y+a[5]*x*y+a[6]*x2*y+a[7]*x3*y+a[8]*y2+a[9]*x*y2+a[10]*x2*y2+a[11]*x3*y2+a[12]*y3+a[13]*x*y3+a[14]*x2*y3+a[15]*x3*y3;
    px=a[1]+2*a[2]*x+3*a[3]*x2+a[5]*y+2*a[6]*x*y+3*a[7]*x2*y+a[9]*y2+2*a[10]*x*y2+3*a[11]*x2*y2+a[13]*y3+2*a[14]*x*y3+3*a[15]*x2*y3;
    pxx=2*a[2]+6*a[3]*x+2*a[6]*y+6*a[7]*x*y+2*a[10]*y2+6*a[11]*x*y2+2*a[14]*y3+6*a[15]*x*y3;
    py=a[4]+a[5]*x+a[6]*x2+a[7]*x3+2*a[8]*y+2*a[9]*x*y+2*a[10]*x2*y+2*a[11]*x3*y+3*a[12]*y2+3*a[13]*x*y2+3*a[14]*x2*y2+3*a[15]*x3*y2;
    pyy=2*a[8]+2*a[9]*x+2*a[10]*x2+2*a[11]*x3+6*a[12]*y+6*a[13]*x*y+6*a[14]*x2*y+6*a[15]*x3*y;
    pxy=a[5]+2*a[6]*x+3*a[7]*x2+2*a[9]*y+4*a[10]*x*y+6*a[11]*x2*y+3*a[13]*y2+6*a[14]*x*y2+9*a[15]*x2*y2;    
    
    // note that (xi,yi) will either be (0,0) (0,1) (1,0) or (1,1)
    P = px*(y-yi)-py*(x-xi);
    c = pxx*(y-yi)-pxy*(x-xi)-py;
    d = pxy*(y-yi)+px-pyy*(x-xi);
    D = d*px-c*py;
    
    x -= (d*p - py*P)/D;
    y -= (-c*p + px*P)/D;    
  }
  
  if(p < EPSILON && P < EPSILON && x > -EPSILON && x < 1+EPSILON && y > EPSILON && y < 1+EPSILON)
    return Coord(x, y);
  return Coord(-1,-1);    
}






float speed1(const Coord &){
  return 1.0;
}


int Levelset::_Reinitialize()
{
  for(int i=0; i<Nx*Ny; ++i){
    signs[i] = 2*(PhiCPU[i]>0)-1;
    status[i] = -2;
    Dist[i] = LARGE;
  }
  
  printf("starting Reinit...\n");
  
  
  
  /* Bicubic Interpolation: */
  for(int i=0; i<Nx-1; ++i){
    for(int j=0; j<Ny-1; ++j){
      int parity = signs[i+j*Ny]+signs[i+1+j*Ny]+signs[i+(j+1)*Ny]+signs[i+1+(j+1)*Ny];
      
      
      if(abs(parity)!=4){
        if(j==0 || j==Ny-2) return 1;
        
        float a[16];
        float f[16];
        
        f[0] = PhiCPU[i+j*Ny];
        f[1] = PhiCPU[i+1+j*Ny];
        f[2] = PhiCPU[i+(j+1)*Ny];
        f[3] = PhiCPU[i+1+(j+1)*Ny];
        
        f[8] = (PhiCPU[i+(j+1)*Ny]-PhiCPU[i+(j-1)*Ny])/2;
        f[9] = (PhiCPU[i+1+(j+1)*Ny]-PhiCPU[i+1+(j-1)*Ny])/2;
        f[10] = (PhiCPU[i+(j+2)*Ny]-PhiCPU[i+j*Ny])/2;
        f[11] = (PhiCPU[i+1+(j+2)*Ny]-PhiCPU[i+1+j*Ny])/2;
        
        if(i==0) {
          f[4] = 0;
          f[6] = 0;
          f[12] = 0;
          f[14] = 0;          
        } else {
          f[4] = (PhiCPU[i+1+j*Ny]-PhiCPU[i-1+j*Ny])/2;
          f[6] = (PhiCPU[i+1+(j+1)*Ny]-PhiCPU[i-1+(j+1)*Ny])/2;
          f[12] = (PhiCPU[i+1+(j+1)*Ny]-PhiCPU[i+1+(j-1)*Ny]-PhiCPU[i-1+(j+1)*Ny]+PhiCPU[i-1+(j-1)*Ny])/4;
          f[14] = (PhiCPU[i+1+(j+2)*Ny]-PhiCPU[i+1+j*Ny]-PhiCPU[i-1+(j+2)*Ny]+PhiCPU[i-1+j*Ny])/4;          
        }
        
        if(i==Nx-2) {
          f[5] = 0;
          f[7] = 0;
          f[13] = 0;
          f[15] = 0;
        } else {
          f[5] = (PhiCPU[i+2+j*Ny]-PhiCPU[i+j*Ny])/2;
          f[7] = (PhiCPU[i+2+(j+1)*Ny]-PhiCPU[i+(j+1)*Ny])/2;
          f[13] = (PhiCPU[i+2+(j+1)*Ny]-PhiCPU[i+2+(j-1)*Ny]-PhiCPU[i+(j+1)*Ny]+PhiCPU[i+(j-1)*Ny])/4;
          f[15] = (PhiCPU[i+2+(j+2)*Ny]-PhiCPU[i+2+j*Ny]-PhiCPU[i+(j+2)*Ny]+PhiCPU[i+j*Ny])/4;
        }                                              
        
        
        // gets coefficients for Newton solve
        a[0] = f[0];
        a[1] = f[4];
        a[2] = -3*f[0]+3*f[1]-2*f[4]-f[5];
        a[3] = 2*f[0]-2*f[1]+f[4]+f[5];
        a[4] = f[8];
        a[5] = f[12];
        a[6] = -3*f[8]+3*f[9]-2*f[12]-f[13];
        a[7] = 2*f[8]-2*f[9]+f[12]+f[13];
        a[8] = -3*f[0]+3*f[2]-2*f[8]-f[10];
        a[9] = -3*f[4]+3*f[6]-2*f[12]-f[14];
        a[10] = 9*f[0]-9*f[1]-9*f[2]+9*f[3]+6*f[4]+3*f[5]-6*f[6]-3*f[7]+6*f[8]-6*f[9]+3*f[10]
        -3*f[11]+4*f[12]+2*f[13]+2*f[14]+f[15];
        a[11] = -6*f[0]+6*f[1]+6*f[2]-6*f[3]-3*f[4]-3*f[5]+3*f[6]+3*f[7]-4*f[8]+4*f[9]-2*f[10]
        +2*f[11]-2*f[12]-2*f[13]-f[14]-f[15];
        a[12] = 2*f[0]-2*f[2]+f[8]+f[10];
        a[13] = 2*f[4]-2*f[6]+f[12]+f[14];
        a[14] = -6*f[0]+6*f[1]+6*f[2]-6*f[3]-4*f[4]-2*f[5]+4*f[6]+2*f[7]-3*f[8]+3*f[9]-3*f[10]
        +3*f[11]-2*f[12]-f[13]-2*f[14]-f[15];
        a[15] = 4*f[0]-4*f[1]-4*f[2]+4*f[3]+2*f[4]+2*f[5]-2*f[6]-2*f[7]+2*f[8]-2*f[9]+2*f[10]
        -2*f[11]+f[12]+f[13]+f[14]+f[15];
        
        int nn = 0;
        Coord Xc[4];
        if(signs[i+j*Ny] != signs[i+1+j*Ny]){	    
          Xc[nn] = Newton1x(a, 0.0);
          ++nn;
        }
        if(signs[i+1+j*Ny] != signs[i+1+(j+1)*Ny]){
          Xc[nn] = Newton1y(a, 1.0);
          ++nn;
        }
        if(signs[i+1+(j+1)*Ny] != signs[i+(j+1)*Ny]){
          Xc[nn] = Newton1x(a, 1.0);
          ++nn;
        }
        if(signs[i+j*Ny] != signs[i+(j+1)*Ny]){
          Xc[nn] = Newton1y(a, 0.0);
          ++nn;
        }
        
        status[i+j*Ny] = -1;
        status[i+1+j*Ny] = -1;
        status[i+(j+1)*Ny] = -1;
        status[i+1+(j+1)*Ny] = -1;
        
        Coord c = Newton2(a, 0.0, 0.0);
        if(c.x >= 0){
          Dist[i+j*Ny] = min(Dist[i+j*Ny], (float)dx*sqrtf(c.x*c.x+c.y*c.y));          
        } else {
          for(int X=0; X<nn; ++X){	    
            Dist[i+j*Ny] = min(Dist[i+j*Ny], (float)dx*sqrtf(Xc[X].x*Xc[X].x+Xc[X].y*Xc[X].y));
          }
        }
        
        
        c = Newton2(a, 1.0, 0.0);
        if(c.x >= 0){
          Dist[i+1+j*Ny] = min(Dist[i+1+j*Ny], (float)dx*sqrtf(pow(c.x-1.0,2)+c.y*c.y));                    
        } else {
          for(int X=0; X<nn; ++X){	    
            Dist[i+1+j*Ny] = min(Dist[i+1+j*Ny], (float)dx*sqrtf(pow(Xc[X].x-1.0,2)+Xc[X].y*Xc[X].y));
          }
        }
        
        
        c = Newton2(a, 0.0, 1.0);
        if(c.x >= 0){
          Dist[i+(j+1)*Ny] = min(Dist[i+(j+1)*Ny], (float)dx*sqrtf(c.x*c.x+pow(c.y-1.0,2)));
        } else {
          for(int X=0; X<nn; ++X){	    
            Dist[i+(j+1)*Ny] = min(Dist[i+(j+1)*Ny], (float)dx*sqrtf(Xc[X].x*Xc[X].x+pow(Xc[X].y-1.0,2)));
          }
        }
        
        
        c = Newton2(a, 1.0, 1.0);
        if(c.x >= 0){
          Dist[i+1+(j+1)*Ny] = min(Dist[i+1+(j+1)*Ny], (float)dx*sqrtf(pow(c.x-1.0,2)+pow(c.y-1.0,2)));
        } else {
          for(int X=0; X<nn; ++X){	    
            Dist[i+1+(j+1)*Ny] = min(Dist[i+1+(j+1)*Ny], (float)dx*sqrtf(pow(Xc[X].x-1.0,2)+pow(Xc[X].y-1.0,2)));
          }
        }
        
        
        
      }      
    }
  }
  
  FastMarch fm(Nx, Ny, Xmin, Ymin, dx, dx, Dist, status, speed1);
  for(int i=0; i<Nx; ++i){
    for(int j=0; j<Ny; ++j){
      if(status[i+j*Ny] == -1){
        if(j != 0)
          fm.AddTentative(Grid(i, j-1));
        if(i != Nx-1)
          fm.AddTentative(Grid(i+1, j));
        if(j != Ny-1)
          fm.AddTentative(Grid(i, j+1));
        if(i != 0)
          fm.AddTentative(Grid(i-1, j));          
      }
    }
  }
  fm.March();
  for(int i=0; i<Nx*Ny; ++i){
    PhiCPU[i] = Dist[i] * signs[i];   
  }
  return 0;
}
















LevelSet::~LevelSet() //destructor
{ 
    delete[] PhiGPU;
    delete[] PhiCPU;
    delete[] Dist;
    delete[] status;
    delete[] signs;
    delete[] Fext;
    delete[] Accept;    
    delete[] xi;
    delete[] yj;
}








/////////////////////////////////// CONSTRUCTOR ////////////////////////////////////////////

LevelSet::LevelSet(int Nx, int Ny, int example): Nx(Nx), Ny(Ny), PhiGPU(new float[Nx*Ny]),
	PhiCPU(new float[Nx*Ny]), points(Nx*4), xi(new float[Nx*4+1]), yj(new float[Nx*4+1]),
	Fext(new float[Nx*Ny]), Accept(new int[Nx*Ny]), Dist(new float[Nx*Ny]), status(new int[Nx*Ny]),
 	signs(new int[Nx*Ny])   

{
    float x,y,s;

    switch (example){
        case 1 :  
	Xmin = -1.0;
	Ymin = -1.0;
	dx = 2.0/(Nx-1);
	dy = 2.0/(Ny-1);

            x = -1.0;
            y = -1.0;
            for(int i = 0; i < Nx; ++i)
            {
                for(int j = 0; j < Ny; ++j)
                {
                    PhiGPU[i*Ny+j] = (x * x + y * y - 0.25);
                    PhiCPU[i*Ny+j] = (x * x + y * y - 0.25);
                    y += dy;
                }
                y = -1.0;
                x += dx;
            }
	
        break;

        case 2 :  
	Xmin = -1.0;
	Ymin = -1.0;
	dx = 2.0/(Nx-1);
	dy = 2.0/(Ny-1);

            x = -1.0;
            y = -1.0;
            for(int i = 0; i < Nx; ++i)
            {
                for(int j = 0; j < Ny; ++j)
                {
                    PhiGPU[i*Ny+j] = 16*pow(x,4)-4*pow(x,2)+16*pow(y,4)-4*pow(y,2)+0.2;
                    PhiCPU[i*Ny+j] = 16*pow(x,4)-4*pow(x,2)+16*pow(y,4)-4*pow(y,2)+0.2;
                    y += dy;
                }
                y = -1.0;
                x += dx;
            }
	
        break;

        case 3 : // h_Phi is left blank 
            for (int i=0 ; i < points; ++i ){ // TEST CASE from 449 hw3 1.ii
                s = (2.0*PI / (float)points ) * i; //note domain above [Xmin,Xmax]=[-2,2] 
                xi[i] = (1.0 + 0.25*cos(4*s)) * cos(s);
                yj[i] = (1.0 + 0.25*cos(4*s)) * sin(s);
            }
	// close the curve
	xi[points] = xi[0];
	yj[points] = yj[0]; 
	Xmin = -2.0;
	Ymin = -2.0;
	dx = 4.0/(Nx-1);
	dy = 4.0/(Ny-1);	   
        break;

        case 4 : //for debugging indexing problem
            for(int i = 0; i < Nx; ++i)
            {
                for(int j = 0; j < Ny; ++j)
                {
                    PhiGPU[i*Ny+j] = (float)(i*Ny+j);
                    PhiCPU[i*Ny+j] = (float)(i*Ny+j);
                }
            }
        break;

        default :  
	Xmin = -1.0;
	Ymin = -1.0;
	dx = 2.0/(Nx-1);
	dy = 2.0/(Ny-1);

            x = -1.0;
            y = -1.0;
            for(int i = 0; i < Nx; ++i)
            {
                for(int j = 0; j < Ny; ++j)
                {
                    PhiGPU[i*Ny+j] = x * x + y * y - 0.25;
                    PhiCPU[i*Ny+j] = x * x + y * y - 0.25;
                    y += dy;
                }
                y = -1.0;
                x += dx;
            }
	
        break;
        
        


    }


}





