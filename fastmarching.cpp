
#include "fastmarching.h"

inline float FastMarch::Val(const Grid& grid)
{
  return phi[grid.i+grid.j*m];
}

inline Coord FastMarch::Grid2Coord(const Grid& grid)
{
  return Coord(xstart+grid.j*dx, ystart+grid.i*dy);
}

void FastMarch::ComputeTentative(const Grid& grid)
{
  int g0 = grid.i+grid.j*m;
  int g[4] = {grid.i+(grid.j-1)*m, grid.i+1+grid.j*m, grid.i+(grid.j+1)*m, grid.i-1+grid.j*m};
  bool sides[4] = {grid.j>0 && status[g[0]]==-1, grid.i+1<m && status[g[1]]==-1,
		   grid.j+1<n && status[g[2]]==-1, grid.i>0 && status[g[3]]==-1};
  bool pairs[4] = {sides[0] & sides[1], sides[1] & sides[2], sides[2] & sides[3],
		   sides[3] & sides[0]};
  float Fval = F(Grid2Coord(grid));

  phi[g0] = 1.0e10;
  if(pairs[0] || pairs[1] || pairs[2] || pairs[3]){
    if(pairs[0]){
      phi[g0] = min(phi[g0], EvalL(Fval, phi[g[0]], phi[g[1]]));
    }
    if(pairs[1]){
      phi[g0] = min(phi[g0], EvalL(Fval, phi[g[2]], phi[g[1]]));
    }
    if(pairs[2]){
      phi[g0] = min(phi[g0], EvalL(Fval, phi[g[2]], phi[g[3]]));      
    }
    if(pairs[3]){
      phi[g0] = min(phi[g0], EvalL(Fval, phi[g[0]], phi[g[3]]));
    }
  } else {
    if(sides[0]){
      phi[g0] = min(phi[g0], phi[g[0]] + dx/Fval);
    }
    if(sides[1]){
      phi[g0] = min(phi[g0], phi[g[1]] + dy/Fval);
    }
    if(sides[2]){
      phi[g0] = min(phi[g0], phi[g[2]] + dx/Fval);
    }
    if(sides[3]){
      phi[g0] = min(phi[g0], phi[g[3]] + dy/Fval);
    }
  }  
}

inline float FastMarch::EvalL(float Fval, float phix, float phiy)
{
  float d = (dx*dx+dy*dy)/(Fval*Fval) - pow(phix-phiy,2);
  return ( (d<0) ? min(phix+dx/Fval, phiy+dy/Fval) : ((dy/dx*phix+dx/dy*phiy+sqrt(d))/(dx/dy+dy/dx)) );
}




void FastMarch::PopHeap()
{
  unsigned int ix=0;
  unsigned int c1=2*ix+1, c2=2*ix+2;
  while(c1 < heap.size()){
    if(c2 < heap.size() && Val(heap[c1]) > Val(heap[c2])){
      SwapPos(ix, c2);
      ix = c2;      
    } else {
      SwapPos(ix, c1);
      ix = c1;
    }
    c1 = 2*ix+1;
    c2 = 2*ix+2;
  }
  
  if(ix == heap.size()-1) return;

  SwapPos(ix, heap.size()-1);  
  while(ix!=0 && Val(heap[ix]) < Val(heap[(ix-1)/2])){
    SwapPos(ix, (ix-1)/2);
    ix = (ix-1)/2;
  }    
}

void FastMarch::SwapPos(int ix1, int ix2) {
  Grid c1 = heap[ix1], c2 = heap[ix2];
  heap[ix1] = c2;
  heap[ix2] = c1;
  status[c1.i+c1.j*m] = ix2;
  status[c2.i+c2.j*m] = ix1;
}

void FastMarch::Update(const Grid& grid)
{
  int ix = status[grid.i+grid.j*m];
  if(ix==-1) return;
  if(ix==-2){
    AddTentative(grid);
  } else {
    bool swapped = false;
    ComputeTentative(grid);
    
    while(ix!=0 && Val(heap[ix]) < Val(heap[(ix-1)/2])){
      SwapPos(ix, (ix-1)/2);
      ix = (ix-1)/2;
      swapped = true;
    }
    if(swapped) return;
    
    while(true) {
      unsigned int c1 = 2*ix+1, c2 = 2*ix+2;
      if(c1 < heap.size()){
        if(c2 < heap.size()){
          if(Val(heap[ix]) > Val(heap[c1])){
            if(Val(heap[c1]) < Val(heap[c2])){
              SwapPos(ix, c1);
              ix = c1;
            } else {
              SwapPos(ix, c2);
              ix = c2;
            }
            continue;
          } else {
            if(Val(heap[ix]) > Val(heap[c2])){
              SwapPos(ix, c2);
              ix = c2;
              continue;
            }
          }          
        } else {
          if(Val(heap[ix]) > Val(heap[c1])) {
            SwapPos(ix, c1);
            ix = c1;
          }
        }                
      }
      break;
    }
    
  }    
}

FastMarch::FastMarch(int m, int n, float xstart, float ystart, float dx, float dy, float *phi, 
                     int *status, float (*F) (const Coord &)) :
m(m), n(n), xstart(xstart), ystart(ystart), dx(dx), dy(dy), phi(phi), status(status), F(F) 
{}

void FastMarch::AddTentative(const Grid& grid)
{
  if(status[grid.i+grid.j*m] != -2) return;
  
  int ix = heap.size();  
  ComputeTentative(grid);  
  heap.push_back(grid);
  status[grid.i+grid.j*m] = ix;
  while(ix!=0 && Val(heap[ix]) < Val(heap[(ix-1)/2])){
    SwapPos(ix, (ix-1)/2);
    ix = (ix-1)/2;
  }  
}

void FastMarch::ClearTentative()
{
  heap.clear();
}



void FastMarch::UpdateNeighbors(const Grid& grid)
{
  if(grid.j > 0)
    Update(Grid(grid.i, grid.j-1));
  if(grid.i+1 < m)
    Update(Grid(grid.i+1, grid.j));
  if(grid.j+1 < n)
    Update(Grid(grid.i, grid.j+1));
  if(grid.i > 0)
    Update(Grid(grid.i-1, grid.j));              
}


void FastMarch::March()
{
  while(!heap.empty()){
    Grid grid = heap.front();
    PopHeap();
    heap.pop_back();
    status[grid.i+grid.j*m] = -1;
    UpdateNeighbors(grid);    
  }
}

