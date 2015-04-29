

#ifndef _FASTMARCHING_H_
#define _FASTMARCHING_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>

using namespace std;

struct Grid {
  int i;
  int j;
  
  Grid(int i, int j) : i(i), j(j)
  {}
  
  Grid() {}
};

struct Coord {
  float x;
  float y;
  
  Coord(){}
  Coord(float x, float y): x(x), y(y) {}
};

class FastMarch {
  int m, n;  
  float xstart, ystart, dx, dy;
  float *phi;
  int *status; // -1=accepted, -2=distant, 0 1 2 ... = tentative 
  float (*F) (const Coord &);
  vector<Grid> heap;
  
  float Val(const Grid& grid);
  float EvalL(float Fval, float phix, float phiy);
  void ComputeTentative(const Grid& grid);
  Coord Grid2Coord(const Grid& grid);
  void PopHeap();
  void Update(const Grid& grid);
  void UpdateNeighbors(const Grid& grid);
  void SwapPos(int ix1, int ix2);
  
public:
  FastMarch(int m, int n, float xstart, float ystart, float dx, float dy,
            float *phi, int *status, float (*F)(const Coord &));
  void ClearTentative();
  void AddTentative(const Grid& grid);  
  void March();
 
};

#endif

