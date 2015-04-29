################################################################################
#
# Build script for project
#
################################################################################

#  -rdc=true  ::: enables relocatable-device-code allows extern keyword to be honored
#  --ptxas-opions=-v ::: compilation shows register usage of functions  


############################# Makefile ##########################

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64

LDFLAGS  :=  -L$(CUDA_LIB_PATH) -lcuda -lcudart 
CPPFLAGS :=  -g
NVCCFLAGS =  -g -G -arch=compute_50 -code=sm_50 -rdc=true --ptxas-options=-v



INCLUDES    := -I$(CUDA_INC_PATH) -I. -I.. -I$(CUDA_PATH)/samples/common/inc/ 

CC =       $(CUDA_BIN_PATH)/nvcc


all: main


main.o: main.cu
	$(CC) $(CPPFLAGS) $(INCLUDES) -o main.o -c main.cu  

LevelSet.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o LevelSet.o -c LevelSet.cpp   

fastmarching.o:
	$(CC)  $(CPPFLAGS) $(INCLUDES)  -o fastmarching.o -c fastmarching.cpp	

LevelSetGPU.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o LevelSetGPU.o -c LevelSetGPU.cu

Example_Kernel.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o Example_Kernel.o -c Example_Kernel.cu

Signed_Distance_Kernel.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o Signed_Distance_Kernel.o -c Signed_Distance_Kernel.cu

VoxelsFM_Kernel.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o VoxelsFM_Kernel.o -c VoxelsFM_Kernel.cu

VelocityFastMarch.o: 
	$(CC) $(CPPFLAGS) $(INCLUDES)  -o VelocityFastMarch.o -c VelocityFastMarch.cu	
	

main:   LevelSet.o fastmarching.o main.o LevelSetGPU.o Example_Kernel.o Signed_Distance_Kernel.o VoxelsFM_Kernel.o VelocityFastMarch.o
	$(CC) $(NVCCFLAGS) -o main LevelSet.o fastmarching.o LevelSetGPU.o Example_Kernel.o Signed_Distance_Kernel.o VoxelsFM_Kernel.o VelocityFastMarch.o main.o $(LDFLAGS)  

run: main
	./main

clean:
	rm -f main.exe *.o


