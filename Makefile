EXE=$(BDIR)/sobel_cpu $(BDIR)/ex5
CUDA=$(CUDA_SIMPLE) $(BDIR)/ex5_gpu.cu $(BDIR)/sobel_gpu.cu
CUDA_SIMPLE=ex2.cu ex3.cu ex4.cu hello.cu
CCFLAGS = -Wall -O2

IDIR = ./include
ODIR = ./obj
BDIR = ./bin
CDIR = ./src/com
LDIR = ./src/lib
CC = gcc
C++ = g++
NVCC = nvcc
X11LIB = /usr/X11/lib -lX11

all:	$(EXE)
cuda:	$(CUDA)

$(BDIR)/sobel_cpu:	$(CDIR)/sobel_cpu.cpp $(IDIR)/mcimage.h $(ODIR)/mcimage.o 
	$(C++) $(CCFLAGS) -I$(IDIR) $< $(ODIR)/mcimage.o -o $(BDIR)/sobel_cpu

$(BDIR)/ex5: $(CDIR)/ex5.cpp
	$(C++) $(CCFLAGS) $< -L$(X11LIB) -o $(BDIR)/ex5

$(BDIR)/ex5_gpu.cu: $(CDIR)/ex5_gpu.cu
	$(NVCC) $< -L$(X11LIB) -o $(BDIR)/ex5_gpu

$(BDIR)/sobel_gpu.cu: $(CDIR)/sobel_gpu.cu
	$(NVCC) -I$(IDIR) $< $(ODIR)/mcimage.o -o $(BDIR)/sobel_gpu	

$(CUDA_SIMPLE): 
	$(NVCC) $(CDIR)/$@ -o $(BDIR)/$(basename $@)

$(ODIR)/mcimage.o:	$(LDIR)/mcimage.c $(IDIR)/mcimage.h
	$(C++) -c $(CCFLAGS) -I$(IDIR) $< -o $(ODIR)/mcimage.o

clean:
	rm -rf *~ $(ODIR)/* $(IDIR)/*~ $(CDIR)/*~ $(LDIR)/*~ $(DOCDIR)/*~ 