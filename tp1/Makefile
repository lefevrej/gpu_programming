#EXE=$(BDIR)/ex2.cu $(BDIR)/ex3.cu $(BDIR)/ex4.cu $(BDIR)/ex5_gpu.cu $(BDIR)/hello.cu $(BDIR)/ex5.cpp $(BDIR)/sobel_cpu.cpp

EXE=$(BDIR)/sobel_cpu $(BDIR)/ex5
CCFLAGS = -Wall -O2

IDIR = ./include
ODIR = ./obj
BDIR = ./bin
CDIR = ./src/com
LDIR = ./src/lib
CC = gcc
C++ = g++
nvcc = ncvv
X11LIB = /usr/X11/lib -lX11

all:	$(EXE)

$(BDIR)/sobel_cpu:	$(CDIR)/sobel_cpu.cpp $(IDIR)/mcimage.h $(ODIR)/mcimage.o 
	$(C++) $(CCFLAGS) -I$(IDIR) $(CDIR)/sobel_cpu.cpp $(ODIR)/mcimage.o -o $(BDIR)/sobel_cpu

$(BDIR)/ex5: $(CDIR)/ex5.cpp
	$(C++) $(CCFLAGS) $(CDIR)/ex5.cpp -L$(X11LIB) -o $(BDIR)/ex5


$(ODIR)/mcimage.o:	$(LDIR)/mcimage.c $(IDIR)/mcimage.h
	$(C++) -c $(CCFLAGS) -I$(IDIR) $(LDIR)/mcimage.c -o $(ODIR)/mcimage.o

clean:
	rm -rf *~ $(ODIR)/* $(IDIR)/*~ $(CDIR)/*~ $(LDIR)/*~ $(DOCDIR)/*~ 