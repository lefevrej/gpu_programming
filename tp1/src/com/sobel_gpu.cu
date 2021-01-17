#include <mcimage.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

__global__ void sobel_gpu(unsigned char *in, unsigned char *out, const unsigned int rs, const unsigned int cs) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    int newval;
    if( x > 0 && y > 0 && x < rs-1 && y < cs-1) {
        dx = (-1* in[(y-1)*rs + (x-1)]) + (-2*in[y*rs+(x-1)]) + (-1*in[(y+1)*rs+(x-1)]) +
             (    in[(y-1)*rs + (x+1)]) + ( 2*in[y*rs+(x+1)]) + (   in[(y+1)*rs+(x+1)]);
        dy = (    in[(y-1)*rs + (x-1)]) + ( 2*in[(y-1)*rs+x]) + (   in[(y-1)*rs+(x+1)]) +
             (-1* in[(y+1)*rs + (x-1)]) + (-2*in[(y+1)*rs+x]) + (-1*in[(y+1)*rs+(x+1)]);
        newval = abs(dx) + abs(dy);
        if (newval > NDG_MAX) newval = NDG_MAX;
        out[y*rs + x] = newval;
    }
}

int main(int argc, char **argv){
  struct xvimage * img;

  if (argc != 3){
    fprintf(stderr, "usage: %s img.pgm out.pgm \n", argv[0]);
    exit(0);
  }

  img = readimage(argv[1]);  
  if (img == NULL){
    fprintf(stderr, "%s: readimage failed\n", argv[0]);
    exit(0);
  }

  unsigned char *in, *out;
  unsigned int size = img->row_size*img->col_size;
  cudaMalloc( (void**)&in, size*sizeof(unsigned  char));
  cudaMalloc( (void**)&out, size*sizeof(unsigned  char));
  cudaMemcpy(in, img->image_data, size, cudaMemcpyHostToDevice);
  cudaMemset(out, 0, size);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(16, 16);

  cudaEvent_t cuda_start, cuda_stop;
  cudaEventCreate( &cuda_start);
  cudaEventCreate( &cuda_stop);
  cudaEventRecord( cuda_start, 0 );

  sobel_gpu<<<numBlocks, threadsPerBlock>>>(in, out, img->row_size, img->col_size);

  cudaEventRecord( cuda_stop, 0 );
  cudaEventSynchronize( cuda_stop );
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
  printf( "Parallel time to generate:  %3.7f ms\n", elapsedTime);
  cudaEventDestroy( cuda_start );
  cudaEventDestroy( cuda_stop );

  cudaMemcpy(img->image_data, out, size, cudaMemcpyDeviceToHost);
  writeimage(img, argv[2]);
  freeimage(img);
  cudaFree(in);
  cudaFree(out);
  return 0;
  //0.9014400 ms
}