#include <mcimage.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

texture<unsigned char> srcImg;

__global__ void sobel_gpu(unsigned char *out, const unsigned int rs, const unsigned int cs) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    int newval;
    if( x > 0 && y > 0 && x < rs-1 && y < cs-1) {
        dx = (-1* tex1Dfetch(srcImg,(y-1)*rs + (x-1))) + (-2*tex1Dfetch(srcImg,y*rs+(x-1))) + (-1*tex1Dfetch(srcImg,(y+1)*rs+(x-1))) +
             (    tex1Dfetch(srcImg,(y-1)*rs + (x+1))) + ( 2*tex1Dfetch(srcImg,y*rs+(x+1))) + (   tex1Dfetch(srcImg, (y+1)*rs+(x+1)));
        dy = (    tex1Dfetch(srcImg,(y-1)*rs + (x-1))) + ( 2*tex1Dfetch(srcImg,(y-1)*rs+x)) + (   tex1Dfetch(srcImg,(y-1)*rs+(x+1))) +
             (-1* tex1Dfetch(srcImg,(y+1)*rs + (x-1))) + (-2*tex1Dfetch(srcImg,(y+1)*rs+x)) + (-1*tex1Dfetch(srcImg,(y+1)*rs+(x+1)));
        /*dx = (-1* tex2D(srcImg,y-1, x-1)) + (-2*tex2D(srcImg,y, x-1)) + (-1*tex2D(srcImg,y+1,x-1)) +
            (         tex2D(srcImg,y-1, x+1)) + ( 2*tex2D(srcImg,y, x+1)) + (   tex2D(srcImg,y+1,x+1));
        dy = (    tex2D(srcImg,y-1,x-1)) + ( 2*tex2D(srcImg,y-1,x)) + (   tex2D(srcImg,y-1,x+1)) +
            (    -1* tex2D(srcImg,y+1,x-1)) + (-2*tex2D(srcImg,y+1,x)) + (-1*tex2D(srcImg,y+1,x+1));*/
             newval = abs(dx) + abs(dy);
        if (newval > NDG_MAX) newval = NDG_MAX;
        out[y*rs + x] = newval;
    }
}

int main(int argc, char **argv){
  struct xvimage * img;
  int b_cnt, t_cnt; // square root of thread and block count

  if (argc != 5){
    fprintf(stderr, "usage: %s img.pgm out.pgm b_cnt t_cnt\n", argv[0]);
    exit(0);
  }

  b_cnt = atoi(argv[3]);
  t_cnt = atoi(argv[4]);

  img = readimage(argv[1]);  
  if (img == NULL){
    fprintf(stderr, "%s: readimage failed\n", argv[0]);
    exit(0);
  }

  unsigned char *in, *out;
  unsigned int size = (img->row_size*img->col_size)*sizeof(unsigned  char);
  cudaMalloc( (void**)&in, size);
  cudaMalloc( (void**)&out, size);
  //cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  //cudaBindTexture2D(NULL, srcImg, in, img->row_size, img->col_size, sizeof(unsigned char));
  cudaBindTexture( NULL, srcImg, in, size);
  cudaMemset(out, 0, size);
  cudaMemcpy(in, img->image_data, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(t_cnt, t_cnt);
  dim3 numBlocks(b_cnt, b_cnt);

  cudaEvent_t cuda_start, cuda_stop;
  cudaEventCreate( &cuda_start);
  cudaEventCreate( &cuda_stop);
  cudaEventRecord( cuda_start, 0 );

  sobel_gpu<<<numBlocks, threadsPerBlock>>>( out, img->row_size, img->col_size);

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
  cudaUnbindTexture(srcImg);
  return 0;
  //0.9014400 ms
}