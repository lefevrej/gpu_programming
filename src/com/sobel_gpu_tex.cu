#include <mcimage.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

texture<unsigned char> srcImg; //texture

__global__ void sobel_gpu(unsigned char *out, const unsigned int rs, const unsigned int cs) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    int newval;
    // if x and y are belongs to the image convolve pixel by Gx and Gy
    if( x > 0 && y > 0 && x < rs-1 && y < cs-1) {
        dx = (-1* tex1Dfetch(srcImg,(y-1)*rs + (x-1))) + (-2*tex1Dfetch(srcImg,y*rs+(x-1))) + (-1*tex1Dfetch(srcImg,(y+1)*rs+(x-1))) +
             (    tex1Dfetch(srcImg,(y-1)*rs + (x+1))) + ( 2*tex1Dfetch(srcImg,y*rs+(x+1))) + (   tex1Dfetch(srcImg, (y+1)*rs+(x+1)));
        dy = (    tex1Dfetch(srcImg,(y-1)*rs + (x-1))) + ( 2*tex1Dfetch(srcImg,(y-1)*rs+x)) + (   tex1Dfetch(srcImg,(y-1)*rs+(x+1))) +
             (-1* tex1Dfetch(srcImg,(y+1)*rs + (x-1))) + (-2*tex1Dfetch(srcImg,(y+1)*rs+x)) + (-1*tex1Dfetch(srcImg,(y+1)*rs+(x+1)));
        /*dx = (-1* tex2D(srcImg,y-1, x-1)) + (-2*tex2D(srcImg,y, x-1)) + (-1*tex2D(srcImg,y+1,x-1)) +
            (         tex2D(srcImg,y-1, x+1)) + ( 2*tex2D(srcImg,y, x+1)) + (   tex2D(srcImg,y+1,x+1));
        dy = (    tex2D(srcImg,y-1,x-1)) + ( 2*tex2D(srcImg,y-1,x)) + (   tex2D(srcImg,y-1,x+1)) +
            (    -1* tex2D(srcImg,y+1,x-1)) + (-2*tex2D(srcImg,y+1,x)) + (-1*tex2D(srcImg,y+1,x+1));*/
        newval = abs(dx) + abs(dy); // merge the two derivatives
        if (newval > NDG_MAX) newval = NDG_MAX; // if newval > 255 newval = 0
        out[y*rs + x] = newval; // store the result in the area out
    }
}

int main(int argc, char **argv){
  struct xvimage * img;
  int b_cnt, t_cnt;

  /***************** Read args *****************/

  if (argc != 5){
    fprintf(stderr, "usage: %s img.pgm out.pgm b_cnt t_cnt\n", argv[0]);
    exit(0);
  }

  b_cnt = atoi(argv[3]); //square root of the number of blocks
  t_cnt = atoi(argv[4]); //square root of the number of threads per blocks

  img = readimage(argv[1]);  
  if (img == NULL){
    fprintf(stderr, "%s: readimage failed\n", argv[0]);
    exit(0);
  }

  /***************** Init *****************/

  unsigned char *dev_src, *dev_out;
  unsigned int size = (img->row_size*img->col_size)*sizeof(unsigned  char); //size of the image
  cudaMalloc( (void**)&dev_src, size); // dev_src is the memory area on device for the texture
  cudaMalloc( (void**)&dev_out, size); // dev_out is the memory area on devide for the result
  //cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  //cudaBindTexture2D(NULL, srcImg, in, img->row_size, img->col_size, sizeof(unsigned char));
  cudaBindTexture( NULL, srcImg, dev_src, size); // bind texture with dev_src area
  cudaMemset(dev_out, 0, size); // dev_out is set to 0
  cudaMemcpy(dev_src, img->image_data, size, cudaMemcpyHostToDevice); //image data is copied in texture

  dim3 threadsPerBlock(t_cnt, t_cnt);
  dim3 numBlocks(b_cnt, b_cnt);

  /***************** Record execution time *****************/

  cudaEvent_t cuda_start, cuda_stop;
  cudaEventCreate( &cuda_start);
  cudaEventCreate( &cuda_stop);
  cudaEventRecord( cuda_start, 0 );

  sobel_gpu<<<numBlocks, threadsPerBlock>>>( dev_out, img->row_size, img->col_size);

  cudaEventRecord( cuda_stop, 0 );
  cudaEventSynchronize( cuda_stop );
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
  printf("Parallel time to generate:  %3.7f ms\n", elapsedTime);

  cudaEventDestroy( cuda_start );
  cudaEventDestroy( cuda_stop );

  /***************** Write image *****************/

  cudaMemcpy(img->image_data, dev_out, size, cudaMemcpyDeviceToHost); //retrive results on CPU
  writeimage(img, argv[2]);

  /***************** Free and unbind *****************/
  
  freeimage(img);
  cudaFree(dev_src);
  cudaFree(dev_out);
  cudaUnbindTexture(srcImg);
  return 0;
}