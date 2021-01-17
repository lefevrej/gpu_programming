#include <mcimage.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <chrono>

using namespace std::chrono;

void convolve(struct xvimage *in, struct xvimage *out, const float *kernel, const int kn){
  uint32_t rs, cs;
  int newval;
  uint8_t *ptr_in, *ptr_out;
  const int khalf = kn / 2;

  rs = in->row_size;
  cs = in->col_size;

  ptr_in = UCHARDATA(in);
  ptr_out = UCHARDATA(out);
  for (int m = khalf; m < (int) rs - khalf; ++m)
    for (int n = khalf; n < (int) cs - khalf; ++n) {
      newval = 0;
      size_t c = 0;
      for (int j = -khalf; j <= khalf; j++){
        for (int i = -khalf; i <= khalf; i++) {
          newval += ptr_in[(n - j) * rs + m - i] * kernel[c];
          c++;
        }
        ptr_out[n*rs+m]=(uint8_t)newval;
      }
    }
}

int sobel(struct xvimage *in, struct xvimage *out){
  int newval;
  uint32_t rs, cs;
  uint8_t *ptr_in, *ptr_out;
  float dx, dy;
  rs = in->row_size;
  cs = in->col_size;

  ptr_in = UCHARDATA(in);
  ptr_out = UCHARDATA(out);
  for (uint32_t x = 1; x < rs - 1; x++)
      for (uint32_t y = 1; y < cs - 1; y++) {
        if( x > 0 && y > 0 && x < rs-1 && y < cs-1) {
          dx = (-1* ptr_in[(y-1)*rs + (x-1)]) + (-2*ptr_in[y*rs+(x-1)]) + (-1*ptr_in[(y+1)*rs+(x-1)]) +
               (    ptr_in[(y-1)*rs + (x+1)]) + ( 2*ptr_in[y*rs+(x+1)]) + (   ptr_in[(y+1)*rs+(x+1)]);
          dy = (    ptr_in[(y-1)*rs + (x-1)]) + ( 2*ptr_in[(y-1)*rs+x]) + (   ptr_in[(y-1)*rs+(x+1)]) +
               (-1* ptr_in[(y+1)*rs + (x-1)]) + (-2*ptr_in[(y+1)*rs+x]) + (-1*ptr_in[(y+1)*rs+(x+1)]);
          newval = abs(dx) + abs(dy);
          ptr_out[y*rs + x] = newval > NDG_MAX ? NDG_MAX : newval;
        }
      }

  return 1;
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

  struct xvimage *out = allocimage(NULL, img->row_size, img->col_size, 1, VFF_TYP_1_BYTE);
  /*if(!sobel(img, out)){
    fprintf(stderr, "%s: function sobel failed\n",  argv[0]);
    exit(0);
  }*/

  auto start = high_resolution_clock::now(); 
  srand((unsigned)time(NULL));
  sobel(img, out);
  auto stop = high_resolution_clock::now();
  duration<double> duration = stop - start;
  printf( "Time to generate:  %3.7f ms\n", duration.count() * 1000.0F);

  std::cout << argv[2] << std::endl;
  writeimage(out, argv[2]);
  freeimage(img);
  freeimage(out);
}