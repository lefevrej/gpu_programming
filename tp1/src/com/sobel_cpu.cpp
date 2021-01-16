#include <mcimage.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

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
        if (newval < NDG_MIN) newval = NDG_MIN;
			  if (newval > NDG_MAX) newval = NDG_MAX;
        ptr_out[n*rs+m]=(uint8_t)newval;
      }
    }
}

int sobel(struct xvimage *in, struct xvimage *out){
  int newval;
  uint32_t rs, cs;
  uint8_t *ptr_x, *ptr_y, *ptr_out;
  rs = in->row_size;
  cs = in->col_size;

  const float Gy[] = { 1, 2, 1,
                       0, 0, 0,
                      -1,-2,-1};
  const float Gx[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};

  struct xvimage *grad_x = allocimage(NULL, rs, cs, 1, VFF_TYP_1_BYTE);
  struct xvimage *grad_y = copyimage(grad_x);
  
  convolve(in, grad_x, Gx, 3);
  convolve(in, grad_y, Gy, 3);

  ptr_x = UCHARDATA(grad_x);
  ptr_y = UCHARDATA(grad_y);
  ptr_out = UCHARDATA(out);
  for (int i = 1; i < (int) rs - 1; i++)
      for (int j = 1; j < (int) cs - 1; j++) {
        const int c = i + rs * j;
        newval = abs(ptr_x[c]) + abs(ptr_y[c]);
			  if (newval > NDG_MAX) newval = NDG_MAX;
        ptr_out[c] = (uint8_t) newval;
      }
  freeimage(grad_x);
  freeimage(grad_y);
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
  if(!sobel(img, out)){
    fprintf(stderr, "%s: function sobel failed\n",  argv[0]);
    exit(0);
  }
  std::cout << argv[2] << std::endl;
  writeimage(out, argv[2]);
  freeimage(img);
  freeimage(out);
}