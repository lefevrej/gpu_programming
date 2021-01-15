// nvcc -o ex5 ex5_gpu.cu -L/usr/X11/lib -lX11
#include <X11/Xlib.h>
#include <algorithm>
#include <iostream>

#define DIM 1000
struct DataBlock {
    unsigned char * dev_bitmap;
};

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b): r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator * (const cuComplex & a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator + (const cuComplex & a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char * ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int juliaValue = julia(x, y);
    ptr[offset] = 255 * juliaValue;
    /*ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;*/
}



void cropped_coordinates(unsigned char *ptr, int *x_o, int *y_o, int *x_e, int *y_e){
    int x_u=DIM, y_u=DIM, x_d=0, y_d=0;
    for (int i=0; i< DIM*DIM; i++){
        int x, y;
        x=i%DIM;
        y=i/DIM;
        if(((int)ptr[i])!=0){
            if(y<y_u) y_u=y;
            if(x<x_u) x_u=x;
            if(y>y_d) y_d=y;
            if(x>x_d) x_d=x;
        }
    }
    *x_o = std::max(0, x_u-10);
    *y_o = std::max(0, y_u-10);
    *x_e = std::min(DIM, x_d+10);
    *y_e = std::min(DIM, y_d+10);    
}

int main(void) {
    //DataBlock data;
    //CPUBitmap bitmap(DIM, DIM, & data);
    unsigned char *dev_bitmap, *tab= (unsigned char *)malloc(DIM * DIM * sizeof(unsigned char));
    cudaMalloc((void ** ) &dev_bitmap, DIM * DIM * sizeof(unsigned char));
    //data.dev_bitmap = dev_bitmap;
    dim3 grid(DIM, DIM);
    kernel<<< grid, 1 >>> (dev_bitmap);
    //cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    //cudaFree(dev_bitmap);
    //bitmap.display_and_exit();

    cudaMemcpy(tab, dev_bitmap, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::cout << "top" << std::endl;
    int x_u, y_u, x_d, y_d;
    cropped_coordinates(tab, &x_u, &y_u, &x_d, &y_d);
    std::cout << "top" << std::endl;
    std::cout << x_u << " : " << y_u << " : " << x_d << " : " << y_d << std::endl;

    /*XEvent e;
    Display *dpy = XOpenDisplay(NULL); //pointeur sur un ecran
    int Noir = BlackPixel(dpy, DefaultScreen(dpy));
    int Blanc = WhitePixel(dpy, DefaultScreen(dpy)); 
    
    // creation fenetre: taille, couleur... :
    Window w = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, x_d-x_u, y_d-y_u, 0, Noir, Noir);
    XMapWindow(dpy, w); // Affiche la fenetre sur l'ecran
    GC gc = XCreateGC(dpy, w, 0, NULL);  //On a besoin d'un Graphic Context pour dessiner
    // Il faut attendre l'autorisation de dessiner
    XSelectInput(dpy, w, StructureNotifyMask);

    while (e.type != MapNotify) 
        XNextEvent(dpy, &e);

    // On dessine(enfin!) : 
    XSetForeground(dpy, gc, Blanc); //Couleur du stylo*/
    unsigned int x=0, y=0;
    
    FILE *f = fopen ("fractal.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",DIM,DIM);
    for (int i=0; i<DIM*DIM; i++){
        ++x;
        if(x==DIM){
            x=0;
            ++y;
        }
        fputc(tab[i],f);
        //if(tab[i]!=0) XDrawPoint(dpy, w, gc, x-x_u, y-y_u);
    }
    fclose(f);
    //XFlush(dpy); //Force l'affichage

    cropped_coordinates(tab, &x_u, &y_u, &x_d, &y_d);
    std::cout << "top" << std::endl;
    std::cout << x_u << " : " << y_u << " : " << x_d << " : " << y_d << std::endl;

    //std::cin.get();*/

    return 0;
}