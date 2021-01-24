#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <chrono>
#include <X11/Xlib.h>

#define N 1000

using namespace std::chrono;

struct cuComplex{
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void){ return r * r + i * i; }
    cuComplex operator*(const cuComplex& a){
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex & a){
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia (int x, int y){
    const float MIN = 1.5; //entre -1.5 et 1.5
    float jx = (float) (2*MIN*x/N) - MIN;
    float jy = (float) (2*MIN*y/N) - MIN;
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void kernel (unsigned char * ptr){
    int offset;
    for (int y=0; y<N; y++){
        for (int x = 0; x < N; x++) {
            offset = x + y * N;
            int juliaValue = julia (x, y);
            ptr[offset] = 255 * juliaValue;
        }
    }
}

/**
 * Returns coordinates so that the elements of interest are present
 *  in the submatrix defined by these coordinates. 
 */
void cropped_coordinates(unsigned char *ptr, int *x_o, int *y_o, int *x_e, int *y_e){
    int x_u=N, y_u=N, x_d=0, y_d=0;
    for (int i=0; i< N*N; i++){
        if(ptr[i]!=0){
            if(i/N<y_u) y_u=i/N;
            if(i%N<x_u) x_u=i%N;
            if(i/N>y_d) y_d=i/N;
            if(i%N>x_d) x_d=i%N;
        }
    }
    *x_o = std::max(0, x_u-10);
    *y_o = std::max(0, y_u-10);
    *x_e = std::min(N, x_d+10);
    *y_e = std::min(N, y_d+10);    
}

int main() {

    unsigned char tab[N*N];
    auto start = high_resolution_clock::now(); 

    kernel(tab);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf( "Time to generate:  0.%03ld ms\n", duration.count());

    int x_u, y_u, x_d, y_d;
    cropped_coordinates(tab, &x_u, &y_u, &x_d, &y_d);

    XEvent e;
    Display *dpy = XOpenDisplay(NULL); //pointer on screen
    int Noir = BlackPixel(dpy, DefaultScreen(dpy));
    int Blanc = WhitePixel(dpy, DefaultScreen(dpy)); 
    
    // creation fenetre: taille, couleur... :
    Window w = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, x_d-x_u, y_d-y_u, 0, Noir, Noir);
    XMapWindow(dpy, w); // Affiche la fenetre sur l'ecran
    GC gc = XCreateGC(dpy, w, 0, NULL);  //On a besoin d'un Graphic Context pour dessiner
    // Il faut attendre l'autorisation de dessiner
    XSelectInput(dpy, w, StructureNotifyMask);

    while (e.type != MapNotify) 
        XNextEvent(dpy, & e);

    // On dessine(enfin!) : 
    XSetForeground(dpy, gc, Blanc); //Couleur du stylo
    unsigned int x=0, y=0;
    
    FILE *f = fopen ("fractal.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",N,N);
    for (int i=0; i< N*N; i++){
        ++x;
        if(x==N){
            x=0;
            ++y;
        }
        fputc(tab[i],f);
        if(tab[i]!=0) XDrawPoint(dpy, w, gc, x-x_u, y-y_u);
    }
    fclose(f);
    XFlush(dpy); //Force l'affichage

    std::cin.get(); //Wait for input before leave program

    return 0;
}