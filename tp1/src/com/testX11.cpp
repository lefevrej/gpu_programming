#include <X11/Xlib.h> // pour utiliser X11 => gcc-lX11

#include <unistd.h> //pour fonction d'attente sleep

#define LARGEUR 200
#define HAUTEUR 200
int main() {
  XEvent e;
  Display *dpy = XOpenDisplay(NULL); //pointeur sur un ecran
  int Noir = BlackPixel(dpy, DefaultScreen(dpy));
  int Blanc = WhitePixel(dpy, DefaultScreen(dpy)); 
  
  // creation fenetre: taille, couleur... :
  Window w = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, LARGEUR, HAUTEUR, 0, Noir, Noir);
  XMapWindow(dpy, w); // Affiche la fenetresur l'ecran
  GC gc = XCreateGC(dpy, w, 0, NULL);  //On a besoin d'un Graphic Context pour dessiner
  // Il faut attendre l'autorisation de dessiner
  XSelectInput(dpy, w, StructureNotifyMask);

  while (e.type != MapNotify) 
    XNextEvent(dpy, & e);

  // On dessine(enfin!) : 
  XSetForeground(dpy, gc, Blanc); //Couleur du stylo
  XDrawLine(dpy, w, gc, 0, 0, LARGEUR, HAUTEUR);
  XDrawPoint(dpy, w, gc, 10, 50);
  XDrawPoint(dpy, w, gc, 10, 51);
  XFlush(dpy); //Force l'affichage
  sleep(10); //on attend 10s avant de quitter
  
  return 0;
}