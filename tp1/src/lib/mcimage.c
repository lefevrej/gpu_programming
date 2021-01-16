/* 
   Librairie mcimage : 

   fonctions pour les entrees/sortie fichiers et l'allocation de structures
   image en memoire.

   Michel Couprie
   gcc -c -Wall -I. mcimage.c -o mcimage.o
*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mcimage.h>
#include <stdlib.h>
#include <ctype.h>

#define BUFFERSIZE 4096

#ifdef UNIXIO
#  define __pink__inline inline
#  include <stdint.h>
#else /* NOT UNIXIO */
#  define __pink__inline
typedef unsigned char u_int8_t;
typedef unsigned int  u_int32_t;
typedef unsigned char uint8_t;
typedef int           int32_t;
typedef unsigned int  uint32_t;
#endif /* UNIXIO */

// Fopen strategy
FILE* pink_fopen_read( char * filename )
{
# ifdef UNIXIO
  return fopen( filename, "r" );
# else /* NOT UNIXIO */
#   ifdef DOSIO
    return fopen( filename, "rb" );
#   else /* NOT DOSIO */
    //#     warning: FALLING BACK ON DEFAULT
      return fopen( filename, "rb" );
#   endif /* NOT DOSIO */
# endif /* NOT UNIXIO */
} /* pink_fopen_read */

FILE* pink_fopen_write( char * filename )
{
# ifdef UNIXIO
  return fopen( filename, "w" );
# else /* NOT UNIXIO */
#   ifdef DOSIO
    return fopen( filename, "wb" );
#   else /* NOT DOSIO */
    //#     warning: FALLING BACK ON DEFAULT
      return fopen( filename, "wb" );
#   endif /* NOT DOSIO */
# endif /* NOT UNIXIO */
} /* pink_fopen_write */

/* ==================================== */
struct xvimage *allocimage(
  char * name,
  int32_t rs,   /* row size */
  int32_t cs,   /* col size */
  int32_t d,    /* depth */
  int32_t t)    /* data type */
/* ==================================== */
#undef F_NAME
#define F_NAME "allocimage"
{
  int32_t N = rs * cs * d;             /* taille image */
  struct xvimage *g;
  int32_t ts;                          /* type size */

  switch (t)
  {
    case VFF_TYP_BIT:      ts = 1; break;
    case VFF_TYP_1_BYTE:   ts = 1; break;
    case VFF_TYP_2_BYTE:   ts = 2; break;
    case VFF_TYP_4_BYTE:   ts = 4; break;
    case VFF_TYP_FLOAT:    ts = sizeof(float); break;
    case VFF_TYP_DOUBLE:   ts = sizeof(double); break;
    default: fprintf(stderr,"%s : bad data type %d\n", F_NAME, t);
             return NULL;
  } /* switch (t) */

  g = (struct xvimage *)malloc(sizeof(struct xvimage));
  if (g == NULL)
  {   
    fprintf(stderr,"%s : malloc failed (%ld bytes)\n", F_NAME, sizeof(struct xvimage));
    return NULL;
  }

  g->image_data = (void *)calloc(1, N * ts);
  if (g == NULL)
  {   
    fprintf(stderr,"%s : calloc failed (%d bytes)\n", F_NAME, N * ts);
    return NULL;
  }

  if (name != NULL)
  {
    g->name = (char *)calloc(1,strlen(name)+1);
    if (g->name == NULL)
    {   fprintf(stderr,"%s : malloc failed for name\n", F_NAME);
        return NULL;
    }
    strcpy((char *)(g->name), name);
  }
  else
    g->name = NULL;

  rowsize(g) = rs;
  colsize(g) = cs;
  depth(g) = d;
  datatype(g) = t;
  g->xdim = g->ydim = g->zdim = 0.0;

  return g;
} /* allocimage() */

/* ==================================== */
void freeimage(struct xvimage *image)
/* ==================================== */
{
  free(image->image_data);
  free(image);
}

/* ==================================== */
struct xvimage *copyimage(struct xvimage *f)
/* ==================================== */
#undef F_NAME
#define F_NAME "copyimage"
{
  int32_t rs = rowsize(f);         /* taille ligne */
  int32_t cs = colsize(f);         /* taille colonne */
  int32_t ds = depth(f);           /* nb plans */
  int32_t N = rs * cs * ds;        /* taille image */
  int32_t type = datatype(f);
  struct xvimage *g;

  g = allocimage(NULL, rs, cs, ds, type);
  if (g == NULL)
  {
    fprintf(stderr,"%s : allocimage failed\n", F_NAME);
    return NULL;
  }

  switch(type)
  {
    case VFF_TYP_1_BYTE: memcpy(g->image_data, f->image_data, (N*sizeof(int8_t))); break;
    case VFF_TYP_4_BYTE: memcpy(g->image_data, f->image_data, (N*sizeof(int32_t))); break;
    case VFF_TYP_FLOAT:  memcpy(g->image_data, f->image_data, (N*sizeof(float))); break;
    case VFF_TYP_DOUBLE: memcpy(g->image_data, f->image_data, (N*sizeof(double))); break;
    default:
      fprintf(stderr,"%s : bad data type %d\n", F_NAME, type);
      return NULL;
  } /* switch(f->datatype) */

  if (f->name != NULL)
  {
    g->name = (char *)calloc(1,strlen(f->name) + 1);
    if (g->name == NULL)
    {   fprintf(stderr,"%s : malloc failed for name\n", F_NAME);
        return NULL;
    }
    strcpy(g->name, f->name);
  }
  return g;
} // copyimage()

/* ==================================== */
void writeimage(struct xvimage * image, char *filename)
/* ==================================== */
#undef F_NAME
#define F_NAME "writeimage"
{
  writeascimage(image, filename); 
} /* writeimage() */

/* ==================================== */
void writeascimage(struct xvimage * image, char *filename)
/* ==================================== */
#undef F_NAME
#define F_NAME "writeascimage"
{
  FILE *fd = NULL;
  int32_t rs, cs, ps, d, nndg, N, i;

  fd = fopen(filename,"w");
  if (!fd)
  {
    fprintf(stderr, "%s: cannot open file: %s\n", F_NAME, filename);
    exit(0);
  }

  rs = rowsize(image);
  cs = colsize(image);
  d = depth(image);
  ps = rs * cs;
  N = ps * d;

  if (datatype(image) == VFF_TYP_1_BYTE)
  {
    fputs("P2\n", fd);
    if ((image->xdim != 0.0) && (d > 1))
      fprintf(fd, "#xdim %g\n#ydim %g\n#zdim %g\n", image->xdim, image->ydim, image->zdim);
    if (d > 1) fprintf(fd, "%d %d %d\n", rs, cs, d); else  fprintf(fd, "%d %d\n", rs, cs);
    fprintf(fd, "255\n");

    for (i = 0; i < N; i++)
    {
      if (i % rs == 0) fprintf(fd, "\n");
      if (i % ps == 0) fprintf(fd, "\n");
      fprintf(fd, "%3d ", (int32_t)(UCHARDATA(image)[i]));
    } /* for i */
    fprintf(fd, "\n");
  }
  else if (datatype(image) == VFF_TYP_4_BYTE)
  {
    fputs("PB\n", fd);
    if ((image->xdim != 0.0) && (d > 1))
      fprintf(fd, "#xdim %g\n#ydim %g\n#zdim %g\n", image->xdim, image->ydim, image->zdim);
    if (d > 1) fprintf(fd, "%d %d %d\n", rs, cs, d); else  fprintf(fd, "%d %d\n", rs, cs);
    fprintf(fd, "4294967295\n");

    for (i = 0; i < N; i++)
    {
      if (i % rs == 0) fprintf(fd, "\n");
      if (i % ps == 0) fprintf(fd, "\n");
      fprintf(fd, "%d ", ULONGDATA(image)[i]);
    } /* for i */
    fprintf(fd, "\n");
  }
  else if (datatype(image) == VFF_TYP_FLOAT)
  {
    fputs("PA\n", fd);
    if ((image->xdim != 0.0) && (d > 1))
      fprintf(fd, "#xdim %g\n#ydim %g\n#zdim %g\n", image->xdim, image->ydim, image->zdim);
    if (d > 1) fprintf(fd, "%d %d %d\n", rs, cs, d); else  fprintf(fd, "%d %d\n", rs, cs);
    fprintf(fd, "1\n");

    for (i = 0; i < N; i++)
    {
      if (i % rs == 0) fprintf(fd, "\n");
      if (i % ps == 0) fprintf(fd, "\n");
      fprintf(fd, "%8g ", FLOATDATA(image)[i]);
    } /* for i */
    fprintf(fd, "\n");
  }
  fclose(fd);
}

/* ==================================== */
void printimage(struct xvimage * image)
/* ==================================== */
#undef F_NAME
#define F_NAME "printimage"
{
  int32_t rs, cs, d, ps, N, i;

  rs = rowsize(image);
  cs = colsize(image);
  d = depth(image);
  ps = rs * cs;
  N = ps * d;

  for (i = 0; i < N; i++)
  {
    if (i % rs == 0) printf("\n");
    if (i % ps == 0) printf("\n");
    printf("%3d ", (int32_t)(UCHARDATA(image)[i]));
  } /* for i */
  printf("\n");
}

/* ==================================== */
struct xvimage * readimage(char *filename)
/* ==================================== */
#undef F_NAME
#define F_NAME "readimage"
{
  char buffer[BUFFERSIZE];
  FILE *fd = NULL;
  int32_t rs, cs, d, ndgmax, N, i;
  struct xvimage * image;
  int32_t ascii;  
  int32_t typepixel;
  int32_t c;
  double xdim=1.0, ydim=1.0, zdim=1.0;
  char *read;

  fd = fopen(filename,"r");
  if (!fd)
  {
    fprintf(stderr, "%s: file not found: %s\n", F_NAME, filename);
    return NULL;
  }

  read = fgets(buffer, BUFFERSIZE, fd); /* P5: raw byte bw  ; P2: ascii bw */
                                 /* P6: raw byte rgb ; P3: ascii rgb */
                                 /* P8: raw int32_t 2d-3d  ==  extension MC */
                                 /* P9: raw float 2d-3d  ==  extension MC */
                                 /* PA: ascii float 2d-3d  ==  extension LN */
                                 /* PB: ascii int32_t 2d-3d  ==  extension MC */

                                 /* P7: raw byte 3d : OBSOLETE - left for compatibility */
  if (!read)
  {
    fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
    return 0;
  }

  if (buffer[0] != 'P')
  {   fprintf(stderr,"%s : invalid image format\n", F_NAME);
      return NULL;
  }
  switch (buffer[1])
  {
    case '2': ascii = 1; typepixel = VFF_TYP_1_BYTE; break;
    case '5':
    case '7': ascii = 0; typepixel = VFF_TYP_1_BYTE; break;
    case '8': ascii = 0; typepixel = VFF_TYP_4_BYTE; break;
    case '9': ascii = 0; typepixel = VFF_TYP_FLOAT; break;
    case 'A': ascii = 1; typepixel = VFF_TYP_FLOAT; break;
    case 'B': ascii = 1; typepixel = VFF_TYP_4_BYTE; break;
    default:
      fprintf(stderr,"%s : invalid image format : P%c\n", F_NAME, buffer[1]);
      return NULL;
  } /* switch */

  do
  {
    read = fgets(buffer, BUFFERSIZE, fd); /* commentaire */
    if (!read)
    {
      fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
      return 0;
    }
    if (strncmp(buffer, "#xdim", 5) == 0)
      sscanf(buffer+5, "%lf", &xdim);
    else if (strncmp(buffer, "#ydim", 5) == 0)
      sscanf(buffer+5, "%lf", &ydim);
    else if (strncmp(buffer, "#zdim", 5) == 0)
      sscanf(buffer+5, "%lf", &zdim);
  } while (!isdigit(buffer[0]));

  c = sscanf(buffer, "%d %d %d", &rs, &cs, &d);
  if (c == 2) d = 1;
  else if (c != 3)
  {   fprintf(stderr,"%s : invalid image format\n", F_NAME);
      return NULL;
  }

  read = fgets(buffer, BUFFERSIZE, fd);
  if (!read)
  {
    fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
    return 0;
  }

  sscanf(buffer, "%d", &ndgmax);
  N = rs * cs * d;

  image = allocimage(NULL, rs, cs, d, typepixel);
  if (image == NULL)
  {   fprintf(stderr,"%s : alloc failed\n", F_NAME);
      return(NULL);
  }
  image->xdim = xdim;
  image->ydim = ydim;
  image->zdim = zdim;

  if (typepixel == VFF_TYP_1_BYTE)
  {
    if (ascii)
    {
      if (ndgmax == 255)
        for (i = 0; i < N; i++)
        {
          fscanf(fd, "%d", &c);
          UCHARDATA(image)[i] = (uint8_t)c;
        } /* for i */
      else if (ndgmax == 65535)
        for (i = 0; i < N; i++)
        {
          fscanf(fd, "%d", &c);
          UCHARDATA(image)[i] = (uint8_t)(c/256);
        } /* for i */
      else
      {
        fprintf(stderr,"%s : wrong ndgmax = %d\n", F_NAME, ndgmax);
        return(NULL);
      }
    }
    else
    {
      int32_t ret = fread(UCHARDATA(image), sizeof(char), N, fd);
      if (ret != N)
      {
        fprintf(stderr,"%s : fread failed : %d asked ; %d read\n", F_NAME, N, ret);
        return(NULL);
      }
    }
  } /* if (typepixel == VFF_TYP_1_BYTE) */
  else
  if (typepixel == VFF_TYP_4_BYTE)
  {
    if (ascii)
    {
      for (i = 0; i < N; i++)
      {
        fscanf(fd, "%d", &(ULONGDATA(image)[i]));
      } /* for i */
    }
    else 
    {
      int32_t ret = fread(ULONGDATA(image), sizeof(int32_t), N, fd);
      if (ret != N)
      {
        fprintf(stderr,"%s : fread failed : %d asked ; %d read\n", F_NAME, N, ret);
        return(NULL);
      }
    }
  } /* if (typepixel == VFF_TYP_4_BYTE) */
  else
  if (typepixel == VFF_TYP_FLOAT)
  {
    if (ascii)
    {
      for (i = 0; i < N; i++)
      {
        fscanf(fd, "%f", &(FLOATDATA(image)[i]));
      } /* for i */
    }
    else 
    {
      int32_t ret = fread(FLOATDATA(image), sizeof(float), N, fd);
      if (ret != N)
      {
        fprintf(stderr,"%s : fread failed : %d asked ; %d read\n", F_NAME, N, ret);
        return(NULL);
      }
    }
  } /* if (typepixel == VFF_TYP_FLOAT) */

  fclose(fd);

  return image;
} /* readimage() */

/* ==================================== */
int32_t readrgbimage(
  char *filename,
  struct xvimage ** r,
  struct xvimage ** g,
  struct xvimage ** b)
/* ==================================== */
#undef F_NAME
#define F_NAME "readrgbimage"
{
  char buffer[BUFFERSIZE];
  FILE *fd = NULL;
  index_t rs, cs, i, ds;
  index_t N;
  int32_t ascii = 0;  
  int32_t c;
  char *read;
  int32_t nndg, ndgmax;

  fd = pink_fopen_read(filename);

  if (!fd)
  {
    fprintf(stderr, "%s: file not found: %s\n", F_NAME, filename);
    return 0;
  }

  read = fgets(buffer, BUFFERSIZE, fd); /* P6: raw int32_t rgb ; P3: ascii rgb */
  if (!read)
  {
    fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
    return 0;
  }
  if (buffer[0] != 'P')
  {   fprintf(stderr,"%s: invalid image format\n", F_NAME);
      return 0;
  }

  switch (buffer[1])
  {
    case '3': ascii = 1; break;
    case '6': ascii = 0; break;
    default:
      fprintf(stderr,"%s: invalid image format\n", F_NAME);
      return 0;
  } /* switch */

#ifdef MC_64_BITS
  c = sscanf(buffer+2, "%lld %lld %d", (long long int *)&rs, (long long int *)&cs, &ndgmax);
#else
  c = sscanf(buffer+2, "%d %d %d", (int *)&rs, (int *)&cs, (int *)&ndgmax);
#endif

  if (c == 3) /* format ppm MatLab : tout sur une ligne */
  {
    ds = 1;
    goto readdata;
  }



  do 
  {
    read = fgets(buffer, BUFFERSIZE, fd); /* commentaire */
    if (!read)
    {
      fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
      return 0;
    }
  } while (!isdigit(buffer[0]));

#ifdef MC_64_BITS
  c = sscanf(buffer, "%lld %lld", (long long int *)&rs, (long long int *)&cs);
#else
  c = sscanf(buffer, "%d %d", (int *)&rs, (int *)&cs);
#endif
  if (c != 2)
  {   fprintf(stderr,"%s: invalid image format\n", F_NAME);
      return 0;
  }

  read = fgets(buffer, BUFFERSIZE, fd);
  if (!read)
  {
    fprintf(stderr, "%s: fgets returned without reading\n", F_NAME);
    return 0;
  }
  sscanf(buffer, "%d", (int *)&nndg);

  readdata:
  N = rs * cs;

  *r = allocimage(NULL, rs, cs, 1, VFF_TYP_1_BYTE);
  *g = allocimage(NULL, rs, cs, 1, VFF_TYP_1_BYTE);
  *b = allocimage(NULL, rs, cs, 1, VFF_TYP_1_BYTE);
  if ((*r == NULL) && (*g == NULL) && (*b == NULL))
  {   fprintf(stderr,"%s: allocimage failed\n", F_NAME);
      return(0);
  }

  if (ascii)
    for (i = 0; i < N; i++)
    {
      fscanf(fd, "%d", &c);
      (UCHARDATA(*r))[i] = (uint8_t)c;
      fscanf(fd, "%d", &c);
      (UCHARDATA(*g))[i] = (uint8_t)c;
      fscanf(fd, "%d", &c);
      (UCHARDATA(*b))[i] = (uint8_t)c;
    } /* for i */
  else
    for (i = 0; i < N; i++)
    {
      (UCHARDATA(*r))[i] = fgetc(fd);
      (UCHARDATA(*g))[i] = fgetc(fd);
      (UCHARDATA(*b))[i] = fgetc(fd);
    } /* for i */

  fclose(fd);
  return 1;
} /* readrgbimage() */

/* ==================================== */
void writergbimage(
  struct xvimage * redimage,
  struct xvimage * greenimage,
  struct xvimage * blueimage,
  char *filename)
/* ==================================== */
#undef F_NAME
#define F_NAME "writergbimage"
{
  FILE *fd = NULL;
  index_t rs, cs, i;
  index_t N;
  int32_t nndg;

  fd = pink_fopen_write(filename);

  if (!fd)
  {
    fprintf(stderr, "%s: cannot open file: %s\n", F_NAME, filename);
    exit(0);
  }

  rs = redimage->row_size;
  cs = redimage->col_size;
  if ((greenimage->row_size != rs) || (greenimage->col_size != cs) ||
      (blueimage->row_size != rs) || (blueimage->col_size != cs))
  {
    fprintf(stderr, "%s: incompatible image sizes\n", F_NAME);
    exit(0);
  }
  
  N = rs * cs;
  nndg = 255;

  fputs("P6\n", fd);
  /*  fputs("# CREATOR: writeimage by MC - 07/1996\n", fd); */
  fprintf(fd, "##rgb\n");
#ifdef MC_64_BITS
  fprintf(fd, "%lld %lld\n", (long long int)rs, (long long int)cs);
#else
  fprintf(fd, "%d %d\n", rs, cs);
#endif
  fprintf(fd, "%d\n", nndg);

  for (i = 0; i < N; i++)
  {
    fputc((int32_t)(UCHARDATA(redimage)[i]), fd);
    fputc((int32_t)(UCHARDATA(greenimage)[i]), fd);
    fputc((int32_t)(UCHARDATA(blueimage)[i]), fd);
  } /* for i */

  fclose(fd);
} // writergbimage()
