/* 
   Librairie mcimage : 

   fonctions pour les entrees/sortie fichiers et l'allocation de structures
   image en memoire.

   Michel Couprie
*/

#include <stdint.h> 
#define NDG_MAX 255            /* Max grey-level / niveau de gris max */
#define NDG_MIN 0              /* Min grey-level / niveau de gris min */

// Index should be signed
#ifdef MC_64_BITS
typedef int64_t index_t;
#else
typedef int32_t index_t;
#endif

/* definitions for data storage type,
   uint32_t data_storage_type; */
#define	VFF_TYP_BIT		0	/* pixels are on or off (binary image)*/
                                        /* Note: This is an X11 XBitmap 
					   with bits packed into a byte and
					   padded to a byte */
#define	VFF_TYP_1_BYTE		1	/* pixels are byte (uint8_t) */
#define	VFF_TYP_2_BYTE		2	/* pixels are two byte (int16_t) */
#define	VFF_TYP_4_BYTE		4	/* pixels are four byte (integer) */
#define	VFF_TYP_FLOAT		5	/* pixels are float (single precision)*/
#define VFF_TYP_DOUBLE		9	/* pixels are float (double precision)*/

struct xvimage {
  char *name;
  uint32_t row_size;                    /* Size of a row (number of columns) */
  uint32_t col_size;                    /* Size of a column (number of rows) */
  uint32_t depth_size;                  /* Number of planes (for 3d images) */
  uint32_t time_size;                   /* Number of (2d or 3d) images */
  uint32_t num_data_bands;	        /* Number of bands per data pixel,
					   or number of bands per image, or
					   dimension of vector data, or
					   number of elements in a vector */
  uint32_t data_storage_type;           /* storage type for disk data */
  double xdim, ydim, zdim;              /* voxel dimensions in real world */
  void * image_data;                    /* pointer on raw data */
};

#define UCHARDATA(I)  ((uint8_t*)((I)->image_data))
#define USHORTDATA(I) ((uint16_t*)((I)->image_data))
#define ULONGDATA(I)  ((uint32_t*)((I)->image_data))
#define FLOATDATA(I)  ((float*)((I)->image_data))
#define DOUBLEDATA(I) ((double*)((I)->image_data))
#define colsize(I)    ((I)->col_size)
#define rowsize(I)    ((I)->row_size)
#define depth(I)      ((I)->depth_size)
#define tsize(I)      ((I)->time_size)
#define nbands(I)     ((I)->num_data_bands)
#define datatype(I)   ((I)->data_storage_type)

/* ============== */
/* prototypes for mcimage.c    */
/* ============== */

extern struct xvimage *allocimage(
  char * name,
  int rs,
  int cs, 
  int d,
  int t
);

extern void freeimage(
  struct xvimage *image
);

extern struct xvimage *copyimage(
        struct xvimage *f
);

extern void writeimage(
  struct xvimage * image,
  char *filename
);

extern void writeascimage(
  struct xvimage * image,
  char *filename
);

extern void printimage(
  struct xvimage * image
);

extern struct xvimage * readimage(
  char *filename
);

extern int32_t readrgbimage(
  char *filename,
  struct xvimage ** r,
  struct xvimage ** g,
  struct xvimage ** b
);

extern void writergbimage(
  struct xvimage * redimage,
  struct xvimage * greenimage,
  struct xvimage * blueimage,
  char *filename
);