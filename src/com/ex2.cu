#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount( &count );
    for(int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "   ---General Information for device%d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Computecapability:  %d.%d\n", prop.major, prop.minor);
        printf( "Clockrate:  %d\n", prop.clockRate);
        printf( "   ---Memory Information for device%d ---\n", i );
        printf( "Total global mem:  %lu\n", prop.totalGlobalMem);
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem);
        printf( "Max mem pitch:  %ld\n", prop.memPitch);
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment);printf( "   ---MP Information for device%d ---\n", i );
        printf( "Multiprocessorcount:  %d\n",  prop.multiProcessorCount);printf( "Sharedmem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf( "Registersper mp:  %d\n", prop.regsPerBlock);
        printf( "Threads in warp:  %d\n", prop.warpSize);
        printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf( "Max thread dimensions:  (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
        printf( "Max griddimensions:  (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
        printf( "\n" );
    }
    return 0;
}