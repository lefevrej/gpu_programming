#include <corecrt_math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>


#define N 1000

using namespace std::chrono;

void racine(float* tab, float* out){
    for (int i =0; i < N; ++i){
        out[i]=sqrtf(tab[i]);
    }
}

__global__ void cuda_racine_block(float *f, float* f_out){
    int tid = blockIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}

__global__ void cuda_racine_thread(float *f, float* f_out){
    int tid = threadIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}



int main() {

    float rnd_floats[N];
    float sqrt_floats[N];
    float sqrt_floats_cuda[N];
    
    /********* PAS CUDA *********/
        
    for (int i =0; i < N; ++i){
        rnd_floats[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    auto start = high_resolution_clock::now(); 
    srand((unsigned)time(NULL));

    racine(rnd_floats, sqrt_floats);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf( "Time to generate:  0.%03lld ms\n", duration.count());

    /********* CUDA 1000 blocks *********/

    float *dev_rnd_floats, *dev_rnd_floats_out;
    cudaMalloc((void**)&dev_rnd_floats, N * sizeof(float));
    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));
    cudaMemcpy(dev_rnd_floats, rnd_floats, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_racine_block<<<N,1>>>(dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf( "Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats_out );

    /********* CUDA 1000 threads *********/

    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));

    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_racine_thread<<<1,N>>>(dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf( "Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats );
    cudaFree( dev_rnd_floats_out );

    return 0;
}