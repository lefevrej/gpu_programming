#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>

int N;

using namespace std::chrono;

void add(float* tab1, float* tab2, float* out){
    for (int i =0; i < N; ++i)
        out[i]=tab1[i]+tab2[i];
}

__global__ void cuda_add_block(float *f1, float *f2, float* f_out){
    int tid = blockIdx.x;
    f_out[tid] = f1[tid]+f2[tid];
}

__global__ void cuda_add_thread(float *f1, float *f2, float* f_out){
    int tid = threadIdx.x;
    f_out[tid] = f1[tid]+f2[tid];
}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);

    /*float rnd_floats1[N];
    float rnd_floats2[N];
    float sum[N];
    float sum_cuda[N];*/

    float *rnd_floats1 = (float*) malloc(N*sizeof(float));
    float *rnd_floats2 = (float*) malloc(N*sizeof(float));
    float *sum = (float*) malloc(N*sizeof(float));
    float *sum_cuda= (float*) malloc(N*sizeof(float));
    
    /********* PAS CUDA *********/
        
    for (int i =0; i < N; ++i){
        rnd_floats1[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        rnd_floats2[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    auto start = high_resolution_clock::now(); 
    srand((unsigned)time(NULL));

    add(rnd_floats1, rnd_floats2, sum);
    
    auto stop = high_resolution_clock::now();
    duration<double> duration = stop - start;
    printf( "Time to generate:  %3.7f ms\n", duration.count() * 1000.0F);

    /********* CUDA N blocks *********/

    float *dev_rnd_floats1, *dev_rnd_floats2, *dev_rnd_floats_out;
    cudaMalloc((void**)&dev_rnd_floats1, N * sizeof(float));
    cudaMalloc((void**)&dev_rnd_floats2, N * sizeof(float));
    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));
    cudaMemcpy(dev_rnd_floats1, rnd_floats1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rnd_floats2, rnd_floats2, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_add_block<<<N,1>>>(dev_rnd_floats1, dev_rnd_floats2, dev_rnd_floats_out);
    cudaMemcpy(sum_cuda, dev_rnd_floats_out, N * sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf( "Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats_out );

    /********* CUDA N threads *********/

    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));

    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_add_thread<<<1,N>>>(dev_rnd_floats1, dev_rnd_floats2, dev_rnd_floats_out);
    cudaMemcpy(sum_cuda, dev_rnd_floats_out, N * sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf( "Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats1 );
    cudaFree( dev_rnd_floats2 );
    cudaFree( dev_rnd_floats_out );

    return 0;
}
