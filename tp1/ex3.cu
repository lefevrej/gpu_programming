#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>

int N;

using namespace std::chrono;
using namespace std;

void racine(float *tab, float *out) {
    for (int i = 0; i < N; ++i) 
        out[i] = sqrtf(tab[i]);
}

__global__ void cuda_racine_block(float *f, float *f_out) {
    int tid = blockIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}

__global__ void cuda_racine_thread(float *f, float *f_out) {
    int tid = threadIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);

    float rnd_floats[N];
    float sqrt_floats[N];
    float sqrt_floats_cuda[N];

    srand((unsigned) time(NULL));
    for (int i = 0; i < N; ++i)
        rnd_floats[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

    /********* PAS CUDA *********/

    auto start = high_resolution_clock::now();

    racine(rnd_floats, sqrt_floats);

    auto stop = high_resolution_clock::now();
    duration < double > duration = stop - start;
    printf("Time to generate: %3.7f ms\n", duration.count() * 1000.0 f);

    ofstream myfile;
    myfile.open("cpu.txt");
    for (int i = 0; i < N; ++i) myfile << sqrt_floats[i] << "\n";
    myfile.close();

    /********* CUDA 1000 blocks *********/

    float * dev_rnd_floats, * dev_rnd_floats_out;
    cudaMalloc((void ** ) & dev_rnd_floats, N * sizeof(float));
    cudaMalloc((void ** ) & dev_rnd_floats_out, N * sizeof(float));
    cudaMemcpy(dev_rnd_floats, rnd_floats, N * sizeof(int), cudaMemcpyHostToDevice);

    float elapsedTime;
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate( & cuda_start);
    cudaEventCreate( & cuda_stop);
    cudaEventRecord(cuda_start, 0);

    cuda_racine_block <<< N, 1 >>> (dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime( & elapsedTime, cuda_start, cuda_stop);
    printf("Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    cudaFree(dev_rnd_floats_out);

    myfile.open("blocks.txt");
    for (int i = 0; i < N; ++i) myfile << sqrt_floats_cuda[i] << "\n";
    myfile.close();

    /********* CUDA 1000 threads *********/

    cudaMalloc((void**) & dev_rnd_floats_out, N * sizeof(float));

    cudaEventCreate( & cuda_start);
    cudaEventCreate( & cuda_stop);
    cudaEventRecord(cuda_start, 0);

    cuda_racine_thread <<< 1, N >>> (dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);

    cudaEventElapsedTime( & elapsedTime, cuda_start, cuda_stop);
    printf("Parallel time to generate:  %3.7f ms\n", elapsedTime);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    cudaFree(dev_rnd_floats);
    cudaFree(dev_rnd_floats_out);

    myfile.open("threads.txt");
    for (int i = 0; i < N; ++i) myfile << sqrt_floats_cuda[i] << "\n";
    myfile.close();

    return 0;
}