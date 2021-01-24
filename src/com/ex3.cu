/*
The purpose of this program is to compare the performance of calculating
the square root element-wise on an array. The 3 types of executions compared
will be CPU, GPU with only blocks and GPU with only threads.

The size of array <N> is taken as a parameter when the program is executed.
*/

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

void write_table(float *t, string fname){
    ofstream myfile;
    myfile.open(fname);
    for (int i = 0; i < N; ++i) myfile << t[i] << "\n";
    myfile.close();
}

// Compares two arrays and print error if there is a difference.
void cmp_tab(float *t1, float *t2){
    for(int i=0; i<N; ++i)
        if(t1[i]!=t2[i]){
            printf("Error at index %d: %f - %f\n", i, t1[i], t2[i]);
            return;
        }
    printf("Tables are identical\n");
}

/**
 * Performs square root for each elements of <tab>
 * and write them in <out>. 
*/
void racine(float *tab, float *out) {
    for (int i = 0; i < N; ++i) 
        out[i] = sqrtf(tab[i]);
}

/**
 * For each element of <f> a block performs a square root 
 * and write the result in <f_out>.
*/
__global__ void cuda_racine_block(float *f, float *f_out) {
    int tid = blockIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}

/**
 * For each element of <f> a thread performs a square root 
 * and write the result in <f_out>.
*/
__global__ void cuda_racine_thread(float *f, float *f_out) {
    int tid = threadIdx.x;
    f_out[tid] = sqrtf(f[tid]);
}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        exit(-1);
    }

    //Retrieve the table size from args
    N = atoi(argv[1]);

    if(N>1024 || N<=0){
        printf("N must belong to ]0:1024]\n");
        exit(-1);
    }

    float *rnd_floats = (float*) malloc(N*sizeof(float)); //random vector of floats
    float *sqrt_floats = (float*) malloc(N*sizeof(float)); //output vector
    float *sqrt_floats_cuda = (float*) malloc(N*sizeof(float)); //output vector for CUDA runs

    //Create table of random floats simple precision between 0 and 1
    srand((unsigned) time(NULL));
    for (int i = 0; i < N; ++i)
        rnd_floats[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

    std::cout << "********************************************************" << std::endl;
    std::cout << "                        Run on CPU                      " << std::endl;
    std::cout << "********************************************************" << std::endl;

    auto start = high_resolution_clock::now();

    racine(rnd_floats, sqrt_floats);

    auto stop = high_resolution_clock::now();
    duration<double> duration = stop - start;
    printf("\nTime to generate: %3.7f ms\n\n", duration.count() * 1000.0F);

    //write_table(sqrt_floats, "cpu.txt");

    std::cout << "********************************************************" << std::endl;
    std::cout << "                   CUDA run on N blocks                 " << std::endl;
    std::cout << "********************************************************" << std::endl;

    float *dev_rnd_floats, *dev_rnd_floats_out;
    cudaMalloc((void ** ) &dev_rnd_floats, N * sizeof(float));
    cudaMalloc((void ** ) &dev_rnd_floats_out, N * sizeof(float));
    cudaMemcpy(dev_rnd_floats, rnd_floats, N * sizeof(float), cudaMemcpyHostToDevice);

    float elapsedTime;
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord(cuda_start, 0);

    cuda_racine_block<<< N, 1>>> (dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop);
    printf("\nParallel time to generate:  %3.7f ms\n\n", elapsedTime);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    cudaFree(dev_rnd_floats_out);

    cmp_tab(sqrt_floats, sqrt_floats_cuda);
    //write_table(sqrt_floats_cuda, "blocks.txt")

    std::cout << "********************************************************" << std::endl;
    std::cout << "                  CUDA run on N threads                 " << std::endl;
    std::cout << "********************************************************" << std::endl;

    cudaMalloc((void**) &dev_rnd_floats_out, N * sizeof(float));

    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord(cuda_start, 0);

    cuda_racine_thread<<< 1, N >>>(dev_rnd_floats, dev_rnd_floats_out);
    cudaMemcpy(sqrt_floats_cuda, dev_rnd_floats_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);

    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop);
    printf("\nParallel time to generate:  %3.7f ms\n\n", elapsedTime);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    cudaFree(dev_rnd_floats);
    cudaFree(dev_rnd_floats_out);

    cmp_tab(sqrt_floats, sqrt_floats_cuda);
    //write_table(sqrt_floats_cuda, "threads.txt")

    return 0;
}