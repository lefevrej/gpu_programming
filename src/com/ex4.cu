#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>

int N;

using namespace std::chrono;

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
 * Performs vector addition between <tab1> and <tab2> and store 
 * result in <out>
 */
void add(float* tab1, float* tab2, float* out){
    for (int i =0; i < N; ++i)
        out[i]=tab1[i]+tab2[i];
}

/** 
 * Each block performs the addition of two elements of <f1> 
 * and <f2> and store result in <f_out>
 */
__global__ void cuda_add_block(float *f1, float *f2, float* f_out){
    int tid = blockIdx.x;
    f_out[tid] = f1[tid]+f2[tid];
}

/** 
 * Each thread performs the addition of two elements of <f1> 
 * and <f2> and store result in <f_out>
*/
__global__ void cuda_add_thread(float *f1, float *f2, float* f_out){
    int tid = threadIdx.x;
    f_out[tid] = f1[tid]+f2[tid];
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

    float *rnd_floats1 = (float*) malloc(N*sizeof(float)); //first random vector
    float *rnd_floats2 = (float*) malloc(N*sizeof(float)); //second random vector
    float *sum = (float*) malloc(N*sizeof(float)); //add output for CPU
    float *sum_cuda= (float*) malloc(N*sizeof(float)); //add output for GPU
        
    //Create table of random floats simple precision between 0 and 1
    srand((unsigned)time(NULL));
    for (int i =0; i < N; ++i){
        rnd_floats1[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        rnd_floats2[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    std::cout << "********************************************************" << std::endl;
    std::cout << "                        Run on CPU                      " << std::endl;
    std::cout << "********************************************************" << std::endl;

    auto start = high_resolution_clock::now(); 

    add(rnd_floats1, rnd_floats2, sum);
    
    auto stop = high_resolution_clock::now();
    duration<double> duration = stop - start;
    printf("\nTime to generate: %3.7f ms\n\n", duration.count() * 1000.0F);

    std::cout << "********************************************************" << std::endl;
    std::cout << "                   CUDA run on N blocks                 " << std::endl;
    std::cout << "********************************************************" << std::endl;

    float *dev_rnd_floats1, *dev_rnd_floats2, *dev_rnd_floats_out;
    cudaMalloc((void**)&dev_rnd_floats1, N * sizeof(float));
    cudaMalloc((void**)&dev_rnd_floats2, N * sizeof(float));
    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));
    cudaMemcpy(dev_rnd_floats1, rnd_floats1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rnd_floats2, rnd_floats2, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_add_block<<<N,1>>>(dev_rnd_floats1, dev_rnd_floats2, dev_rnd_floats_out);
    cudaMemcpy(sum_cuda, dev_rnd_floats_out, N * sizeof(float),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf("\nParallel time to generate: %3.7f ms\n\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats_out );

    cmp_tab(sum, sum_cuda);//compare results between CPU and GPU

    std::cout << "********************************************************" << std::endl;
    std::cout << "                  CUDA run on N threads                 " << std::endl;
    std::cout << "********************************************************" << std::endl;

    cudaMalloc((void**)&dev_rnd_floats_out, N * sizeof(float));

    cudaEventCreate( &cuda_start);
    cudaEventCreate( &cuda_stop);
    cudaEventRecord( cuda_start, 0 );

    cuda_add_thread<<<1,N>>>(dev_rnd_floats1, dev_rnd_floats2, dev_rnd_floats_out);
    cudaMemcpy(sum_cuda, dev_rnd_floats_out, N * sizeof(float),cudaMemcpyDeviceToHost);

    cudaEventRecord( cuda_stop, 0 );
    cudaEventSynchronize( cuda_stop );
    
    cudaEventElapsedTime( &elapsedTime, cuda_start, cuda_stop );
    printf("\nParallel time to generate:  %3.7f ms\n\n", elapsedTime);
    cudaEventDestroy( cuda_start );
    cudaEventDestroy( cuda_stop );

    cudaFree( dev_rnd_floats1 );
    cudaFree( dev_rnd_floats2 );
    cudaFree( dev_rnd_floats_out );

    cmp_tab(sum, sum_cuda);//compare results between CPU and GPU

    return 0;
}
