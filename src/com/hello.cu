#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    /*As we have 2 blocks and 5 thread per blocks
    this program will print the message 2*5=10 times*/
    cuda_hello<<<2,5>>>();
    return 0;
}