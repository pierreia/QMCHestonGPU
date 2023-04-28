#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "product.h"
#include "kernel.h"

#include <curand.h>
#include <random>
#include "helper_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>


void MCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, DiscretisationType mode){

    float *d_S;
    float *h_S;

    
    checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float) * N_PATHS));
    h_S = (float*)malloc(sizeof(float) * N_PATHS);
    /* Generation with Curand State */

    curandState *devStates;
    curandStateMRG32k3a *devMRGStates;
    checkCudaErrors(cudaMalloc((void **)&devStates, N_PATHS *
                sizeof(curandState)));

    checkCudaErrors(cudaMalloc((void **)&devMRGStates, N_PATHS *
                sizeof(curandStateMRG32k3a)));
    

    //set value to zero
    checkCudaErrors(cudaMemset(d_S, 0, N_PATHS *
            sizeof(unsigned int)));

    const unsigned BLOCK_SIZE = 512;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates);
    heston_kernel_curand<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S, mode);
    checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    

    // compute the payoff average
    float temp_price =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_price +=h_S[i];
    }
    
    *price = temp_price/N_PATHS;

    //Set values to zero
    
    cudaFree(d_S);
    free(h_S);
    
}


void calculateOptionPrice(OptionPriceResult& result) {
    // Declare variables and constants
    int N_PATHS = result.N_PATHS;
    int N_STEPS = result.N_STEPS;

    float dt = result.T / float(N_STEPS);

    float gpu_price;

    double t1 = double(clock()) / CLOCKS_PER_SEC;

    if (result.type == EURO) {
        MCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
           result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, result.discretisation);
    }
    else {
    // Call the MCEuro function
        MCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
            result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, result.discretisation);
    }

    double t2 = double(clock()) / CLOCKS_PER_SEC;
    // Calculate execution time
    result.execution_time = t2 - t1;

    // Store the output result in the struct
    result.price = gpu_price;
}