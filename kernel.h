#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "product.h"


void Heston_Euler_Cuda(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N, float *d_S, float * d_N);
void heston_euro_call(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float * d_Z);

__global__ void setup_kernel(curandStateMRG32k3a *state);
__global__ void heston_kernel_curand(curandStateMRG32k3a *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, DiscretisationType mode);

__global__ void setup_kernel(unsigned int * sobolDirectionVectors, unsigned int *sobolScrambleConstants, curandStateScrambledSobol32 *state);
__global__ void heston_kernel_curand(curandStateScrambledSobol32 *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S);

__global__ void heston_kernel_asian(curandStateMRG32k3a *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float *d_delta, int m);


#endif