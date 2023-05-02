#include "kernel.h"
#include "product.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandStateMRG32k3a *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(clock() + id/1000, id, 0, &state[id]);
}


/* This kernel initializes state per thread for each of x, y, and z */

__global__ void setup_kernel(unsigned int * sobolDirectionVectors,
                             unsigned int *sobolScrambleConstants,
                             curandStateScrambledSobol32 *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int dim = 2*id;
    int const VECTOR_SIZE = 32;
    unsigned int offset = id * ((10000 / blockDim.x) + 1);
    /* Each thread uses 3 different dimensions */
    curand_init(sobolDirectionVectors + VECTOR_SIZE*dim,
                sobolScrambleConstants[dim],
                offset,
                &state[dim]);

    curand_init(sobolDirectionVectors + VECTOR_SIZE*(dim + 1),
                sobolScrambleConstants[dim + 1],
                offset,
                &state[dim + 1]);

}

__global__ void heston_kernel_curand(curandStateMRG32k3a *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float *d_Delta, float *d_Rho, DiscretisationType mode)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    //unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random normals */

    float k_payoff = 0.;

    float v = v0;
    float s = s0;

    float dt = T/N_timesteps;

    float v_plus, s_plus;

    
    if (id < N_paths) {

            for (int j = 0; j < N_timesteps; ++j) {
                x = curand_normal2(&localState);
                float z1 = x.x;
                float z2 = x.y;

                float dw1 = z1;
                float dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;

                if (mode==EULER) {

                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2; // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    s = s_plus;
                    v = v_plus;

                } else {
                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    v = max(v_plus, 0.0);
                    s = max(s_plus, 0.0);
                
                }
                
            
            }
        float payoff = max(s - K, 0.0);
        d_S[id] = exp(-r * T) * payoff;
        d_Delta[id] = exp(-r * T) * ((s>K) ? 1: 0) * s/s0;
        d_Rho[id] = exp(-r * T) * ((s>K) ? 1: 0) * K*T;
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
}


__global__ void heston_kernel_curand(curandStateScrambledSobol32 *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float *d_Delta, float *d_Rho, DiscretisationType mode)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int baseDim = 2 * id;
    //unsigned int count = 0;
    //float2 x;
    /* Copy state to local memory for efficiency */
    //curandStateScrambledSobol32 localState = state[id];
    /* Generate pseudo-random normals */

    

    float v = v0;
    float s = s0;
    

    float dt = T/N_timesteps;

    float v_plus, s_plus;
    
    float z1, z2;
    float payoff;
    
    if (id < N_paths) {
            for (int j = 0; j < N_timesteps; ++j) {
                z1 = curand_normal(&state[baseDim]);
                z2 = curand_normal(&state[baseDim + 1]);
                
                float dw1 = z1;
                float dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;

                if (mode==EULER) {

                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2; // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    s = s_plus;
                    v = v_plus;

                } else {
                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    v = max(v_plus, 0.0);
                    s = max(s_plus, 0.0);
                
                }
                
            
            }
        payoff = max(s - K, 0.0);
        d_S[id] = exp(-r * T) * payoff;
        d_Delta[id] = exp(-r * T) * ((s>K) ? 1: 0) * s/s0;
        d_Rho[id] = exp(-r * T) * ((s>K) ? 1: 0) * K*T;
    }
    /* Copy state back to global memory */
    
    /* Store results */
}

__global__ void heston_kernel(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float * d_Z){
        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned bsz = blockDim.x;
        int s_idx = tid + bid * bsz;
        int n_idx = tid + bid * bsz;

        float k_payoff = 0.;

        float v = v0;
        float s = s0;

        float dt = T/N_timesteps;

        float v_plus, s_plus;

        
        if (s_idx < N_paths) {

                for (int j = 0; j < N_timesteps; ++j) {
                    float z1 = d_Z[2*n_idx];
                    float z2 = d_Z[2*n_idx+1];

                    float dw1 = z1;
                    float dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;

                    //float v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * dw2; //Euler
                    //float s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * dw1);


                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    //float s_plus = s*(1 + r*dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1 + s*0.25*dt*(dw1*dw1 - 1));

                    v = max(v_plus, 0.0);
                    s = max(s_plus, 0.0);

                    n_idx ++;
                
                }
            float payoff = max(s - K, 0.0);
            d_S[s_idx] = exp(-r * T) * payoff;
        }
}

void heston_euro_call(
        float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, float *d_S, float * d_Z) {
        const unsigned BLOCK_SIZE = 512;
        const unsigned GRID_SIZE = ceil(float(N_paths) / float(BLOCK_SIZE));
        heston_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        kappa, theta, sigma, v0, T, r, s0, K, rho, N_timesteps, N_paths, d_S, d_Z);
    }



__global__ void heston_kernel_asian(curandStateMRG32k3a *state, float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int N_timesteps, int N_paths, int m, float *d_S, float *d_Delta, float *d_Rho, DiscretisationType mode)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    //unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random normals */

    float ts[4]{31, 63, 95, 127};

    float v = v0;
    float s = s0;
    float s_mean = 0.;
    float delta = 0.;
    float greek_rho = 0.;
    float tmp_rho = 0.;

    float dt = T/N_timesteps;

    float v_plus, s_plus;
    int k = 0;
    
    
    
    if (id < N_paths) {

            for (int j = 0; j < N_timesteps; ++j) {
                x = curand_normal2(&localState);
                float z1 = x.x;
                float z2 = x.y;

                float dw1 = z1;
                float dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;
                if (mode==EULER) {

                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2; // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    s = s_plus;
                    v = v_plus;

                } else {
                    v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
                    s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                    v = max(v_plus, 0.0);
                    s = max(s_plus, 0.0);
                
                }

                
                if ((j+1)%(N_timesteps/4) == 0){
                    s_mean += s;
                    greek_rho += s/(k+1);
                    k++;
                }
                

                //float ts[4]{31, 63, 95, 127};
                //float ts[4]{7, 15, 23, 31};
                //float ts[4]{15, 31, 47, 63};
                //float ts[4]{255, 511, 767, 1023};

                // for (int k=0; k<4; k++){
                //     if (j==ts[k]){
                //         s_mean+=s;
                //         tmp_rho += s/(k+1);
                //     }
                // }
            
            }
        s_mean/=m;
        
        float payoff = max(s_mean - K, 0.0);

        d_S[id] = exp(-r * T) * payoff;

        if (s_mean > K) {
            delta = exp(-r * T) * s_mean/s0;
        }
        d_Delta[id] = delta;

        tmp_rho/= m;
        tmp_rho -= T*(s_mean-K);
        tmp_rho *= exp(-r * T)* ((s_mean > K ? 1 :0));

        greek_rho/= m;
        greek_rho -= T*(s_mean-K);
        greek_rho *= exp(-r * T)* ((s_mean > K ? 1 :0));

        d_Rho[id] = greek_rho;

        
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
}