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



double normal_random()
{
    static std::mt19937 e(clock());
    static std::normal_distribution<> normal;
    return normal(e);
}



void MCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode){

    float *d_S, *d_Delta, *d_Rho;
    float *h_S, *h_Delta, *h_Rho;

    
    checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float) * N_PATHS));
    checkCudaErrors(cudaMalloc((void **)&d_Delta, sizeof(float) * N_PATHS));
    checkCudaErrors(cudaMalloc((void **)&d_Rho, sizeof(float) * N_PATHS));
    h_S = (float*)malloc(sizeof(float) * N_PATHS);
    h_Delta = (float*)malloc(sizeof(float) * N_PATHS);
    h_Rho = (float*)malloc(sizeof(float) * N_PATHS);
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
    heston_kernel_curand<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S,  d_Delta, d_Rho, mode);
    checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Delta, d_Delta, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Rho, d_Rho, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    

    // compute the payoff average
    float temp_price =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_price +=h_S[i];
    }
    
    *price = temp_price/N_PATHS;

    float temp_delta =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_delta +=h_Delta[i];
    }
    
    *greek_delta = temp_delta/N_PATHS;

    float temp_rho =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_rho +=h_Rho[i];
    }
    
    *greek_rho = temp_rho/N_PATHS;

    //Free Memory, Free LACRIM
    
    cudaFree(d_S);
    cudaFree(d_Delta);
    cudaFree(d_Rho);
    free(h_S);
    free(h_Delta);
    free(h_Rho);
    
}


void QMCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode){

        /* START OF GENERATION WITH CURAND QUASIRANDOM */
        curandStateScrambledSobol32 *devSobol32States;
        curandDirectionVectors32_t *hostVectors32;
        unsigned int * hostScrambleConstants32;
        unsigned int * devDirectionVectors32;
        unsigned int * devScrambleConstants32;
        const int VECTOR_SIZE = 32; 

        float *d_S, *d_Delta, *d_Rho;
        float *h_S, *h_Delta, *h_Rho;

        
        checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float) * N_PATHS));
        checkCudaErrors(cudaMalloc((void **)&d_Delta, sizeof(float) * N_PATHS));
        checkCudaErrors(cudaMalloc((void **)&d_Rho, sizeof(float) * N_PATHS));
        h_S = (float*)malloc(sizeof(float) * N_PATHS);
        h_Delta = (float*)malloc(sizeof(float) * N_PATHS);
        h_Rho = (float*)malloc(sizeof(float) * N_PATHS);
        

        /* Get pointers to the 32 bit scrambled direction vectors and constants*/
        checkCudaErrors(curandGetDirectionVectors32( &hostVectors32,
                                                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));

        checkCudaErrors(curandGetScrambleConstants32( &hostScrambleConstants32));


        /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
        checkCudaErrors(cudaMalloc((void **)&devSobol32States,
                N_PATHS * 2 * sizeof(curandStateScrambledSobol32)));

        /* Allocate memory and copy 3 sets of vectors per thread to the device */

        checkCudaErrors(cudaMalloc((void **)&(devDirectionVectors32),
                            N_PATHS * 2 * VECTOR_SIZE * sizeof(long long int)));

        checkCudaErrors(cudaMemcpy(devDirectionVectors32, hostVectors32,
                            N_PATHS * 2 * VECTOR_SIZE * sizeof(long long int),
                            cudaMemcpyHostToDevice));

        /* Allocate memory and copy 3 scramble constants (one costant per dimension)
        per thread to the device */

        checkCudaErrors(cudaMalloc((void **)&(devScrambleConstants32),
                            N_PATHS * 2 * sizeof(long long int)));

        checkCudaErrors(cudaMemcpy(devScrambleConstants32, hostScrambleConstants32,
                            N_PATHS * 2 * sizeof(long long int),
                            cudaMemcpyHostToDevice)); 

        /* Initialize the states */

        const unsigned BLOCK_SIZE = 512;
        const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));

        setup_kernel<<<BLOCK_SIZE, GRID_SIZE>>>(devDirectionVectors32,
                                                        devScrambleConstants32,
                                                        devSobol32States);

        /* Generate and count quasi-random points  */

        
        heston_kernel_curand<<<BLOCK_SIZE, GRID_SIZE>>>(devSobol32States, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S, d_Delta, d_Rho, mode);
        checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    
    

    // compute the payoff average
    float temp_price =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_price +=h_S[i];
    }
    
    *price = temp_price/N_PATHS;

    float temp_delta =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_delta +=h_Delta[i];
    }
    
    *greek_delta = temp_delta/N_PATHS;

    float temp_rho =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_rho +=h_Rho[i];
    }
    
    *greek_rho = temp_rho/N_PATHS;

    //Free Memory, Free LACRIM
    
    cudaFree(d_S);
    cudaFree(d_Delta);
    cudaFree(d_Rho);
    free(h_S);
    free(h_Delta);
    free(h_Rho);
    
}

void MCAsian(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, int m, float * price, float * greek_delta, float * greek_rho,  DiscretisationType mode){

    float *d_S, *d_Delta, *d_Rho;
    float *h_S, *h_Delta, *h_Rho;

    
    checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float) * N_PATHS));
    checkCudaErrors(cudaMalloc((void **)&d_Delta, sizeof(float) * N_PATHS));
    checkCudaErrors(cudaMalloc((void **)&d_Rho, sizeof(float) * N_PATHS));
    h_S = (float*)malloc(sizeof(float) * N_PATHS);
    h_Delta = (float*)malloc(sizeof(float) * N_PATHS);
    h_Rho = (float*)malloc(sizeof(float) * N_PATHS);
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
    heston_kernel_asian<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, m, d_S,  d_Delta, d_Rho, mode);
    checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Delta, d_Delta, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Rho, d_Rho, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
    

    // compute the payoff average
    float temp_price =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_price +=h_S[i];
    }
    
    *price = temp_price/N_PATHS;

    float temp_delta =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_delta +=h_Delta[i];
    }
    
    *greek_delta = temp_delta/N_PATHS;

    float temp_rho =0.0;
    for(size_t i=0; i<N_PATHS; i++) {
        temp_rho +=h_Rho[i];
    }
    
    *greek_rho = temp_rho/N_PATHS;

    //Free Memory, Free LACRIM
    
    cudaFree(d_S);
    cudaFree(d_Delta);
    cudaFree(d_Rho);
    free(h_S);
    free(h_Delta);
    free(h_Rho);
    
}

void calculateOptionPrice(OptionPriceResult& result) {
    // Declare variables and constants
    int N_PATHS = result.N_PATHS;
    int N_STEPS = result.N_STEPS;

    float dt = result.T / float(N_STEPS);

    float gpu_price;

    float greek_delta = 0;
    float greek_rho = 0;

    double t1 = double(clock()) / CLOCKS_PER_SEC;


    if (result.random == PSEUDO) {

        if (result.type == EURO) {
            MCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
            result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }
        else {
        // Call the MCEuro function
            MCAsian(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
                result.s0, result.K, result.rho, N_STEPS, N_PATHS, 4, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }

    } else {

        if (result.type == EURO) {
            QMCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
            result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }
        else {
        // Call the MCEuro function
            QMCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
                result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }

    }

    double t2 = double(clock()) / CLOCKS_PER_SEC;
    // Calculate execution time
    result.execution_time = t2 - t1;

    // Store the output result in the struct
    result.price = gpu_price;
    result.greek_delta = greek_delta;
    result.greek_rho = greek_rho;
}

void MCEuroCPU(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode){
    float MC_price = 0.;
    float MC_rho = 0.;
    float MC_delta = 0.;


    // Setup Boost random number generator
    //boost::mt19937 rng;
    //boost::normal_distribution<> dist(0, 1);
    //boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > normal_random(rng, dist);
    for (int i = 0; i < N_PATHS; ++i) {

        float v = v0;
        float s = s0;
        float dt = T/N_STEPS;

        float s_plus, v_plus;
        float payoff, greek_rho, delta;

        for (int j = 0; j < N_STEPS; ++j) {

            float z1 = normal_random();
            float z2 = normal_random();


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

        MC_delta += exp(-r * T) * ((s>K) ? 1: 0) * s/s0;
        MC_rho += exp(-r * T) * ((s>K) ? 1: 0) * K*T;

        MC_price += exp(-r * T) * payoff;
        
    }


    *price = MC_price/N_PATHS;
    *greek_rho = MC_rho/N_PATHS;
    *greek_delta = MC_delta/N_PATHS;

}


void MCAsianCPU(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, int m, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode){
    float MC_price = 0.;
    float MC_rho = 0.;
    float MC_delta = 0.;

        // Setup Boost random number generator
    //boost::mt19937 rng(clock());
    //boost::normal_distribution<> dist(0, 1);
    //boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > normal_random(rng, dist);

    
    float ts[4]{31, 63, 95, 127};
    


    for (int i = 0; i < N_PATHS; ++i) {

        float v = v0;
        float s = s0;
        float dt = T/N_STEPS;
        float s_mean = 0.;
        int k = 0.;

        float s_plus, v_plus;
        float payoff, tmp_rho, delta;
        float greek_rho;

        for (int j = 0; j < N_STEPS; ++j) {

            float z1 = normal_random();
            float z2 = normal_random();


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


        // for (int k=0; k<4; k++){
        //     if (j==ts[k]){
        //         s_mean+=s;
        //         tmp_rho += s/(k+1);
        //     }
        // }

                        
        if ((j+1)%(N_STEPS/4) == 0){
            s_mean += s;
            greek_rho += s/(k+1);
            k++;
        }
                
            
        }

        s_mean/=m;
        
        payoff = max(s_mean - K, 0.0);
        MC_price += exp(-r * T) * payoff;
        MC_delta += exp(-r * T) * s_mean/s0 * (s_mean > K ? 1 :0);

        greek_rho/= m;
        greek_rho -= T*(s_mean-K);
        greek_rho *= exp(-r * T)* ((s_mean > K ? 1 :0));
        MC_rho += greek_rho;
        
    }


    *price = MC_price/N_PATHS;
    *greek_rho = MC_rho/N_PATHS;
    *greek_delta = MC_delta/N_PATHS;

}


void calculateOptionPriceCPU(OptionPriceResult& result) {
    // Declare variables and constants
    int N_PATHS = result.N_PATHS;
    int N_STEPS = result.N_STEPS;

    float dt = result.T / float(N_STEPS);

    float gpu_price;

    float greek_delta = 0;
    float greek_rho = 0;

    double t1 = double(clock()) / CLOCKS_PER_SEC;


    if (result.random == PSEUDO) {

        if (result.type == EURO) {
            MCEuroCPU(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
            result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }
        else {
        // Call the MCEuro function
            MCAsianCPU(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
                result.s0, result.K, result.rho, N_STEPS, N_PATHS, 4, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }

    } else {

        if (result.type == EURO) {
            QMCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
            result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }
        else {
        // Call the MCEuro function
            QMCEuro(result.kappa, result.theta, result.sigma, result.v0, result.T, result.r,
                result.s0, result.K, result.rho, N_STEPS, N_PATHS, &gpu_price, &greek_delta, &greek_rho, result.discretisation);
        }

    }

    double t2 = double(clock()) / CLOCKS_PER_SEC;
    // Calculate execution time
    result.execution_time = t2 - t1;

    // Store the output result in the struct
    result.price = gpu_price;
    result.greek_delta = greek_delta;
    result.greek_rho = greek_rho;
}

void calculateOptionStats(OptionPriceResult result, OptionPriceStats& stats){

    float prices[stats.N_runs];
    float deltas[stats.N_runs];
    float rhos[stats.N_runs];

    float m_price;
    float m_delta;
    float m_rho;

    float m_time;


    for (int i= 0; i < stats.N_runs; i++){
       if (stats.device == CPU) {

            calculateOptionPriceCPU(result);
            prices[i] = result.price;
            deltas[i] = result.greek_delta;
            rhos[i] = result.greek_rho;
            m_time += result.execution_time;

       } else {
        
            calculateOptionPrice(result);
            prices[i] = result.price;
            deltas[i] = result.greek_delta;
            rhos[i] = result.greek_rho;
            m_time += result.execution_time;

       }
    }

    // calculate the mean of the arrays
    m_price = 0;
    m_delta = 0;
    m_rho = 0;
    for (int i = 0; i < stats.N_runs; i++) {
        m_price += prices[i];
        m_delta += deltas[i];
        m_rho += rhos[i];
    }
    m_price /= stats.N_runs;
    m_delta /= stats.N_runs;
    m_rho /= stats.N_runs;

    // calculate the standard deviation of the arrays
    float s_price = 0;
    float s_delta = 0;
    float s_rho = 0;
    for (int i = 0; i < stats.N_runs; i++) {
        s_price += pow(prices[i] - m_price, 2);
        s_delta += pow(deltas[i] - m_delta, 2);
        s_rho += pow(rhos[i] - m_rho, 2);
    }
    s_price = sqrt(s_price / (stats.N_runs));
    s_delta = sqrt(s_delta / (stats.N_runs));
    s_rho = sqrt(s_rho / (stats.N_runs));

    // update OptionPriceStats struct with the calculated values
    stats.price = m_price;
    stats.std_dev_price = s_price;
    stats.greek_delta = m_delta;
    stats.std_dev_delta = s_delta;
    stats.greek_rho = m_rho;
    stats.std_dev_rho = s_rho;
    stats.execution_time = m_time/stats.N_runs;

}