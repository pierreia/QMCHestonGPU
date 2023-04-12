#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
//#include "dev_array.h"
#include <curand.h>
#include <random>
#include "helper_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)



#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


double normal_random()
{
    static std::default_random_engine e(1234);
    static std::normal_distribution<> normal;
    return normal(e);
}

int main() {
    try {
        // declare variables and constants
        const size_t N_PATHS = 10000;
        const size_t N_STEPS = 1000;
        const size_t N_NORMALS = N_PATHS*N_STEPS * 2;
        // const float T = 1.0f;
        // const float K = 100.0f;
        // const float B = 95.0f;
        // const float S0 = 100.0f;
        // const float sigma = 0.2f;
        // const float mu = 0.1f;
        // const float r = 0.05f;
        //float dt = float(T)/float(N_STEPS);
        //float sqrdt = sqrt(dt);

        const float kappa = 6.21;
        const float theta = 0.019;
        const float sigma = 0.61;
        const float v0 = 0.010201;
        const float T = 1;
        const float r = 0.0319;
        const float s0 = 100;
        const float K = 100;
        const float rho = -0.7;
        float dt = T/float(N_STEPS);

        // generate arrays
        //vector<float> s(N_PATHS);
        //dev_array<float> d_s(N_PATHS);
        //dev_array<float> d_normals(N_NORMALS);

        float *d_Z, *d_S;

        float *h_S;
        float *h_Z;


        //checkCudaErrors(cudaMalloc((void **)&d_Z, sizeof(float) * N_NORMALS));
        checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float) * N_PATHS));

        //h_Z = (float*)malloc(sizeof(float) * N_NORMALS);
        h_S = (float*)malloc(sizeof(float) * N_PATHS);

        double t1=double(clock())/CLOCKS_PER_SEC;
 /*        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        curandSetPseudoRandomGeneratorSeed(curandGenerator, clock());
        curandGenerateNormal(curandGenerator, d_Z, N_NORMALS, 0.0f, 1.0f);

        double t2=double(clock())/CLOCKS_PER_SEC;



        // call the kernel
        //mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, N_PATHS);
        cudaDeviceSynchronize();

        //call the kernel
        heston_euro_call(kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S, d_Z);

        // copy results from device to host
        //d_s.get(&s[0], N_PATHS);

        checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));

        // compute the payoff average
        double temp_sum=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_sum +=h_S[i];
        }
        
        double gpu_price = temp_sum/N_PATHS; */
        double gpu_price = 0.;
        double t4=double(clock())/CLOCKS_PER_SEC;

        // init variables for CPU Monte Carlo
        //vector<float> normals(N_NORMALS);
        //d_normals.get(&normals[0],N_NORMALS);

       //cudaMemcpy(h_Z, d_Z, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost);

        /* Generation with Curand State */

        curandState *devStates;
        curandStateMRG32k3a *devMRGStates;
        CUDA_CALL(cudaMalloc((void **)&devStates, N_PATHS *
                  sizeof(curandState)));

        CUDA_CALL(cudaMalloc((void **)&devMRGStates, N_PATHS *
                  sizeof(curandStateMRG32k3a)));
        

        //set value to zero
        CUDA_CALL(cudaMemset(d_S, 0, N_PATHS *
              sizeof(unsigned int)));

        const unsigned BLOCK_SIZE = 512;
        const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
        setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates);
        heston_kernel_curand<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S);
        checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
        
        // compute the payoff average
        double temp_sum2=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_sum2 +=h_S[i];
        }
        
        double gpu_price2 = temp_sum2/N_PATHS;
        double t5=double(clock())/CLOCKS_PER_SEC;

        /* END OF GENERATION WITH CURAND PSEUDORANDOM */

        /* START OF GENERATION WITH CURAND QUASIRANDOM */
        curandStateScrambledSobol32 *devSobol32States;
        curandDirectionVectors32_t *hostVectors32;
        unsigned int * hostScrambleConstants32;
        unsigned int * devDirectionVectors32;
        unsigned int * devScrambleConstants32;
        const int VECTOR_SIZE = 32; 


        /* Set results to 0 */
        checkCudaErrors(cudaMemset(d_S, 0,
                            N_PATHS * sizeof(float)));
        
        memset(h_S, 0, N_PATHS * sizeof(float));


        /* Get pointers to the 32 bit scrambled direction vectors and constants*/
        CURAND_CALL(curandGetDirectionVectors32( &hostVectors32,
                                                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));

        CURAND_CALL(curandGetScrambleConstants32( &hostScrambleConstants32));


        /* Allocate memory for 3 states per thread (x, y, z), each state to get a unique dimension */
        checkCudaErrors(cudaMalloc((void **)&devSobol32States,
                N_PATHS * 2 * sizeof(curandStateScrambledSobol32)));

        /* Allocate memory and copy 3 sets of vectors per thread to the device */

        checkCudaErrors(cudaMalloc((void **)&(devDirectionVectors32),
                            N_PATHS * 2 * VECTOR_SIZE * sizeof(int)));

        checkCudaErrors(cudaMemcpy(devDirectionVectors32, hostVectors32,
                            N_PATHS * 2 * VECTOR_SIZE * sizeof(int),
                            cudaMemcpyHostToDevice));

        /* Allocate memory and copy 3 scramble constants (one costant per dimension)
        per thread to the device */

        checkCudaErrors(cudaMalloc((void **)&(devScrambleConstants32),
                            N_PATHS * 2 * sizeof(int)));

        checkCudaErrors(cudaMemcpy(devScrambleConstants32, hostScrambleConstants32,
                            N_PATHS * 2 * sizeof(int),
                            cudaMemcpyHostToDevice)); 

        /* Initialize the states */

        setup_kernel<<<BLOCK_SIZE, GRID_SIZE>>>(devDirectionVectors32,
                                                        devScrambleConstants32,
                                                        devSobol32States);

        /* Generate and count quasi-random points  */

        
        heston_kernel_curand<<<BLOCK_SIZE, GRID_SIZE>>>(devSobol32States, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S);
        checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
        
        // compute the payoff average
        double temp_sum_qmc=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_sum_qmc +=h_S[i];
        }
        
        double gpu_price_qmc = temp_sum_qmc/N_PATHS;
        double t6=double(clock())/CLOCKS_PER_SEC;

        float sum_price = 0;

        float h_z_m = 0.0;
        int idx;

        for (int i = 0; i < N_PATHS; ++i) {
            float v = v0;
            float s = s0;
            
            idx = i*N_STEPS;

            //float v_plus, s_plus;

            for (int j = 0; j < N_STEPS; ++j) {

                //float z1 = h_Z[2*idx];
                //float z2 = h_Z[2*idx + 1];


                float z1 = normal_random();
                float z2 = normal_random();

                h_z_m += z1+z2;

                float dw1 = z1;
                float dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;

                //float v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * dw2;
                //float s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * dw1);

                float v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
                float s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

                //float s_plus = s*(1 + r*dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1 + s*0.25*dt*(dw1*dw1 - 1));

                v = max(v_plus, 0.0);
                s = max(s_plus, 0.0);

                idx++;
            }

            float payoff = max(s - K, 0.0);
            sum_price += exp(-r * T) * payoff;
            
        }

        h_z_m/= idx;

        double cpu_price = sum_price/N_PATHS;

        double t7=double(clock())/CLOCKS_PER_SEC;

        double t2 = 0;

        double t8=double(clock())/CLOCKS_PER_SEC;

        

        /* Asian Generation with Curand State */

        int m = 4;
        //curandState *devStates;
        //curandStateMRG32k3a *devMRGStates;
        //CUDA_CALL(cudaMalloc((void **)&devStates, N_PATHS *
        //          sizeof(curandState)));

        //CUDA_CALL(cudaMalloc((void **)&devMRGStates, N_PATHS *
        //          sizeof(curandStateMRG32k3a)));
        

        //set value to zero
        CUDA_CALL(cudaMemset(d_S, 0, N_PATHS *
              sizeof(unsigned int)));
        
        float *h_delta;
        float *d_delta;

        checkCudaErrors(cudaMalloc((void **)&d_delta, sizeof(float) * N_PATHS));

        h_delta = (float*)malloc(sizeof(float) * N_PATHS);
        
        setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates);
        heston_kernel_asian<<<GRID_SIZE, BLOCK_SIZE>>>(devMRGStates, kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, d_S, d_delta, m);
        
        checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
        
        checkCudaErrors(cudaMemcpy(h_delta, d_delta, sizeof(float) * N_PATHS, cudaMemcpyDeviceToHost));
        
        // compute the payoff average
        double temp_asian=0.0;
        double temp_delta=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_asian +=h_S[i];
            temp_delta +=h_delta[i];
        }
        
        double gpu_asian = temp_asian/N_PATHS;
        double asian_delta = temp_delta/N_PATHS;
        double t9=double(clock())/CLOCKS_PER_SEC;

        cout<<"****************** INFO ******************\n";
        cout<<"Number of Paths: " << N_PATHS << "\n";
        cout<<"Underlying Initial Price: " << s0 << "\n";
        cout<<"Initial Variance: " << v0 << "\n";
        cout<<"Strike: " << K << "\n";
        cout<<"Time to Maturity: " << T << " years\n";
        cout<<"Risk-free Interest Rate: " << r << "%\n";
        cout<<"Annual drift: " << theta << "%\n";
        cout<<"Volatility: " << sigma << "%\n";
        cout<<"****************** PRICE ******************\n";
        cout<<"Option Price (GPU): " << gpu_price << "\n";
        cout<<"Option Price MC (GPU): " << gpu_price2 << "\n";
        cout<<"Option Price QMC (GPU): " << gpu_price_qmc << "\n";
        cout<<"Option Price (CPU): " << cpu_price << "\n";
        cout<<"Option Price (Real): 6.8061 \n";
        cout<<"Asian Option Price (GPU): " << gpu_asian << "\n";
        cout<<"Asian Option Delta (GPU): " << asian_delta << "\n";
        cout<<"******************* TIME *****************\n";
        cout<<"Random Number Generation: " << (t2-t1)*1e3 << " ms\n";
        cout<<"GPU Monte Carlo Computation: " << (t4-t2)*1e3 << " ms\n";
        cout<<"GPU 2 Monte Carlo Computation: " << (t5-t4)*1e3 << " ms\n";
        cout<<"GPU Quasi Monte Carlo Computation: " << (t6-t5)*1e3 << " ms\n";
        cout<<"CPU Monte Carlo Computation: " << (t7-t6)*1e3 << " ms\n";
        cout<<"Speed up Factor: " << (t7-t6)/(t4-t2) << "\n";
        
        cout<<"h_Z mean:" << h_z_m << "\n";
        cout<<"******************* END *****************\n";
        // destroy generator
        //curandDestroyGenerator( curandGenerator ) ;

    /* Cleanup */

        checkCudaErrors(cudaFree(devSobol32States));
        checkCudaErrors(cudaFree(devDirectionVectors32));
        checkCudaErrors(cudaFree(devScrambleConstants32));
        checkCudaErrors(cudaFree(d_S));
        free(h_S);
    }
    catch(exception& e) {
        cout<< "exception: " << e.what() << "\n";
    }
}