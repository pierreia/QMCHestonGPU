#ifndef _PRODUCT_H_
#define _PRODUCT_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

enum OptionType { ASIAN, EURO };
enum DiscretisationType {EULER, MILSTEIN};
enum RandomType {PSEUDO, QUASI};

struct OptionPriceResult {
    OptionType type;
    DiscretisationType discretisation; 
    RandomType random;
    float kappa;
    float theta;
    float sigma;
    float v0;
    float T;
    float r;
    float s0;
    float K;
    float rho;
    int N_STEPS;
    int N_PATHS;
    float price;
    float delta;
    float rho;
    double execution_time;
    
};

void MCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, DiscretisationType mode);
void calculateOptionPrice(OptionPriceResult& result);

#endif