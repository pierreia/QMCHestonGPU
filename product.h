#ifndef _PRODUCT_H_
#define _PRODUCT_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

enum OptionType { ASIAN, EURO };
enum DiscretisationType {EULER, MILSTEIN};
enum RandomType {PSEUDO, QUASI};
enum DeviceType {CPU, GPU};

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
    float greek_delta;
    float greek_rho;
    double execution_time;
    
};

struct OptionPriceStats {
    int N_runs;
    float price;
    float std_dev_price;
    float greek_delta;
    float std_dev_delta;
    float greek_rho;
    float std_dev_rho;
    double execution_time;
    DeviceType device;
};

void MCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode);
void QMCEuro(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, float * price,float * greek_delta, float * greek_rho, DiscretisationType mode);
void MCAsian(float kappa, float theta, float sigma, float v0, float T, float r, float s0, float K, float rho, int  N_STEPS, int N_PATHS, int m, float * price, float * greek_delta, float * greek_rho, DiscretisationType mode);
void calculateOptionPrice(OptionPriceResult& result);
void calculateOptionPriceCPU(OptionPriceResult& result);
void calculateOptionStats(OptionPriceResult result, OptionPriceStats& stats);

#endif