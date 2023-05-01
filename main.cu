#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <iomanip> 
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
//#include "dev_array.h"
#include <curand.h>
#include <random>
#include "helper_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "product.h"

#include <vector>
#include <string>

using namespace std;


void printResultsLatex(const std::vector<OptionPriceResult>& params, OptionPriceStats& results) {
    // Print header
    std::cout << "\\begin{tabular}{|c|c|c|c|c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "N Timesteps & Results & $Price$ & $Delta$ & $Rho$ & Time (ms) \\\\" << std::endl;
    std::cout << "\\hline" << std::endl;

    // Print results
    for (size_t i = 0; i < params.size(); ++i) {
        calculateOptionStats(params[i], results);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << params[i].N_PATHS << "x" << params[i].N_STEPS << " ";
        std::cout << "& " << results.price << " (" << results.std_dev_price << ")";
        //std::cout << " & " << results[i].std_dev_price;
        std::cout << " & " << results.greek_delta << " (" << results.std_dev_delta << ")";
        //std::cout << " & " << results[i].std_dev_delta;
        std::cout << " & " << results.greek_rho << " (" << results.std_dev_rho << ")";
        //std::cout << " & " << results[i].std_dev_rho;
        std::cout << " & " << results.execution_time * 1e3 ;
        std::cout << " \\\\" << std::endl;
        std::cout << "\\hline" << std::endl;
    }

    // Print footer
    std::cout << "\\end{tabular}" << std::endl;
}

int main() {
    try {
        // declare variables and constants
        int N_PATHS = 10000;
        int N_STEPS = 1000;
        

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


        float gpu_price;
        
        double t1=double(clock())/CLOCKS_PER_SEC;
        //MCEuro(kappa, theta, sigma, v0, T, r, s0, K, rho, N_STEPS, N_PATHS, &gpu_price);

        OptionPriceResult EuroMCMilstein = {
            EURO,
            MILSTEIN,
            PSEUDO,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPrice(EuroMCMilstein);

        OptionPriceResult EuroMCEuler = {
            EURO,
            EULER,
            PSEUDO,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPrice(EuroMCEuler);


        OptionPriceResult EuroQMCEuler = {
            EURO,
            EULER,
            QUASI,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPrice(EuroQMCEuler);


        OptionPriceResult AsianMCEuler = {
            ASIAN,
            EULER,
            PSEUDO,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPrice(AsianMCEuler);

        OptionPriceResult EuroMCEulerCPU = {
            EURO,
            EULER,
            PSEUDO,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPriceCPU(EuroMCEulerCPU);


        OptionPriceResult AsianMCEulerCPU = {
            ASIAN,
            EULER,
            PSEUDO,
            kappa,
            theta,
            sigma,
            v0,
            T,
            r,
            s0,
            K,
            rho,
            N_STEPS,
            N_PATHS,
            0.,
            0.,
            0., // price will be set by calculateOptionPrice
            0.0 // execution_time will be set by calculateOptionPrice
        };

        calculateOptionPriceCPU(AsianMCEulerCPU);


    OptionPriceStats Results {5,0,0,0,0,0,0,0, CPU};
    calculateOptionStats(AsianMCEulerCPU, Results);


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
        cout<<"Option Price - EULER (GPU): " << EuroMCEuler.price << "\n";
        cout<<"Option Price - MILSTEIN (GPU): " << EuroMCMilstein.price << "\n";
        cout<<"Option Price QMC (GPU): " << EuroQMCEuler.price << "\n";
        cout<<"European Option Delta (GPU): " << EuroMCEuler.greek_delta << "\n";
        cout<<"European Option Rho (GPU): " << EuroMCEuler.greek_rho << "\n";
        cout<<"Option Price (CPU): " << EuroMCEulerCPU.price << "\n";
        cout<<"European Option Delta (GPU): " << EuroMCEulerCPU.greek_delta << "\n";
        cout<<"European Option Rho (GPU): " << EuroMCEulerCPU.greek_rho << "\n";
        cout<<"Option Price (Real): 6.8061 \n";
        cout<<"Asian Option Price (GPU): " << AsianMCEuler.price << "\n";
        cout<<"Asian Option Delta (GPU): " << AsianMCEuler.greek_delta << "\n";
        cout<<"Asian Option Rho (GPU): " << AsianMCEuler.greek_rho << "\n";

        cout<<"Asian Option Price (CPU): " << Results.price << "\n";
        cout<<"Asian Option Delta (CPU): " << Results.greek_delta << "\n";
        cout<<"Asian Option Rho (CPU): " << Results.greek_rho << "\n";

        cout<<"Asian Option Price R (CPU): " << Results.std_dev_price << "\n";
        cout<<"Asian Option Delta R (CPU): " << Results.std_dev_delta << "\n";
        cout<<"Asian Option Rho R (CPU): " << Results.std_dev_rho << "\n";

        cout<<"******************* TIME *****************\n";
        cout<<"GPU Monte Carlo EULER : " << EuroMCEuler.execution_time*1e3 << " ms\n";
        cout<<"GPU Monte Carlo Milstein " << EuroMCMilstein.execution_time*1e3 << " ms\n";
        cout<<"CPU Monte Carlo Computation: " << EuroMCEulerCPU.execution_time*1e3 << " ms\n";
        cout<<"CPU Monte Carlo Computation R : " << Results.execution_time*1e3 << " ms\n";
        cout<<"Speed up Factor: " << EuroMCEulerCPU.price/EuroMCEuler.execution_time << "\n";
        
        cout<<"******************* END *****************\n";
        // destroy generator
        //curandDestroyGenerator( curandGenerator ) ;

    /* Cleanup */
    OptionPriceStats TestStats {10,0,0,0,0,0,0,0, GPU};
    std::vector<OptionPriceResult> TestVector = {
        {EURO, MILSTEIN, PSEUDO, kappa, theta, sigma, v0, T, r, s0, K, rho, 100, N_PATHS, 0., 0., 0., 0.0},
        {EURO, MILSTEIN, PSEUDO, kappa, theta, sigma, v0, T, r, s0, K, rho, 500, N_PATHS, 0., 0., 0., 0.0},
        {EURO, MILSTEIN, PSEUDO, kappa, theta, sigma, v0, T, r, s0, K, rho, 1000, N_PATHS, 0., 0., 0., 0.0}, 
        {EURO, MILSTEIN, PSEUDO, kappa, theta, sigma, v0, T, r, s0, K, rho, 2000, N_PATHS, 0., 0., 0., 0.0}, 
        };
    std::vector<OptionPriceStats> AsianResults = {Results};

    printResultsLatex(TestVector, TestStats);
    }
    catch(exception& e) {
        cout<< "exception: " << e.what() << "\n";
    }
}