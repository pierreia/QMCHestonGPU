#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

double normal_random()
{
    static std::default_random_engine e{static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    static std::normal_distribution<> normal;
    return normal(e);
}

double heston_price_FT_Euler(double kappa, double theta, double sigma, double v0, double T, double r, double s0, double K, double rho, int N_timesteps, int N_paths)
{
    double dt = T / N_timesteps;

    double sum_price = 0;

    for (int i = 0; i < N_paths; ++i) {
        double v = v0;
        double s = s0;

        for (int j = 0; j < N_timesteps; ++j) {
            double z1 = normal_random();
            double z2 = normal_random();

            double dw1 = z1;
            double dw2 = rho * z1 + sqrt(1 - rho * rho) * z2;

            //double v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * dw2;
            //double s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * dw1);


            double v_plus = v + kappa * (theta - max(v, 0.0)) * dt + sigma * sqrt(max(v, 0.0)) * sqrt(dt) * dw2 + 0.25*sigma*sigma*dt*(dw2*dw2 - 1); // Milstein
            double s_plus = s * exp((r - 0.5 * max(v, 0.0)) * dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1);

            //double s_plus = s*(1 + r*dt + sqrt(max(v, 0.0)) * sqrt(dt)* dw1 + s*0.25*dt*(dw1*dw1 - 1));

            v = max(v_plus, 0.0);
            s = max(s_plus, 0.0);
        }

        double payoff = max(s - K, 0.0);
        sum_price += exp(-r * T) * payoff;
    }

    return sum_price / N_paths;
}

int main()
{
    double kappa = 6.21;
    double theta = 0.019;
    double sigma = 0.61;
    double v0 = 0.010201;
    double T = 1;
    double r = 0.0319;
    double s0 = 100;
    double K = 100;
    double rho = -0.7;

    int N_timesteps = 100;
    int N = 1000000;


    double price = heston_price_FT_Euler(kappa, theta, sigma, v0, T, r, s0, K, rho, N_timesteps, N);

    cout << "Option price: " << price << endl;

    return 0;
}
