# QMCHestonGPU
Flexible framework to compute Asian and European options prices on GPU with Milstein discretisation. 

Requirements: 
- CUDA

How to use:

- Define (in main.cu) an `OptionPriceStats` Vector that will keep computation stats.
- Define (in main.cu) your options parameters in a `OptionPriceResult`
- Then compile and execute `make && ./HestonMC`
