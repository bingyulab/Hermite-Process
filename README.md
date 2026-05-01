# Hermite-Process

This repository contains code for simulating the Hermite process, a non-Gaussian self-similar process with stationary increments. The main file is `simulation.py`, which implements the simulation algorithm and includes tests to verify the properties of the simulated process.

## Simulation Algorithm
The Hermite process can be simulated using a method based on the Fast Fourier Transform (FFT). The key steps of the algorithm are as follows:
1. Generate a grid of points and compute the covariance structure of the process.
2. Use the FFT to efficiently generate samples of the process at the specified time points. 
3. Verify the properties of the simulated process, such as variance, mean, and non-Gaussianity.

## Tests
The `simulation.py` file includes several tests to validate the properties of the simulated Hermite process:
- **Covariance Structure**: Checks that the empirical covariance matches the theoretical covariance.
- **Non-Gaussianity**: Verifies that the distribution of the simulated process is non-Gaussian, as expected for the Rosenblatt distribution.
- **Density Smoothness**: Tests that the density of the simulated process is smooth, consistent with theoretical results.

## Usage
To run the simulations and tests, simply execute the `simulation.py` file. The script will generate samples of the Hermite process, perform the tests, and display the results. You can modify the parameters such as the number of samples, time points, and grid size to explore different aspects of
the process.   

```python
pip install -r requirements.txt
python simulation.py
```

### Reference
https://github.com/markveillette/rosenblatt