# Stochastic Volatility Jump-Diffusion (SVJD) Model

A comprehensive Python implementation of the Stochastic Volatility Jump-Diffusion model for option pricing and volatility forecasting. This model combines the Heston stochastic volatility model with jump processes to capture both continuous and discontinuous price movements.

## Table of Contents
- [Mathematical Framework](#mathematical-framework)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Model Components](#model-components)
- [Implementation Details](#implementation-details)
- [Model Calibration](#model-calibration)
- [Performance Considerations](#performance-considerations)
- [Model Limitations](#model-limitations)
- [Future Extensions](#future-extensions)
- [Contributing](#contributing)
- [License](#license)

## Mathematical Framework

The SVJD model is described by the following stochastic differential equations:

```math
dS_t = (r - \lambda \mu_J)S_t dt + \sqrt{v_t}S_t dW^S_t + S_t dJ_t
```
```math
dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t}dW^v_t
```

where:
- $S_t$ is the asset price
- $v_t$ is the variance process
- $r$ is the risk-free rate
- $\kappa$ is the mean reversion speed of variance
- $\theta$ is the long-term variance
- $\sigma$ is the volatility of variance
- $\rho$ is the correlation between $dW^S_t$ and $dW^v_t$
- $J_t$ is a compound Poisson process with intensity $\lambda$
- Jump sizes follow a log-normal distribution: $\log(1 + J) \sim N(\mu_J, \sigma_J^2)$

### Discretization Scheme

The variance process is discretized using the Quadratic-Exponential (QE) scheme, which ensures positivity and stability:

1. For small variance values (ψ ≤ 1.5):
   ```math
   v_{t+dt} = a(b\sqrt{dt}Z + \sqrt{b})^2
   ```

2. For large variance values (ψ > 1.5):
   ```math
   v_{t+dt} = \begin{cases}
   0 & \text{if } U \leq p \\
   \frac{1}{\beta}\log(\frac{1-p}{1-U}) & \text{if } U > p
   \end{cases}
   ```

where ψ = s²/m², and m, s² are the conditional mean and variance of the process.

## Features

### 1. Simulation Module (`simulation.py`)
- Monte Carlo simulation of price paths
- Quadratic Exponential (QE) discretization for variance
- Jump process simulation with Poisson arrivals
- Correlated Brownian motions
- Numba optimization for performance

### 2. Pricing Module (`pricing.py`)
- European options (calls and puts)
- Asian options (arithmetic and geometric)
- Barrier options (up/down, in/out)
- Standard error estimation
- Greeks calculation via finite differences

### 3. Calibration Module (`calibration.py`)
- Market data fitting
- Parameter optimization using L-BFGS-B
- Implied volatility surface matching
- RMSE computation
- Flexible parameter bounds

### 4. Visualization Tools (`visualization.py`)
- Price path plots with jump highlighting
- Volatility surface visualization
- Greeks surface plots
- Calibration fit comparison
- Market vs. model implied volatility plots

## Installation

```bash
# Clone the repository
git clone https://github.com/irudrakshgupta/Volatility-Forecasting-Models.git
cd Volatility-Forecasting-Models

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Basic Simulation and Pricing

```python
from svjd.simulation import SVJDSimulator
from svjd.pricing import OptionPricer

# Set model parameters
params = {
    'S0': 100.0,     # Initial price
    'v0': 0.04,      # Initial variance
    'kappa': 2.0,    # Mean reversion speed
    'theta': 0.04,   # Long-term variance
    'sigma': 0.3,    # Volatility of variance
    'rho': -0.7,     # Correlation
    'r': 0.02,       # Risk-free rate
    'lambda': 1.0,   # Jump intensity
    'mu_j': -0.05,   # Jump mean
    'sigma_j': 0.1   # Jump volatility
}

# Create simulator and simulate paths
simulator = SVJDSimulator(params)
S, v = simulator.simulate(T=1.0, n_steps=252, n_paths=10000)

# Price options
pricer = OptionPricer(params)
call_price, call_se = pricer.price_european(K=100, T=1.0, option_type='call')
print(f"Call Price: {call_price:.4f} (±{call_se:.4f})")
```

### Model Calibration

```python
from svjd.calibration import SVJDCalibrator, MarketData
import numpy as np

# Prepare market data
market_data = MarketData(
    strikes=np.linspace(90, 110, 21),
    maturities=np.array([0.25, 0.5, 1.0]),
    implied_vols=your_market_vols,  # Your market volatility data
    spot=100.0,
    rate=0.02
)

# Calibrate model
calibrator = SVJDCalibrator(market_data)
calibrated_params = calibrator.calibrate()
rmse = calibrator.compute_rmse(calibrated_params)
```

### Visualization

```python
from svjd.visualization import SVJDVisualizer

visualizer = SVJDVisualizer()

# Plot price paths
visualizer.plot_paths(S, v, T=1.0, n_paths=5)

# Plot implied volatility surface
visualizer.plot_implied_vol_surface(
    pricer,
    strikes=np.linspace(80, 120, 41),
    maturities=np.linspace(0.1, 2.0, 20)
)

# Plot Greeks
visualizer.plot_greeks(pricer, strikes=np.linspace(80, 120, 41))
```

## Model Components

### 1. Parameter Classes
- `SVJDParameters`: Dataclass for model parameters
- `MarketData`: Dataclass for market calibration data

### 2. Core Classes
- `SVJDSimulator`: Path simulation
- `OptionPricer`: Option pricing and Greeks
- `SVJDCalibrator`: Model calibration
- `SVJDVisualizer`: Visualization tools

## Implementation Details

### Performance Optimizations
- Numba-accelerated jump process generation
- Vectorized operations for path simulation
- Efficient variance process discretization
- Parallel Monte Carlo simulation

### Numerical Stability
- Positive variance enforcement
- Adaptive QE scheme selection
- Robust implied volatility calculation
- Careful handling of extreme parameters

## Model Calibration

The calibration process minimizes the sum of squared differences between model and market implied volatilities using the L-BFGS-B algorithm. Parameter bounds ensure economically meaningful results:

```python
bounds = {
    'kappa': (0.1, 10.0),    # Mean reversion speed
    'theta': (0.01, 0.5),    # Long-term variance
    'sigma': (0.1, 1.0),     # Volatility of variance
    'rho': (-0.99, -0.1),    # Correlation
    'lambda': (0.1, 5.0),    # Jump intensity
    'mu_j': (-0.5, 0.0),     # Jump mean
    'sigma_j': (0.05, 0.5)   # Jump volatility
}
```

## Performance Considerations

1. **Simulation Efficiency**
   - Vectorized numpy operations
   - Numba-compiled jump generation
   - Pre-allocated arrays

2. **Memory Management**
   - Efficient array handling
   - Garbage collection for large simulations
   - Stream processing for big datasets

3. **Numerical Stability**
   - Robust variance process
   - Careful correlation handling
   - Stable Greeks calculation

## Model Limitations

1. **Parameter Stability**
   - Calibrated parameters may be unstable over time
   - Jump parameters can be difficult to estimate
   - Multiple local optima in calibration

2. **Computational Intensity**
   - Monte Carlo simulation is computationally expensive
   - QE discretization helps but still requires many paths
   - Greeks calculation requires multiple pricing runs

3. **Market Assumptions**
   - Perfect liquidity assumed
   - No transaction costs considered
   - Continuous trading possible
   - Constant interest rates

## Future Extensions

1. **Additional Features**
   - American option pricing
   - Multi-asset extensions
   - Term structure of parameters
   - Local volatility component

2. **Performance Optimizations**
   - GPU acceleration
   - Parallel path simulation
   - Variance reduction techniques
   - Quasi-Monte Carlo methods

3. **Model Enhancements**
   - Regime-switching extension
   - Time-dependent parameters
   - Alternative jump distributions
   - Stochastic interest rates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 