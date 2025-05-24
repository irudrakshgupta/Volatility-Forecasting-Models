"""
SVJD Model Simulation Module

Implements the Stochastic Volatility Jump-Diffusion model simulation
using the Quadratic-Exponential scheme for variance process.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from numba import jit

@dataclass
class SVJDParameters:
    """Model parameters for the SVJD model."""
    S0: float       # Initial asset price
    v0: float       # Initial variance
    kappa: float    # Mean reversion speed
    theta: float    # Long-term variance
    sigma: float    # Volatility of variance
    rho: float      # Correlation between price and variance
    r: float        # Risk-free rate
    lambda_: float  # Jump intensity
    mu_j: float     # Jump size mean
    sigma_j: float  # Jump size volatility

    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'SVJDParameters':
        """Create parameters from dictionary."""
        return cls(
            S0=params.get('S0', 100.0),
            v0=params.get('v0', 0.04),
            kappa=params.get('kappa', 2.0),
            theta=params.get('theta', 0.04),
            sigma=params.get('sigma', 0.3),
            rho=params.get('rho', -0.7),
            r=params.get('r', 0.02),
            lambda_=params.get('lambda', 1.0),
            mu_j=params.get('mu_j', -0.05),
            sigma_j=params.get('sigma_j', 0.1)
        )

@jit(nopython=True)
def _generate_jumps(n_steps: int, T: float, lambda_: float, mu_j: float, 
                   sigma_j: float, n_paths: int) -> np.ndarray:
    """Generate jump process paths."""
    dt = T / n_steps
    jump_times = np.random.poisson(lambda_ * dt, size=(n_paths, n_steps))
    jump_sizes = np.random.normal(mu_j, sigma_j, size=(n_paths, n_steps))
    return jump_times * jump_sizes

class SVJDSimulator:
    """Simulator for the SVJD model."""
    
    def __init__(self, params: Dict[str, float]):
        """Initialize the simulator with model parameters."""
        self.params = SVJDParameters.from_dict(params)
        
    def _qe_scheme(self, v: float, dt: float) -> Tuple[float, float]:
        """Quadratic-Exponential scheme for variance process."""
        m = self.params.theta + (v - self.params.theta) * np.exp(-self.params.kappa * dt)
        s2 = (v * self.params.sigma**2 * np.exp(-self.params.kappa * dt) / self.params.kappa) * \
             (1 - np.exp(-self.params.kappa * dt)) + \
             (self.params.theta * self.params.sigma**2 / (2 * self.params.kappa)) * \
             (1 - np.exp(-self.params.kappa * dt))**2
        psi = s2 / m**2
        
        if psi <= 1.5:
            # Truncated Gaussian scheme
            b2 = 2 / psi - 1 + np.sqrt(2 / psi * (2 / psi - 1))
            a = m / (1 + b2)
            return a, b2
        else:
            # Quadratic scheme
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            return beta, p
            
    def simulate(self, T: float, n_steps: int, n_paths: int, 
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset price and variance paths.
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.params.S0
        v[:, 0] = self.params.v0
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, 1, size=(n_paths, n_steps))
        dW2 = self.params.rho * dW1 + np.sqrt(1 - self.params.rho**2) * \
              np.random.normal(0, 1, size=(n_paths, n_steps))
        
        # Generate jumps
        jumps = _generate_jumps(n_steps, T, self.params.lambda_, 
                              self.params.mu_j, self.params.sigma_j, n_paths)
        
        # Simulate paths
        for t in range(n_steps):
            # Simulate variance using QE scheme
            a, b = self._qe_scheme(v[:, t].mean(), dt)
            U = np.random.uniform(0, 1, n_paths)
            
            if b > 0:  # b2 case
                v[:, t+1] = a * (np.sqrt(b) * dW2[:, t] + np.sqrt(b))**2
            else:  # p case
                v[:, t+1] = np.where(U <= b,
                                   0,
                                   np.log((1 - b)/(1 - U)) / a)
            
            # Ensure positive variance
            v[:, t+1] = np.maximum(v[:, t+1], 1e-10)
            
            # Simulate asset price
            drift = (self.params.r - self.params.lambda_ * self.params.mu_j - \
                    0.5 * v[:, t]) * dt
            diffusion = np.sqrt(v[:, t]) * dW1[:, t] * sqrt_dt
            jump = jumps[:, t]
            
            S[:, t+1] = S[:, t] * np.exp(drift + diffusion + jump)
        
        return S, v 