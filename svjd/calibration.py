"""
Calibration Module for SVJD Model

Implements parameter calibration methods for the SVJD model
using market implied volatility data.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .pricing import OptionPricer

@dataclass
class MarketData:
    """Container for market option data."""
    strikes: np.ndarray
    maturities: np.ndarray
    implied_vols: np.ndarray
    spot: float
    rate: float

class SVJDCalibrator:
    """Calibrator for SVJD model parameters."""
    
    def __init__(self, market_data: MarketData,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize calibrator with market data and parameter bounds.
        
        Args:
            market_data: Market option data
            bounds: Dictionary of parameter bounds (min, max)
        """
        self.market_data = market_data
        self.bounds = bounds or {
            'kappa': (0.1, 10.0),
            'theta': (0.01, 0.5),
            'sigma': (0.1, 1.0),
            'rho': (-0.99, -0.1),
            'lambda': (0.1, 5.0),
            'mu_j': (-0.5, 0.0),
            'sigma_j': (0.05, 0.5)
        }
        
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        return np.array([
            params['kappa'],
            params['theta'],
            params['sigma'],
            params['rho'],
            params['lambda'],
            params['mu_j'],
            params['sigma_j']
        ])
        
    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {
            'S0': self.market_data.spot,
            'v0': x[1],  # theta as initial variance
            'r': self.market_data.rate,
            'kappa': x[0],
            'theta': x[1],
            'sigma': x[2],
            'rho': x[3],
            'lambda': x[4],
            'mu_j': x[5],
            'sigma_j': x[6]
        }
        
    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for calibration.
        
        Computes the sum of squared differences between model and market implied vols.
        """
        params = self._array_to_params(x)
        pricer = OptionPricer(params)
        
        total_error = 0.0
        
        for i, T in enumerate(self.market_data.maturities):
            for j, K in enumerate(self.market_data.strikes):
                # Get market implied vol
                market_vol = self.market_data.implied_vols[i, j]
                
                # Calculate model price
                model_price = pricer.price_european(K, T)[0]
                
                # Convert model price to implied vol using Newton's method
                model_vol = self._price_to_implied_vol(
                    model_price, K, T,
                    self.market_data.spot,
                    self.market_data.rate
                )
                
                # Add squared error
                total_error += (market_vol - model_vol)**2
                
        return total_error
        
    def _price_to_implied_vol(self, price: float, K: float, T: float,
                             S: float, r: float, tol: float = 1e-5,
                             max_iter: int = 100) -> float:
        """
        Convert option price to implied volatility using Newton's method.
        
        Args:
            price: Option price
            K: Strike price
            T: Time to maturity
            S: Spot price
            r: Risk-free rate
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            
        Returns:
            Implied volatility
        """
        def bs_price(sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            
        def bs_vega(sigma):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S*np.sqrt(T)*norm.pdf(d1)
            
        from scipy.stats import norm
        
        # Initial guess
        sigma = 0.3
        
        for _ in range(max_iter):
            price_diff = bs_price(sigma) - price
            if abs(price_diff) < tol:
                return sigma
                
            sigma = sigma - price_diff/bs_vega(sigma)
            
        return sigma
        
    def calibrate(self, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calibrate model parameters to market data.
        
        Args:
            initial_params: Initial parameter guess
            
        Returns:
            Calibrated parameters
        """
        if initial_params is None:
            initial_params = {
                'kappa': 2.0,
                'theta': 0.04,
                'sigma': 0.3,
                'rho': -0.7,
                'lambda': 1.0,
                'mu_j': -0.05,
                'sigma_j': 0.1
            }
            
        # Convert bounds to list for scipy.optimize
        bounds_list = [
            self.bounds['kappa'],
            self.bounds['theta'],
            self.bounds['sigma'],
            self.bounds['rho'],
            self.bounds['lambda'],
            self.bounds['mu_j'],
            self.bounds['sigma_j']
        ]
        
        # Run optimization
        x0 = self._params_to_array(initial_params)
        result = minimize(
            self._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Warning: Calibration did not converge. Message: {result.message}")
            
        # Convert result back to parameter dictionary
        return self._array_to_params(result.x)
        
    def compute_rmse(self, params: Dict[str, float]) -> float:
        """
        Compute Root Mean Square Error of parameter fit.
        
        Args:
            params: Model parameters
            
        Returns:
            RMSE of implied volatility fit
        """
        x = self._params_to_array(params)
        mse = self._objective(x) / (len(self.market_data.strikes) * len(self.market_data.maturities))
        return np.sqrt(mse) 