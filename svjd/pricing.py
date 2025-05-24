"""
Option Pricing Module for SVJD Model

Implements various option pricing methods using Monte Carlo simulation
for European, Asian, and barrier options under the SVJD model.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from .simulation import SVJDSimulator

class OptionPricer:
    """Option pricing class for SVJD model."""
    
    def __init__(self, params: Dict[str, float]):
        """Initialize pricer with model parameters."""
        self.simulator = SVJDSimulator(params)
        self.params = params
        
    def _discount_factor(self, T: float) -> float:
        """Calculate discount factor."""
        return np.exp(-self.params['r'] * T)
        
    def price_european(self, K: float, T: float, option_type: str = 'call',
                      n_paths: int = 100000, n_steps: int = 252,
                      seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price European options using Monte Carlo simulation.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Simulate paths
        S, _ = self.simulator.simulate(T, n_steps, n_paths, seed)
        
        # Calculate terminal payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
            
        # Discount payoffs
        discount = self._discount_factor(T)
        discounted_payoffs = payoffs * discount
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        return price, std_err
        
    def price_asian(self, K: float, T: float, averaging_type: str = 'arithmetic',
                   option_type: str = 'call', n_paths: int = 100000,
                   n_steps: int = 252, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price Asian options using Monte Carlo simulation.
        
        Args:
            K: Strike price
            T: Time to maturity
            averaging_type: 'arithmetic' or 'geometric'
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Simulate paths
        S, _ = self.simulator.simulate(T, n_steps, n_paths, seed)
        
        # Calculate average prices
        if averaging_type.lower() == 'arithmetic':
            avg_prices = np.mean(S, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(S), axis=1))
            
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
            
        # Discount payoffs
        discount = self._discount_factor(T)
        discounted_payoffs = payoffs * discount
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        return price, std_err
        
    def price_barrier(self, K: float, T: float, barrier: float,
                     barrier_type: str = 'up-and-out', option_type: str = 'call',
                     n_paths: int = 100000, n_steps: int = 252,
                     seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price barrier options using Monte Carlo simulation.
        
        Args:
            K: Strike price
            T: Time to maturity
            barrier: Barrier level
            barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', or 'down-and-in'
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Simulate paths
        S, _ = self.simulator.simulate(T, n_steps, n_paths, seed)
        
        # Check barrier conditions
        if barrier_type.startswith('up'):
            barrier_hit = np.any(S > barrier, axis=1)
        else:  # down
            barrier_hit = np.any(S < barrier, axis=1)
            
        # Calculate terminal payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
            
        # Apply barrier condition
        if barrier_type.endswith('out'):
            payoffs[barrier_hit] = 0
        else:  # in
            payoffs[~barrier_hit] = 0
            
        # Discount payoffs
        discount = self._discount_factor(T)
        discounted_payoffs = payoffs * discount
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        return price, std_err
        
    def calculate_greeks(self, K: float, T: float, option_type: str = 'call',
                        eps: float = 1e-4) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            eps: Finite difference step size
            
        Returns:
            Dictionary containing Delta, Gamma, Vega, Theta, and Rho
        """
        base_price = self.price_european(K, T, option_type)[0]
        params = self.params.copy()
        
        # Delta
        params_up = params.copy()
        params_up['S0'] = params['S0'] * (1 + eps)
        delta = (OptionPricer(params_up).price_european(K, T, option_type)[0] - base_price) / \
                (params['S0'] * eps)
                
        # Gamma
        params_down = params.copy()
        params_down['S0'] = params['S0'] * (1 - eps)
        price_down = OptionPricer(params_down).price_european(K, T, option_type)[0]
        price_up = OptionPricer(params_up).price_european(K, T, option_type)[0]
        gamma = (price_up - 2*base_price + price_down) / (params['S0'] * eps)**2
        
        # Vega
        params_vol_up = params.copy()
        params_vol_up['v0'] = params['v0'] * (1 + eps)
        vega = (OptionPricer(params_vol_up).price_european(K, T, option_type)[0] - base_price) / \
               (params['v0'] * eps)
               
        # Theta
        price_dt = self.price_european(K, T*(1-eps), option_type)[0]
        theta = (price_dt - base_price) / (T*eps)
        
        # Rho
        params_r_up = params.copy()
        params_r_up['r'] = params['r'] + eps
        rho = (OptionPricer(params_r_up).price_european(K, T, option_type)[0] - base_price) / eps
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        } 