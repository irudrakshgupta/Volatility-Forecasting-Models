"""
Basic tests for the SVJD model implementation.
"""

import numpy as np
import pytest
from svjd.simulation import SVJDSimulator
from svjd.pricing import OptionPricer

def test_simulation():
    """Test basic simulation functionality."""
    params = {
        'S0': 100.0,
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'r': 0.02,
        'lambda': 1.0,
        'mu_j': -0.05,
        'sigma_j': 0.1
    }
    
    simulator = SVJDSimulator(params)
    
    # Test simulation dimensions
    T = 1.0
    n_steps = 252
    n_paths = 1000
    
    S, v = simulator.simulate(T, n_steps, n_paths, seed=42)
    
    assert S.shape == (n_paths, n_steps + 1)
    assert v.shape == (n_paths, n_steps + 1)
    
    # Test initial values
    np.testing.assert_allclose(S[:, 0], params['S0'])
    np.testing.assert_allclose(v[:, 0], params['v0'])
    
    # Test variance positivity
    assert np.all(v > 0)

def test_option_pricing():
    """Test basic option pricing functionality."""
    params = {
        'S0': 100.0,
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'r': 0.02,
        'lambda': 1.0,
        'mu_j': -0.05,
        'sigma_j': 0.1
    }
    
    pricer = OptionPricer(params)
    
    # Test ATM European call and put prices
    K = 100.0
    T = 1.0
    
    call_price, call_se = pricer.price_european(K, T, 'call', n_paths=10000, seed=42)
    put_price, put_se = pricer.price_european(K, T, 'put', n_paths=10000, seed=42)
    
    # Test put-call parity approximately
    S0 = params['S0']
    r = params['r']
    parity_diff = abs(call_price - put_price - S0 + K * np.exp(-r * T))
    assert parity_diff < 0.1  # Allow for Monte Carlo error
    
    # Test positive prices
    assert call_price > 0
    assert put_price > 0
    
    # Test standard errors
    assert call_se > 0
    assert put_se > 0
    
    # Test barrier option price is less than vanilla
    barrier = 120.0
    barrier_price, _ = pricer.price_barrier(
        K, T, barrier, 'up-and-out', 'call',
        n_paths=10000, seed=42
    )
    assert barrier_price < call_price

def test_greeks():
    """Test basic properties of option Greeks."""
    params = {
        'S0': 100.0,
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'r': 0.02,
        'lambda': 1.0,
        'mu_j': -0.05,
        'sigma_j': 0.1
    }
    
    pricer = OptionPricer(params)
    
    # Calculate Greeks for ATM option
    K = 100.0
    T = 1.0
    greeks = pricer.calculate_greeks(K, T, 'call')
    
    # Test basic properties
    assert 0 < greeks['delta'] < 1  # Call delta between 0 and 1
    assert greeks['gamma'] > 0      # Gamma is positive
    assert greeks['vega'] > 0       # Vega is positive
    assert greeks['theta'] < 0      # Theta is negative for calls
    
if __name__ == '__main__':
    pytest.main([__file__]) 