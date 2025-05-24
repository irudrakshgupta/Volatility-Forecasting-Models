"""
Visualization Module for SVJD Model

Implements plotting functions for price paths, volatility surfaces,
and option Greeks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from .simulation import SVJDSimulator
from .pricing import OptionPricer

class SVJDVisualizer:
    """Visualization tools for SVJD model."""
    
    def __init__(self):
        """Initialize visualizer with plot style settings."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_paths(self, S: np.ndarray, v: np.ndarray, jumps: Optional[np.ndarray] = None,
                  T: float = 1.0, n_paths: int = 5, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot sample price and variance paths.
        
        Args:
            S: Array of price paths
            v: Array of variance paths
            jumps: Array of jump times (optional)
            T: Time horizon
            n_paths: Number of paths to plot
            figsize: Figure size
        """
        t = np.linspace(0, T, S.shape[1])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot price paths
        for i in range(min(n_paths, S.shape[0])):
            ax1.plot(t, S[i], label=f'Path {i+1}', alpha=0.7)
            
            # Mark jumps if provided
            if jumps is not None:
                jump_times = t[jumps[i] != 0]
                jump_prices = S[i, jumps[i] != 0]
                ax1.scatter(jump_times, jump_prices, color='red', marker='x')
                
        ax1.set_title('Asset Price Paths')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot variance paths
        for i in range(min(n_paths, v.shape[0])):
            ax2.plot(t, v[i], label=f'Path {i+1}', alpha=0.7)
            
        ax2.set_title('Variance Paths')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Variance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_implied_vol_surface(self, pricer: OptionPricer,
                               strikes: np.ndarray, maturities: np.ndarray,
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot implied volatility surface.
        
        Args:
            pricer: Option pricer instance
            strikes: Array of strike prices
            maturities: Array of maturities
            figsize: Figure size
        """
        K, T = np.meshgrid(strikes, maturities)
        implied_vols = np.zeros_like(K)
        
        # Calculate implied vols
        for i in range(len(maturities)):
            for j in range(len(strikes)):
                price = pricer.price_european(strikes[j], maturities[i])[0]
                implied_vols[i, j] = pricer._price_to_implied_vol(
                    price, strikes[j], maturities[i],
                    pricer.params['S0'], pricer.params['r']
                )
                
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(K, T, implied_vols, cmap='viridis',
                             rstride=1, cstride=1, alpha=0.8)
        
        ax.set_title('Implied Volatility Surface')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
        
    def plot_greeks(self, pricer: OptionPricer, strikes: np.ndarray,
                   T: float = 1.0, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot option Greeks across strikes.
        
        Args:
            pricer: Option pricer instance
            strikes: Array of strike prices
            T: Time to maturity
            figsize: Figure size
        """
        greeks = []
        for K in strikes:
            greeks.append(pricer.calculate_greeks(K, T))
            
        # Extract individual Greeks
        deltas = [g['delta'] for g in greeks]
        gammas = [g['gamma'] for g in greeks]
        vegas = [g['vega'] for g in greeks]
        thetas = [g['theta'] for g in greeks]
        rhos = [g['rho'] for g in greeks]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=figsize)
        
        # Delta
        ax1.plot(strikes, deltas)
        ax1.set_title('Delta')
        ax1.set_xlabel('Strike')
        ax1.grid(True)
        
        # Gamma
        ax2.plot(strikes, gammas)
        ax2.set_title('Gamma')
        ax2.set_xlabel('Strike')
        ax2.grid(True)
        
        # Vega
        ax3.plot(strikes, vegas)
        ax3.set_title('Vega')
        ax3.set_xlabel('Strike')
        ax3.grid(True)
        
        # Theta
        ax4.plot(strikes, thetas)
        ax4.set_title('Theta')
        ax4.set_xlabel('Strike')
        ax4.grid(True)
        
        # Rho
        ax5.plot(strikes, rhos)
        ax5.set_title('Rho')
        ax5.set_xlabel('Strike')
        ax5.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_calibration_fit(self, market_vols: np.ndarray, model_vols: np.ndarray,
                           strikes: np.ndarray, maturities: np.ndarray,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot comparison of market and model implied volatilities.
        
        Args:
            market_vols: Market implied volatilities
            model_vols: Model implied volatilities
            strikes: Array of strike prices
            maturities: Array of maturities
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, len(maturities),
                                figsize=figsize, sharey=True)
        
        for i, T in enumerate(maturities):
            ax = axes[i] if len(maturities) > 1 else axes
            
            ax.plot(strikes, market_vols[i], 'o-', label='Market')
            ax.plot(strikes, model_vols[i], 's--', label='Model')
            
            ax.set_title(f'T = {T:.2f}')
            ax.set_xlabel('Strike')
            if i == 0:
                ax.set_ylabel('Implied Volatility')
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        plt.show() 