"""
GFlowNet module for Logic Gate Network generation.

This module provides the core GFlowNet components for training neural networks
to generate Logic Gate Networks through generative flow networks.

Main Components:
- LGNMDP: MDP wrapper for backward trajectory sampling
- LGNActionSpace: Action space for forward policy sampling
"""

from .lgn_mdp import LGNMDP
from .action_space import LGNActionSpace

__all__ = ['LGNMDP', 'LGNActionSpace']
