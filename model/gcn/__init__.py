"""
GCN model package.
"""

from .model import GCN, MLP
from .party import Server, Client

__all__ = ['GCN', 'MLP', 'Server', 'Client']
