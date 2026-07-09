"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.dim_in = state_dim
        self.dim_out = action_dim * chunk_size

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in, hidden_dims[0]),
            nn.ReLU(),
            *[
                layer
                for i in range(1, len(hidden_dims))
                for layer in [
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.ReLU()
                ]
            ],
            nn.Linear(hidden_dims[-1], self.dim_out),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.sample_actions(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.mlp(state).reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.dim_action_chunk = action_dim * chunk_size
        self.dim_in = state_dim + self.dim_action_chunk + 1
        self.dim_out = self.dim_action_chunk
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in, hidden_dims[0]),
            nn.ReLU(),
            *[
                layer
                for i in range(1, len(hidden_dims))
                for layer in [
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.ReLU()
                ]
            ],
            nn.Linear(hidden_dims[-1], self.dim_out),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        a1 = action_chunk.reshape(-1, self.dim_action_chunk)
        a0 = torch.randn_like(a1)
        t = torch.rand(batch_size, 1, device=state.device, dtype=torch.float32)

        at = t * a1 + (1 - t) * a0

        vel = a1 - a0
        pred_vel = self.mlp(torch.cat([state, at, t], dim=-1))
        return F.mse_loss(pred_vel, vel)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        a = torch.randn(batch_size, self.dim_action_chunk, device=state.device, dtype=torch.float32)
        for n in range(num_steps):
            t = torch.full((batch_size, 1), n / num_steps, device=state.device, dtype=torch.float32)
            v = self.mlp(torch.cat([state, a, t], dim=-1))
            a = a + dt * v
        return a.reshape(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
