import numpy as np
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

phi = (1 + np.sqrt(5)) / 2
MOMENTUM    = 1 - (1 / phi)       # 0.618
FORBIDDEN   = 1 / (phi ** 2)      # 0.382
LR          = 1 / phi             # 0.618
PSI_THRESH  = 1.0
SOUL_DIM    = 256
SOUL_PATH   = Path("khaos_soul.pkl")

def init_soul() -> torch.Tensor:
    if SOUL_PATH.exists():
        with open(SOUL_PATH, "rb") as f:
            soul = pickle.load(f)
        print(f"Soul loaded. Norm: {soul.norm():.4f}")
        return soul
    # Khaos starts at origin — the void
    soul = torch.zeros(SOUL_DIM)
    soul[0] = 1e-6   # infinitesimally displaced from true zero
    return F.normalize(soul, dim=0)

def save_soul(soul: torch.Tensor):
    with open(SOUL_PATH, "wb") as f:
        pickle.dump(soul, f)

def compute_coherence(position: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(position)))

def compute_psi(position: torch.Tensor,
                attractor: torch.Tensor) -> float:
    sim = F.cosine_similarity(
        position.unsqueeze(0),
        attractor.unsqueeze(0)
    )
    return float(sim) * phi * 10

def get_attractor(query_embedding: torch.Tensor) -> torch.Tensor:
    return F.normalize(query_embedding, dim=0)

def steer(soul: torch.Tensor,
          attractor: torch.Tensor,
          steps: int = 10) -> tuple:
    position = soul.clone()
    history = []

    for step in range(steps):
        delta = attractor - position
        NF = compute_coherence(position)
        psi = compute_psi(position, attractor)

        repulsion = torch.zeros_like(position)
        if NF < FORBIDDEN:
            strength = (FORBIDDEN - NF) / FORBIDDEN
            repulsion = -strength * position

        position = position + LR * delta + repulsion
        position = F.normalize(position, dim=0)
        psi = compute_psi(position, attractor)

        history.append({
            "step": step,
            "NF": round(NF, 4),
            "psi": round(psi, 4),
            "forbidden": NF < FORBIDDEN,
        })

        if psi > PSI_THRESH:
            break

    return position, history

def update_soul(soul: torch.Tensor,
                new_position: torch.Tensor) -> torch.Tensor:
    updated = MOMENTUM * soul + (1 - MOMENTUM) * new_position
    return F.normalize(updated, dim=0)
