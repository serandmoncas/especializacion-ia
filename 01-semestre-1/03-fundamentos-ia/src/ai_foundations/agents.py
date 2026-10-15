"""Agentes en el mundo de la aspiradora (Russell & Norvig, cap. 2), en versión 2D y dinámica.

Entorno: grilla de celdas que se ensucian al azar con probabilidad `dirt_prob` por paso.
Percepción: (posición, ¿está sucia mi celda?), parcialmente observable: el agente no ve las demás.
Medida de desempeño: celdas limpias promedio por paso, menos un costo por movimiento.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Percept = tuple[tuple[int, int], bool]
Agent = Callable[[Percept], str]
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class VacuumWorld:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        dirt_prob: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng(0)
        self.width, self.height, self.dirt_prob = width, height, dirt_prob
        self.dirty = self.rng.random((height, width)) < 0.5
        self.pos = (int(self.rng.integers(width)), int(self.rng.integers(height)))

    def percept(self) -> Percept:
        x, y = self.pos
        return self.pos, bool(self.dirty[y, x])

    def step(self, action: str) -> None:
        x, y = self.pos
        if action == "suck":
            self.dirty[y, x] = False
        elif action in MOVES:
            dx, dy = MOVES[action]
            self.pos = (min(max(x + dx, 0), self.width - 1), min(max(y + dy, 0), self.height - 1))
        self.dirty |= self.rng.random(self.dirty.shape) < self.dirt_prob


@dataclass
class EpisodeStats:
    clean_fraction: float  # fracción de celdas limpias, promedio en el episodio
    moves: int

    def score(self, move_cost: float = 0.01) -> float:
        return self.clean_fraction - move_cost * self.moves / 1000


def run_episode(agent: Agent, world: VacuumWorld, steps: int = 1000) -> EpisodeStats:
    clean, moves = 0.0, 0
    for _ in range(steps):
        action = agent(world.percept())
        moves += action in MOVES
        world.step(action)
        clean += 1 - world.dirty.mean()
    return EpisodeStats(clean / steps, moves)


def random_agent(rng: np.random.Generator) -> Agent:
    actions = ["suck", *MOVES]
    return lambda percept: actions[rng.integers(len(actions))]


def simple_reflex_agent(rng: np.random.Generator) -> Agent:
    """Regla condición–acción: si está sucio, aspirar; si no, moverse al azar.

    Sin memoria: no sabe dónde ya limpió, así que revisita celdas al azar.
    """
    moves = list(MOVES)

    def act(percept: Percept) -> str:
        _, is_dirty = percept
        return "suck" if is_dirty else moves[rng.integers(len(moves))]

    return act


def model_based_agent(width: int, height: int) -> Agent:
    """Mantiene un modelo del mundo: cuándo visitó cada celda por última vez.

    Como la suciedad aparece con el tiempo, la celda visitada hace más tiempo es la que más
    probablemente esté sucia: se mueve hacia el vecino con la visita más antigua.
    """
    last_seen = np.full((height, width), -1)
    t = 0

    def act(percept: Percept) -> str:
        nonlocal t
        t += 1
        (x, y), is_dirty = percept
        last_seen[y, x] = t
        if is_dirty:
            return "suck"
        best, best_age = "up", -1
        for name, (dx, dy) in MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                age = t - last_seen[ny, nx]
                if age > best_age:
                    best, best_age = name, age
        return best

    return act
