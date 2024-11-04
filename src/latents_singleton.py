import torch
from typing import Optional, List


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Latents(metaclass=Singleton):
    def __init__(self) -> None:
        self.history: List[torch.FloatTensor] = []

    def __str__(self) -> Optional[str]:
        if self.is_empty():
            return None

        return str(len(self.history))

    def is_empty(self) -> bool:
        return self.latents is None

    def add_latents(self, latents: torch.FloatTensor):
        self.history.append(latents)

    def get_latents(self, diffusion_step: int) -> torch.FloatTensor:
        return self.history[diffusion_step]

    def get_history(self) -> List[torch.FloatTensor]:
        return self.history

    def clear(self):
        self.history = []

    def dump_and_clear(self):
        history = self.history
        self.clear()
        return history
