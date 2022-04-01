import random


def happen(probability: float) -> bool:
    return random.random() < probability
