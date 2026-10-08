from dataclasses import dataclass
import numpy as np


@dataclass
class Track:

    track_id: int

    x: np.ndarray

    P: np.ndarray

    age: int = 1

    missed: int = 0

    hits: int = 1

    confirmed: bool = False