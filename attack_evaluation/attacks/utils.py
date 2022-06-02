from dataclasses import dataclass
from typing import Callable


@dataclass
class ConfigGetter:
    config: Callable
    getter: Callable
