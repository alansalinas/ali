from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Entity:
    id: int
    name: str
    confidence: float
    children_only: bool = False


@dataclass
class Topic:
    id: int
    topic: str
    document: int
    text: str
    confidence: float
    vector: NDArray
    start_token: int = -1
    end_token: int = 0

    def __post_init__(self):
        if self.start_token >= self.end_token:
            raise ValueError('End token must be higher than start token')

    def __repr__(self) -> str:
        return (
            f'Topic(id={self.id}, topic={self.topic}, document={self.document}, '
            f'text={self.text}, confidence={self.confidence}, '
            f'start_token={self.start_token}, end_token={self.end_token})'
        )