from __future__ import annotations
import collections
from typing import Optional, List, Counter, NamedTuple
import weakref
import math


class Sample(NamedTuple):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class KnownSample(NamedTuple):
    sample: Sample
    species: str


class TestingKnownSample:
    def __init__(
        self, sample: KnownSample, classification: Optional[str] = None
    ) -> None:
        self.sample = sample
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sample={self.sample!r}, "
            f"classification={self.classification!r})"
        )


class TrainingKnownSample(NamedTuple):
    sample: KnownSample


class UnknownSample:
    def __init__(
        self, sample: Sample, classification: Optional[str] = None
    ) -> None:
        self.sample = sample
        self.classification = classification

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sample={self.sample!r}, classification={self.classification!r})"


class Distance:
    """유클라디언 거리 계산"""

    def distance(self, s1: Sample, s2: KnownSample) -> float:
        squared_distance = (
            (s1.sepal_length - s2.sample.sepal_length) ** 2
            + (s1.sepal_width - s2.sample.sepal_width) ** 2
            + (s1.petal_length - s2.sample.petal_length) ** 2
            + (s1.petal_width - s2.sample.petal_width) ** 2
        )
        return math.sqrt(squared_distance)


class Hyperparameter:

    def __init__(self, k: int, algorithm: Distance, data: TrainingData) -> None:
        self.k = k
        self.algorithm = algorithm
        self.data = weakref.ref(data)

    def classify(self, unknown: Sample) -> str:
        """K-NN 알고리즘"""
        if not (training_data := self.data()):
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            (self.algorithm.distance(unknown, known.sample), known)
            for known in training_data.training
        )
        k_nearest = (known.sample.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


class TrainingData:
    def __init__(self) -> None:
        self.testing: List[TestingKnownSample] = []
        self.training: List[TrainingKnownSample] = []
        self.tuning: List[Hyperparameter] = []
        self.test_placeholder()

    def load(self, raw_data_source: List[dict[str, str]]) -> None:
        for n, row in enumerate(raw_data_source):
            sample = Sample(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
            )
            known_sample = KnownSample(sample=sample, species=row["species"])
            if n % 5 == 0:
                self.testing.append(TestingKnownSample(sample=known_sample))
            else:
                self.training.append(TrainingKnownSample(sample=known_sample))