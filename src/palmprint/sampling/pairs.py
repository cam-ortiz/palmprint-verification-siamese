"""
Balanced pair sampling utilities for Siamese palmprint verification.

Each generated pair is labeled as:
    1 = genuine pair, same person-hand identity
    0 = impostor pair, different person-hand identity
"""

from __future__ import annotations

import random
from collections import defaultdict

from palmprint.datasets.bmpd import Sample


Pair = tuple[str, str, int]


def group_samples_by_label(samples: list[Sample]) -> dict[int, list[Sample]]:
    label_to_samples: dict[int, list[Sample]] = defaultdict(list)

    for sample in samples:
        label_to_samples[sample.label].append(sample)

    return dict(label_to_samples)


def generate_genuine_pair(
    label_to_samples: dict[int, list[Sample]],
    rng: random.Random,
) -> Pair:
    """
    Generate one genuine pair from the same identity.
    """
    valid_labels = [
        label for label, label_samples in label_to_samples.items()
        if len(label_samples) >= 2
    ]

    if not valid_labels:
        raise ValueError("No labels have at least two samples for genuine pairs.")

    label = rng.choice(valid_labels)
    sample1, sample2 = rng.sample(label_to_samples[label], 2)

    return sample1.img_path, sample2.img_path, 1


def generate_impostor_pair(
    label_to_samples: dict[int, list[Sample]],
    rng: random.Random,
) -> Pair:
    """
    Generate one impostor pair from two different identities.
    """
    valid_labels = list(label_to_samples.keys())

    if len(valid_labels) < 2:
        raise ValueError("Need at least two labels for impostor pairs.")

    label1, label2 = rng.sample(valid_labels, 2)

    sample1 = rng.choice(label_to_samples[label1])
    sample2 = rng.choice(label_to_samples[label2])

    return sample1.img_path, sample2.img_path, 0


def generate_balanced_pairs(
    samples: list[Sample],
    num_pairs: int,
    seed: int = 42,
) -> list[Pair]:
    """
    Generate balanced Siamese pairs.

    The result is approximately:
        50% genuine pairs
        50% impostor pairs
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")

    rng = random.Random(seed)
    label_to_samples = group_samples_by_label(samples)

    num_genuine = num_pairs // 2
    num_impostor = num_pairs - num_genuine

    pairs: list[Pair] = []

    for _ in range(num_genuine):
        pairs.append(generate_genuine_pair(label_to_samples, rng))

    for _ in range(num_impostor):
        pairs.append(generate_impostor_pair(label_to_samples, rng))

    rng.shuffle(pairs)

    return pairs