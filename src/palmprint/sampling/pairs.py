import random

def generate_pairs(samples: List[Sample], num_pairs: int, seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Generate pairs of image paths with labels indicating if they belong to the same subject (1) or not (0).
    """
    random.seed(seed)
    pairs = []
    n = len(samples)

    for _ in range(num_pairs):
        idx1, idx2 = random.sample(range(n), 2)
        img1, label1 = samples[idx1].img_path, samples[idx1].label
        img2, label2 = samples[idx2].img_path, samples[idx2].label
        same_subject = int(label1 == label2)
        pairs.append((img1, img2, same_subject))

    return pairs
