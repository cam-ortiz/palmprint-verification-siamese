import random 

def generate_triplets(samples: List[Sample], num_triplets: int, seed: int = 42) -> List[Tuple[str, str, str]]:
    """
    Generate triplets of image paths (anchor, positive, negative) where:
    - anchor and positive belong to the same subject
    - negative belongs to a different subject
    """
    random.seed(seed)
    triplets = []
    n = len(samples)

    for _ in range(num_triplets):
        # Sample anchor and positive from the same subject
        idx_anchor = random.randint(0, n - 1)
        anchor_sample = samples[idx_anchor]
        positive_samples = [s for s in samples if s.label == anchor_sample.label and s.img_path != anchor_sample.img_path]
        
        if not positive_samples:
            continue  # Skip if no positive sample is available
        
        positive_sample = random.choice(positive_samples)

        # Sample negative from a different subject
        negative_samples = [s for s in samples if s.label != anchor_sample.label]
        
        if not negative_samples:
            continue  # Skip if no negative sample is available
        
        negative_sample = random.choice(negative_samples)

        triplets.append((anchor_sample.img_path, positive_sample.img_path, negative_sample.img_path))

    return triplets
