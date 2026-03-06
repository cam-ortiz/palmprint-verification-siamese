import numpy as np
from time import perf_counter
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

def _compute_eer(fpr, tpr):
    """Compute the Equal Error Rate (EER) given the false positive rates and true positive rates."""
    # Calculate the absolute difference between FPR and TPR
    fnr = 1.0 - tpr
    
    # Find the index where this difference is minimized
    idx = np.nanargmin(np.abs(fpr - fnr))
    
    # The EER is the value of FPR (or TPR) at this index
    eer = (fpr[idx] + tpr[idx]) / 2.0

    return eer

def _sample_genuine_pairs(X, y, n_pairs, rng):
    """
    Sample genuine pairs (same subject) from PCA feature matrix X and labels y.
    Returns arrays of distances and labels (all 1s).
    """
    distances = []
    labels = []

    # Indices for each subject
    unique_labels = np.unique(y)
    label_to_indices = {lab: np.where(y == lab)[0] for lab in unique_labels}

    attempts = 0
    max_attempts = n_pairs * 10  # safety to avoid infinite loops

    while len(distances) < n_pairs and attempts < max_attempts:
        attempts += 1
        # Pick a random subject that has at least 2 samples
        lab = rng.choice(unique_labels)
        idxs = label_to_indices[lab]
        if len(idxs) < 2:
            continue

        i, j = rng.choice(idxs, size=2, replace=False)
        d = np.linalg.norm(X[i] - X[j])
        distances.append(d)
        labels.append(1)  # 1 = genuine

    return np.array(distances), np.array(labels)


def _sample_impostor_pairs(X, y, n_pairs, rng):
    """
    Sample impostor pairs (different subjects) from PCA feature matrix X and labels y.
    Returns arrays of distances and labels (all 0s).
    """
    distances = []
    labels = []

    unique_labels = np.unique(y)
    label_to_indices = {lab: np.where(y == lab)[0] for lab in unique_labels}

    attempts = 0
    max_attempts = n_pairs * 10

    while len(distances) < n_pairs and attempts < max_attempts:
        attempts += 1
        # Pick two different subjects
        lab_a, lab_b = rng.choice(unique_labels, size=2, replace=False)
        idxs_a = label_to_indices[lab_a]
        idxs_b = label_to_indices[lab_b]
        if len(idxs_a) == 0 or len(idxs_b) == 0:
            continue

        i = rng.choice(idxs_a)
        j = rng.choice(idxs_b)
        d = np.linalg.norm(X[i] - X[j])
        distances.append(d)
        labels.append(0)  # 0 = impostor

    return np.array(distances), np.array(labels)

def _compute_best_accuracy(labels, scores):
    """
    Compute best achievable verification accuracy by sweeping thresholds.
    """
    thresholds = np.unique(scores)
    best_acc = 0.0
    best_thresh = None

    for t in thresholds:
        preds = (scores >= t).astype(int)  # 1 = genuine
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_acc, best_thresh

def evaluate_embedding_verification(
    X_pca,
    y,
    n_genuine_pairs=5000,
    n_impostor_pairs=5000,
    random_state=42,
):
    """
    PCA-only verification baseline on a given split (typically TEST):

    - X_pca: PCA feature matrix, shape (N, d)
    - y: subject labels, shape (N,)
    - n_genuine_pairs: number of genuine (same-subject) pairs to sample
    - n_impostor_pairs: number of impostor (different-subject) pairs to sample

    Returns:
        metrics: dict with keys:
            'roc_auc', 'eer', 'n_genuine', 'n_impostor', 'eval_time'
        pair_labels: array of 0/1 labels for each pair (1 = genuine, 0 = impostor)
        pair_scores: array of similarity scores (higher = more similar)
    """

    rng = np.random.default_rng(random_state)


    # 1) Sample genuine and impostor pairs
    genuine_distances, genuine_labels = _sample_genuine_pairs(
        X_pca, y, n_genuine_pairs, rng
    )
    impostor_distances, impostor_labels = _sample_impostor_pairs(
        X_pca, y, n_impostor_pairs, rng
    )

    # Concatenate
    all_distances = np.concatenate([genuine_distances, impostor_distances])
    all_labels = np.concatenate([genuine_labels, impostor_labels])

    # 2) Convert distances to similarity scores for ROC:
    #    Higher score = more similar
    #    A simple choice: similarity = -distance
    scores = -all_distances

    # 3) Compute ROC-AUC and EER
    fpr, tpr, _ = roc_curve(all_labels, scores)
    roc_auc = roc_auc_score(all_labels, scores)
    eer = _compute_eer(fpr, tpr)


    # Compute best accuracy
    best_acc, best_thresh = _compute_best_accuracy(all_labels, scores)

    metrics = {
        "roc_auc": roc_auc,
        "eer": eer,
        "best_accuracy": best_acc,
        "best_threshold": float(best_thresh),
        "n_genuine": int(genuine_distances.shape[0]),
        "n_impostor": int(impostor_distances.shape[0])
    }

    return metrics

def estimate_embedding_verification_time(extractor, X_raw, n_trials=1000, random_state=42):
    """
    Estimate average verification time for ANY embedding extractor.

    Process per trial:
        - Select two random images from X_raw
        - extractor.transform(...) to produce embeddings
        - Compute Euclidean distance between embeddings

    Args:
        extractor: Any object with a .transform(X) -> embedding_matrix method
                   (PCAFeatureExtractor, HandcraftedFeaturesExtractor, CNNFeatureExtractor)
        X_raw: (N, D) flattened images (after preprocessing)
        n_trials: number of random comparisons to test
        random_state: reproducibility seed

    Returns:
        avg_time_per_pair (seconds), total_time (seconds)
    """
    rng = np.random.default_rng(random_state)
    N = X_raw.shape[0]

    start = perf_counter()

    for _ in range(n_trials):
        i, j = rng.integers(0, N, size=2)

        xi = X_raw[i:i+1]  # (1, D)
        xj = X_raw[j:j+1]  # (1, D)

        emb_i = extractor.transform(xi)  # shape (1, embedding_dim)
        emb_j = extractor.transform(xj)

        # Euclidean distance (value itself not needed)
        _ = np.linalg.norm(emb_i - emb_j)

    total_time = perf_counter() - start
    avg_time = total_time / n_trials

    return avg_time, total_time

def estimate_fusion_verification_time(
    pca_extractor,
    cnn_extractor,
    hand_extractor,
    X_raw,
    n_trials=500,
    random_state=42
):
    """
    Estimate the verification time for the fused (PCA + CNN + Handcrafted) embedding.

    Steps per trial:
      - choose two random images
      - compute PCA embedding for each
      - compute CNN embedding for each
      - compute Hand-crafted embedding for each
      - concatenate embeddings
      - compute Euclidean distance
    """
    rng = np.random.default_rng(random_state)
    N = X_raw.shape[0]

    start = perf_counter()

    for _ in range(n_trials):
        i, j = rng.integers(0, N, size=2)

        xi = X_raw[i:i+1]
        xj = X_raw[j:j+1]

        # Compute embeddings
        pca_i  = pca_extractor.transform(xi)
        pca_j  = pca_extractor.transform(xj)

        cnn_i  = cnn_extractor.transform(xi)
        cnn_j  = cnn_extractor.transform(xj)

        hand_i = hand_extractor.transform(xi)
        hand_j = hand_extractor.transform(xj)

        # Concatenate
        emb_i = np.concatenate([pca_i, cnn_i, hand_i], axis=1)
        emb_j = np.concatenate([pca_j, cnn_j, hand_j], axis=1)

        # Distance
        _ = np.linalg.norm(emb_i - emb_j)

    total_time = perf_counter() - start
    avg_time = total_time / n_trials

    return avg_time, total_time
