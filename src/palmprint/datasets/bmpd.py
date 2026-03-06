import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Sample:
    img_path: str
    label: int          # subject label (0 .. N-1)
    subject_id: str     # original folder name ("001", "002", ...)

def get_subject_ids(root_dir: str) -> List[str]:
    """
    Return sorted list of subject folder names: ["001", "002", ..., "041"].
    """
    subject_ids = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    subject_ids.sort()
    return subject_ids

def split_subjects(
    subject_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split SUBJECTS (NOT IMAGES!!!) into train/val/test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    subject_ids = subject_ids.copy()
    random.Random(seed).shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    n_test  = n - n_train - n_val

    return {
        "train": subject_ids[:n_train],
        "val":   subject_ids[n_train:n_train+n_val],
        "test":  subject_ids[n_train+n_val:]
    }

def collect_samples_for_subjects(
    root_dir: str,
    subjects: List[str],
    subject_to_label: Dict[str, int]
) -> List[Sample]:
    """
    Collect (image_path, label) pairs for each subject folder.
    """
    samples = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heif", ".heic")

    for sid in subjects:
        subj_dir = os.path.join(root_dir, sid)
        label = subject_to_label[sid]

        for fname in os.listdir(subj_dir):
            if fname.lower().endswith(valid_ext):
                img_path = os.path.join(subj_dir, fname)
                samples.append(Sample(
                    img_path=img_path,
                    label=label,
                    subject_id=sid
                ))
    return samples

def build_bmpd_splits(root_dir: str, seed: int = 42) -> Dict[str, List[Sample]]:
    """
    Full processing:
      - scan subject folders
      - split subject list into train/val/test
      - collect image paths for each split
      - map subjects → integer labels (0 .. N-1)
    """
    subject_ids = get_subject_ids(root_dir)
    subject_to_label = {sid: i for i, sid in enumerate(subject_ids)}

    splits = split_subjects(subject_ids, seed=seed)

    return {
        "train": collect_samples_for_subjects(root_dir, splits["train"], subject_to_label),
        "val":   collect_samples_for_subjects(root_dir, splits["val"],   subject_to_label),
        "test":  collect_samples_for_subjects(root_dir, splits["test"],  subject_to_label),
    }



