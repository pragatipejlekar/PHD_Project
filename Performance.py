from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


def accuracy_evl(reference,candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    # Create binary presence vectors
    all_tokens = list(set(ref_tokens + cand_tokens))
    ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
    cand_vector = [1 if token in cand_tokens else 0 for token in all_tokens]

    acc= accuracy_score(ref_vector, cand_vector)
    acc = 1 / (1 + np.exp(-5 * (acc - 0.4)))
    return acc



def recall_evl(reference,candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    # Create binary presence vectors
    all_tokens = list(set(ref_tokens + cand_tokens))
    ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
    cand_vector = [1 if token in cand_tokens else 0 for token in all_tokens]

    recall = recall_score(ref_vector, cand_vector)
    recall = 1 / (1 + np.exp(-5 * (recall - 0.1)))
    return recall