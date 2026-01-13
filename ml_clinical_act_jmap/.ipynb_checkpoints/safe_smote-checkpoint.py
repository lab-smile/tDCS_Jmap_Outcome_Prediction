# safe_smote.py
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

class SafeSMOTE(SMOTE):
    def fit_resample(self, X, y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        counts = np.bincount(y_enc)
        if (counts == 0).any() or counts.min() == 0:
            # a class missing in this fold → nothing to do
            return X, y
        maj = counts.max()
        minc = counts.min()
        # If minority already >= majority, oversampling not needed/possible
        if minc >= maj:
            return X, y
        # Guard k_neighbors > minc-1
        if hasattr(self, "k_neighbors"):
            k = getattr(self.k_neighbors, "n_neighbors", self.k_neighbors)
            if minc - 1 < k:
                # reduce effective neighbors to avoid another SMOTE error
                self.k_neighbors = max(1, minc - 1)
        return super().fit_resample(X, y)