from umap import UMAP
from sklearn.base import BaseEstimator, TransformerMixin

class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.random_state is not None:
            # Ensures reproducibility but disables parallelism (n_jobs=1)
            self.umap_ = UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                n_jobs=1  # Forced by UMAP anyway when random_state is set
            )
        else:
            # Allows parallelism (n_jobs can be >1 if set externally or by default)
            self.umap_ = UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_jobs=-1  # Use all processors
            ) # random_state is not set, enabling parallelism
        self.umap_.fit(X)
        return self

    def transform(self, X):
        return self.umap_.transform(X)
