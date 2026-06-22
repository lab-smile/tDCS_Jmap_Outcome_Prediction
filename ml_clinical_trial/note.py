from skopt.space import Integer, Real
from skopt import BayesSearchCV

search_spaces = {
    # 1. mRMR number of top features retained
    "mrmr__max_topk": Integer(10, 500),

    # 2. RBF kernel width (gamma in RBFSampler)
    "rbf__gamma": Real(1e-3, 1e1, prior="log-uniform"),

    # 3. Number of random Fourier features
    "rbf__n_components": Integer(100, 1000),
}

opt = BayesSearchCV(
    estimator=pipeline_generator(
        features, numerical_features, categorical_features, jmap_features
    ),
    search_spaces=search_spaces,
    n_iter=40,                      # suggested 30–60 iterations
    cv=3,
    scoring="balanced_accuracy",
    n_jobs=-1,
    random_state=42
)

opt.fit(X, y)

print("Best parameters:", opt.best_params_)
print("Best score:", opt.best_score_)
