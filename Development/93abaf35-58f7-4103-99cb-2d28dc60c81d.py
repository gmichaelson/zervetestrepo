logistic_reg = LogisticRegression(
    penalty="elasticnet", solver="saga", C=0.5, max_iter=1000
)
pipeline_log = Pipeline(
    [("transform", transform_pipeline), ("logistic_reg", logistic_reg)]
)
parameters_log = {"logistic_reg__l1_ratio": [0.1, 0.3, 0.8]}
logistic_pipeline = GridSearchCV(
    pipeline_log, parameters_log, cv=5, scoring="roc_auc", verbose=3, n_jobs=6
)
logistic_pipeline.fit(X, y.values.ravel())
print(logistic_pipeline.best_params_)
del X
del transform_pipeline
