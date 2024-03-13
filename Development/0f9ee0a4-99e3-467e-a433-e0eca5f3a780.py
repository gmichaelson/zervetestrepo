svc_model = SVC(gamma="auto", probability=True)
pipeline_svc = Pipeline([("transform", transform_pipeline), ("svc_model", svc_model)])
parameters_svc = {"svc_model__C": [0.1, 0.3]}
svc_pipeline = GridSearchCV(
    pipeline_svc, parameters_svc, cv=5, scoring="roc_auc", verbose=3, n_jobs=6
)
svc_pipeline.fit(X, y.values.ravel())
print(svc_pipeline.best_params_)
del X
del transform_pipeline
