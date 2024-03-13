rf = ensemble.RandomForestClassifier()
pipeline_rf = Pipeline([("transform", transform_pipeline), ("rf", rf)])
parameters_rf = {"rf__max_depth": [1, 3, 8], "rf__min_samples_leaf": [5, 50, 1000]}
rf_pipeline = GridSearchCV(
    pipeline_rf, parameters_rf, cv=5, scoring="roc_auc", verbose=3, n_jobs=6
)
rf_pipeline.fit(X, y.values.ravel())
print(rf_pipeline.best_params_)
