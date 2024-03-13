model_roc_dict = {
    "Random\nForest": rf_pipeline.best_score_,
    "Logistic\nRegression": logistic_pipeline.best_score_,
    "Support Vector\nClassifier": svc_pipeline.best_score_,
}
names = list(model_roc_dict.keys())
values = list(model_roc_dict.values())
fig = plt.figure(figsize=(3, 4))
plt.bar(range(len(model_roc_dict)), values, tick_label=names)
plt.show()
