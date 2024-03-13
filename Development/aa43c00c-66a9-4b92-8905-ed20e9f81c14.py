rf_probs = rf_pipeline.predict_proba(X)
rf_preds = rf_probs[:, 1]
rf_fpr, rf_tpr, rf_threshold = metrics.roc_curve(y, rf_preds)
rf_roc_auc = metrics.auc(rf_fpr, rf_tpr)
rf_fig = plt.figure(figsize=(3, 4))
plt.title("Receiver Operating Characteristic")
plt.plot(rf_fpr, rf_tpr, "b", label="AUC = %0.2f" % rf_roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
