```python

X = [[0, 0], [1, 1]]
y = [0,1]
clf = svm.SVC(
  С=1,# параметр регуляризации
  kernel='linear' # ядро ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
  degree=3,# степень для poly
  decision_function_shape='ovo',# для многоклассовой классификации ovo - one vs one, ovr - one vs rest
  gamma='scale' # 'auto'
)
clf.fit(X, y) # загрузка данных
p = clf.predict([[2., 2.]]) # получение результата
print(p)
# [1]

clf.class_weight_
clf.asses_
clf.coef_ #
clf.ual_coef_
clf.fit_status_
clf.intercept_
clf.n_features_in_
clf.feature_names_in_
clf.n_iter_
clf.support_
clf.support_vectors_
clf.n_support_
clf.probA_
clf.probB_
clf.shape_fit_
# методы
clf.decision_function #
clf.decision_function
clf.fit
clf.get_metadata_routing
clf.get_params
clf.predict
clf.predict_log_proba
clf.predict_proba
clf.score
clf.set_fit_request
clf.set_params
clf.set_score_request
clf.
```
