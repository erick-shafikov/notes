# ml

Линейные модели и обобщения:

- [x] [Linear Regression](../ml/intro/2-linear-model.md) (с разными функциями потерь, [Ridge](../ml/l2-l1-reg/1-l2-reg.md), [Lasso](../ml/l2-l1-reg/2-l1-reg.md), [Elastic Net](../ml/l2-l1-reg/3-l1-l2-elastic-net.md))
- [x] Градиентные методы — [Gradient Descent](../ml/gradient-regression/1-gradient-descent.md), [SGD и Learning rate scheduling](../ml/gradient-regression/2-sgd.md), [Momentum, RMSProp, Adam / AdamW](../ml/gradient-regression/4-grad-optimiztion.md)
- [x] [Generalized Linear Models (GLM)](../ml/probabilistic-methods/6-glm.md) — расширение регрессии и классификации.
- [x] [Stochastic Variance Reduced Gradient (SVRG)](../ml/gradient-regression/5-svrg.md), AdaGrad — развитие SGD.
- Second-order методы ([Newton’s method](../ml/gradient-regression/2-sgd.md), LBFGS) — когда нужно быстрое сходимое обучение.
- [x] [Softmax / Multinomial Logistic Regression](../ml/probabilistic-methods/1-mlm.md) — для многоклассовых задач.
- [x] метрики [Accuracy](../ml/knn/1-multiclass.md), [Precision / Recall / F1](../ml/metrics/2-precision-recall.md), [ROC-AUC, PR-AUC](../ml/metrics/3-roc-curve.md), [LogLoss](../ml/metrics/4-logloss.md), [Calibration curves](../ml/metrics/1-calibration.md), [Ranking metrics](../ml/metrics/5-ranking-metrics.md)
- [x] [Multicollinearity](../ml/pca-svd/2-pca-regression.md), [Feature scaling влияние](../ml/intro/1-intro.md), [Probabilistic interpretation](../ml/probabilistic-methods/3-l1-l2-prob.md)
- [x] Алгоритмы — [k-NN](../ml/knn/2-knn.md), [Naive Bayes](../ml/probabilistic-methods/2-naive-bayes.md), [Decision Trees](../ml/logic-classification/1-decision-trees.md)
- [x] Регуляризация — [L1](../ml/l2-l1-reg/2-l1-reg.md) / [L2](../ml/l2-l1-reg/1-l2-reg.md), [Dropout, Early stopping](../ml/intro/3-overfitting.md), [Weight decay vs L2](../ml/l2-l1-reg/1-l2-reg.md), [Bayesian regularization](../ml/probabilistic-methods/3-l1-l2-prob.md)
- практика:
- - Проект 1. Датасет: [Titanic](https://www.kaggle.com/c/titanic) или [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) Методы: [Логистическая регрессия](../ml/probabilistic-methods/1-mlm.md), [SVM (kernel trick)](../ml/svm/2-svm-nonlinear-kernel.md), [KNN](../ml/knn/2-knn.md), [Bias-variance decomposition](../ml/intro/3-overfitting.md)
- - [Датасет: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) Методы: [Decision Tree](../ml/logic-classification/1-decision-trees.md) vs [Random Forest](../ml/logic-classification/3-rnd-forest.md) vs Gradient Boosting (XGBoost/LightGBM/CatBoost)

Продвинутые деревья и ансамбли:

- [x] [Random Forest](../ml/logic-classification/3-rnd-forest.md) — очень часто применяется на практике.
- Gradient Boosting (XGBoost, LightGBM, CatBoost) — де-факто стандарт в табличных данных.
- Extra Trees (Extremely Randomized Trees) — альтернатива RF.
- Байесовские методы:
- - Байесовские сети (Bayesian Networks) — для причинно-следственных моделей.
- - Latent Dirichlet Allocation (LDA) — тематическое моделирование.

Кластеризация и снижение размерности

- [x] [PCA](../ml/pca-svd/1-pca.md), [SVD](../ml/pca-svd/3-svd.md)
- t-SNE, UMAP — для визуализации и нелинейного уменьшения размерности.
- NMF (Non-negative Matrix Factorization) — часто для текстов.
- ICA (Independent Component Analysis) — разделение независимых сигналов.

Методы ближайших соседей (расширения):

- Ball Tree, KD-Tree — для ускорения kNN.
- Metric Learning (Siamese networks, triplet loss) — улучшение поиска ближайших объектов.
- Проекты:
- - Датасет: MNIST (рукописные цифры). Методы: [PCA](../ml/pca-svd/1-pca.md), t-SNE, UMAP.
- - Датасет: изображения одежды (Fashion-MNIST) Методы: [k-means](../ml/clustering/1-k-means-lloyd.md), [DBSCAN](../ml/clustering/2-cluster-dbscan.md), Gaussian Mixture Models, [иерархическая кластеризация](../ml/clustering/3-agglomerative-clustering.md).

Логические и вероятностные модели:

- CRF (Conditional Random Fields) — для последовательностей (NLP, биоинформатика).
- HMM (Hidden Markov Models) — временные ряды, распознавание речи.

Временные ряды:

- ARIMA / SARIMA — классика анализа рядов.
- Prophet (от Meta) — практический инструмент.
- VAR (Vector AutoRegression) — когда несколько временных рядов.
- IoT
- Проекты:
- - [Датасет: Airline Passengers](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) Методы: ARIMA, Prophet

Гибридные и спец. техники:

- Self-supervised learning — популярный тренд (SimCLR, BYOL и др.).
- Semi-supervised learning — когда мало данных с метками.
- Active learning — выбор данных для разметки.
- Reinforcement Learning (RL, Q-learning, Policy Gradient, DQN, Atari) — если интересен AI-агент.

Современные ансамбли и практические алгоритмы:

- Stacking / Blending — объединение разных моделей
- Feature engineering + AutoML подходы
- Hyperparameter tuning (GridSearch, RandomSearch, Bayesian Optimization)
- проекты:
- - Соревнование Kaggle (Tabular Playground или реальный датасет) Методы: Stacking/Blending нескольких моделей (RF, GBM, нейросеть)

Дополнительные направления:

- Causal ML (Do-calculus)
- Проекты:
- - Задача: обучить агента играть в CartPole (OpenAI Gym).

# Инструменты

- PyTorch, TensorFlow
- - nn.Module
- - Dataset / DataLoader
- - Optimizers
- - Custom layers
- - Custom autograd function
- - Mixed precision training
- - GPU memory optimization
- Autograd (граф вычислений)
- GPU / CUDA basics
- Batch, Epoch, Learning Rate
- Практика
- - Реализовать линейную регрессию и логрег на PyTorch
- - Написать training loop вручную
- - реализовать backprop концептуально
- - debug training instability
