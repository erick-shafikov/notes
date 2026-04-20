# ml

Линейные модели и обобщения:

- [] Linear Regression (с разными функциями потерь, Ridge, Lasso, Elastic Net)
- Градиентные методы - Gradient Descent, SGD, Momentum, RMSProp, Adam / AdamW, Learning rate scheduling
- Generalized Linear Models (GLM) — расширение регрессии и классификации.
- Softmax / Multinomial Logistic Regression — для многоклассовых задач.
- метрики Accuracy, Precision / Recall / F1, ROC-AUC, PR-AUC, LogLoss, Calibration curves, Ranking metrics
- Multicollinearity, Feature scaling влияние, Probabilistic interpretation
- Алгоритмы - k-NN, Naive Bayes, Decision Trees,
- Регуляризация - L1 / L2, Dropout, Early stopping, Weight decay vs L2, Bayesian regularization
- практика:
- - Проект 1. Датасет: [Titanic](https://www.kaggle.com/c/titanic) или [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) Методы: Логистическая регрессия, SVM(kernel trick), KNN, Bias-variance decomposition
- - Проект 2. [Датасет: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) Методы: Decision Tree vs Random Forest vs Gradient Boosting (XGBoost/LightGBM/CatBoost)

Оптимизация и обучение:

- Stochastic Variance Reduced Gradient (SVRG), AdaGrad — развитие SGD.
- Second-order методы (Newton’s method, LBFGS) — когда нужно быстрое сходимое обучение.

Продвинутые деревья и ансамбли:

- Random Forest — очень часто применяется на практике.
- Gradient Boosting (XGBoost, LightGBM, CatBoost) — де-факто стандарт в табличных данных.
- Extra Trees (Extremely Randomized Trees) — альтернатива RF.
- Байесовские методы:
- - Байесовские сети (Bayesian Networks) — для причинно-следственных моделей.
- - Latent Dirichlet Allocation (LDA) — тематическое моделирование.

Кластеризация и снижение размерности

- PCA, SVD
- t-SNE, UMAP — для визуализации и нелинейного уменьшения размерности.
- NMF (Non-negative Matrix Factorization) — часто для текстов.
- ICA (Independent Component Analysis) — разделение независимых сигналов.
  Методы ближайших соседей (расширения):
- Ball Tree, KD-Tree — для ускорения kNN.
- Metric Learning (Siamese networks, triplet loss) — улучшение поиска ближайших объектов.
- Проекты:
- - Датасет: MNIST (рукописные цифры). Методы: PCA, t-SNE, UMAP.
- - Датасет: изображения одежды (Fashion-MNIST) Методы: k-means, DBSCAN, Gaussian Mixture Models, иерархическая кластеризация.

Логические и вероятностные модели:

- CRF (Conditional Random Fields) — для последовательностей (NLP, биоинформатика).
- HMM (Hidden Markov Models) — временные ряды, распознавание речи.
  Временные ряды:
- ARIMA / SARIMA — классика анализа рядов.
- Prophet (от Meta) — практический инструмент.
- VAR (Vector AutoRegression) — когда несколько временных рядов.
- IoT
- Проекты:
- - [Датасет: Airline Passengers](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) Методы: ARIMA, Prophet, LSTM (если захочешь затронуть DL).

Нейросетевые методы (если планируешь в DL):

- Перцептрон, MLP — базовая основа.
- Bag of Words, TF-IDF
- CNN (свёрточные сети) — для картинок.
- RNN, LSTM, GRU — для последовательностей.
- Transformers (BERT, GPT, Vision Transformers,HuggingFace и др.) — современный стандарт в NLP.
- проекты:
- - Датасет: CIFAR-10 или MNIST. Методы: MLP → CNN (сравнить результаты)
- - Датасет: [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) Методы: Наивный Байес → LSTM/GRU → Transformer (BERT).

Гибридные и спец. техники:

- Self-supervised learning — популярный тренд (SimCLR, BYOL и др.).
- Semi-supervised learning — когда мало данных с метками.
- Active learning — выбор данных для разметки.
- Reinforcement Learning (RL, Q-learning, Policy Gradient, DQN) — если интересен AI-агент.

Современные ансамбли и практические алгоритмы:

- Stacking / Blending — объединение разных моделей
- Feature engineering + AutoML подходы
- Hyperparameter tuning (GridSearch, RandomSearch, Bayesian Optimization)
- проекты:
- - Соревнование Kaggle (Tabular Playground или реальный датасет) Методы: Stacking/Blending нескольких моделей (RF, GBM, нейросеть)

Дополнительные направления:

- Reinforcement Learning (Q-learning, Policy Gradient, DQN,CartPole, Atari)
- Semi-supervised / Self-supervised Learning (SimCLR, BYOL и др.)
- Metric Learning (Siamese networks, triplet loss)
- Causal ML (Bayesian Networks, Do-calculus)
- MLP (многослойный перцептрон)
- Методы: Q-learning, Policy Gradient, DQN.
- Проекты:
- - Задача: обучить агента играть в CartPole (OpenAI Gym).

# Инструменты

- PyTorch (рекомендую) или TensorFlow
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
