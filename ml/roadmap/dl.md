# dl

MLP:

- Perceptron, MLP
- Функции активации (ReLU, Sigmoid, Tanh, GELU)
- Forward / Backprop
- Инициализация весов
- Dropout, BatchNorm
- Overfitting в нейросетях
- Computational graph
- Autograd
- Initialization (Xavier/He)
- Vanishing/exploding gradients
- практика:
- - MNIST / Fashion-MNIST, MLP → регуляризация → сравнение с логрегом

CNN:

- Convolution, Padding, Stride
- Pooling
- Архитектуры:
- - LeNet
- - AlexNet
- - VGG
- - ResNet (conceptually!)
- - EfficientNet
- - Feature maps interpretation
- Data Augmentation
- Transfer Learning
- Практика:
- - Своя CNN
- - Fine-tuning предобученной модели
- - CIFAR-10 MLP vs CNN vs ResNet (transfer learning)
- - Классификация изображений из реального мира

Последовательности и RNN:

- RNN (проблема затухающего градиента)
- Embeddings
- LSTM, GRU
- Teacher forcing
- Attention mechanism
- Padding, Masking
- Transformer (must-have)
- Positional encoding
- Sequence-to-sequence
- Проекты:
- - TF-IDF + Naive Bayes → LSTM → GRU
- - ARIMA → LSTM

DL-4. Transformers (современный стандарт):

- Self-Attention
- Multi-Head Attention
- residual connections
- layer normalization (pre-norm vs post-norm)
- Positional Encoding
- Encoder / Decoder
- BERT vs GPT
- Fine-tuning vs Feature extraction
- Tokenization (BPE, SentencePiece)
- KV-cache inference
- проекты
- - Text Classification с BERT IMDB / Toxic Comments / News classification
- - NER или Question Answering

DL-5. Продвинутые DL-темы (по желанию):

- Generative Models
- Autoencoders
- Variational Autoencoders (VAE)
- GANs (DCGAN)
- Diffusion models (современный стандарт)
- проекты:
- - Генерация изображений (MNIST / Faces)
- Self-Supervised Learning
- Contrastive learning (SimCLR)
- BYOL
- Masked modeling
- практика:
- - Обучение эмбеддингов без разметки
- Reinforcement Learning (DL + RL)
- DQN
- Policy Gradient
- Actor-Critic
- проект: CartPole / Atari

NLP + TOKENIZATION (ОЧЕНЬ НЕДООЦЕНЕННЫЙ СЛОЙ):

- Tokenization
- - BPE (Byte Pair Encoding)
- - SentencePiece
- - WordPiece
- BPE (Byte Pair Encoding)
- SentencePiece
- WordPiece
- next token prediction (autoregressive loss)
- cross-entropy loss
- Ты должен понимать:
- - почему LLM не работают с “словами”
- - как устроен vocab
- - почему UTF-8 bytes-based токенизация сейчас популярна

TRAINING LLM:

- next token prediction (autoregressive loss)
- cross-entropy loss
- learning rate warmup
- cosine decay
- gradient clipping
- layer norm placement
- dropout vs no dropout in LLMs
- dataset size vs model size vs compute-why bigger models generalize better
- PEFT methods (must-have in industry)LoRA, QLoRA adapters
- reward model
- PPO optimization
- DPO (Direct Preference Optimization)
- RLAIF (AI feedback instead of human)

RAG (Retrieval Augmented Generation):

- embeddings search (vector DB) (OpenAI / bge / e5)
- cosine similarity search
- chunking strategies
- reranking models
- Tools- FAISS WeaviatePinecone

ADVANCED LLM RESEARCH TOPICS:

- Mixture of Experts (MoE)
- sparse attention
- linear attention (Performer, etc.)
- 10.2 Long context
- RoPE scaling
- sliding window attention
- memory mechanisms
- 10.3 Multimodal LLMs
- text + image (CLIP-like embeddings)
- vision transformers
- diffusion + LLM hybrids

ПРОДАКШН ML:

- Save/load models
- Inference pipeline
- Model serving (FastAPI / TorchServe)
- Quantization
- Pruning
- ONNX export
- KV cache
- batching requests
- speculative decoding
- quantization (8-bit / 4-bit)
- vLLM
- TensorRT-LLM
- HuggingFace inference stack

Уметь:

- Вывести backprop вручную
- Написать linear layer с нуля на NumPy
- Реализовать SGD без PyTorch
- Понять математику attention
- Написать mini-transformer с нуля
- Объяснить bias/variance через формулы
- Разложить PCA через SVD вручную
- sklearn алгоритм руками
- понимаешь градиент как геометрию, а не “формулу”
- можешь объяснить loss function как вероятность
- умеешь дебажить обучение модели
- можешь переписать sklearn алгоритм руками
- знаешь почему transformer лучше RNN
- умеешь улучшить модель, а не просто “подобрать модель”
