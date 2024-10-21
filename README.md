# Sentiment Analysis

This project implements a deep learning-based approach for sentiment analysis on movie reviews. Using LSTM-based Recurrent Neural Networks (RNN), the model is trained to classify movie reviews as positive or negative. The project makes use of the IMDB dataset to train and evaluate the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to classify movie reviews as positive or negative using **Natural Language Processing (NLP)** and a deep learning model based on **LSTM (Long Short-Term Memory)**. It uses word embeddings with **Word2Vec** for text representation, and the model is trained and evaluated on the **IMDB dataset**.

## Features

* **Text Preprocessing:** Includes tokenization, stopword removal, and lemmatization.
* **Word Embeddings:** Generates word embeddings using Word2Vec.
* **Sentiment Classification:** Trains an LSTM model to classify the sentiment of movie reviews as positive or negative.
* **Model Evaluation:** Evaluates the model on the test set and provides accuracy metrics.

## System Requirements

+ **Python:** 3.6 or higher
+ **TensorFlow:** 2.x
+ **Keras:** 2.x
+ **NLTK:** Natural Language Toolkit (for preprocessing)
+ **Gensim:** For Word2Vec embeddings


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guruttamsv/Sentiment-Analysis.git
cd Sentiment-Analysis
```
2. Set up a virtual environment (optional but recommended):
```bash
# Using conda
conda create -n sentiment-analysis python=3.8
conda activate sentiment-analysis
```
3. Install required packages:
```bash
pip install tensorflow keras nltk gensim pandas numpy
```
4. Download the NLTK stopwords and wordnet datasets:
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Dataset

The dataset used is the IMDB Movie Reviews dataset, which contains 50,000 labeled movie reviews. It is split into:

+ **Training Set:** Contains 25,000 movie reviews for training.
+ **Test Set:** Contains 25,000 movie reviews for testing.

The dataset can be downloaded from the following link:
```bash
!wget -P /content/sample_data/ -c "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
```
After downloading, the dataset is extracted and structured into pos (positive) and neg (negative) directories for both training and testing.

## Model Architecture

The model is based on an **LSTM** architecture, which is particularly effective for sequence-based tasks like sentiment analysis.

+ **Embedding Layer:** The model uses pre-trained word embeddings generated via Word2Vec.
+ **Bidirectional LSTM Layer:** Helps capture both past and future context.
+ **Dropout Layers:** Added to prevent overfitting.
+ **Dense Layer:** The output is a single neuron with a sigmoid activation function for binary classification.

### Model Summary:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        input_dim = trigram_model.wv.vectors.shape[0],
        output_dim = trigram_model.wv.vectors.shape[1],
        input_length = input_length,
        weights = [trigram_model.wv.vectors],
        trainable=False
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, recurrent_dropout=0.1)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

```
## Training and Evaluation
The model is trained with the following parameters:

+ **Optimizer:** Adam
+ **Loss Function:** Binary Crossentropy (for binary classification)
+ **Metrics:** Accuracy

```python

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

```

The training loop includes:
+ **Batch Size:** 100
+ **Epochs:** 2 (more epochs may improve accuracy)

```python

model.fit(X_train, y_train, batch_size=100, epochs=2, validation_data=(X_test, y_test))

```
After training, the model is evaluated on the test set to determine its performance:
```python

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

## Results

+ **Training Accuracy:** The model reaches around 85-90% accuracy during training.
+ **Test Accuracy:** The test accuracy can range between 80-90%, depending on the size of the dataset and hyperparameters.

The model can be further improved by tuning hyperparameters such as the number of epochs, batch size, and the architecture of the LSTM layers.

## Limitations and Future Work

### Limitations
* **Data Size:** The accuracy depends heavily on the size and quality of the dataset. Using more data may improve results.
* **Simple Architecture:** The model uses a simple LSTM architecture, which could be improved with more advanced models like GRU or BERT.

### Future Work
* Experiment with more advanced architectures like **GRU** or **transformers (BERT)**.
* Use larger datasets or employ data augmentation techniques to improve performance.
* Fine-tune word embeddings using transfer learning for better results.

## Acknowledgements

* **TensorFlow and Keras:** For providing the framework for deep learning.
* **NLTK:** For providing tools for text preprocessing.
* **Gensim:** For generating word embeddings using Word2Vec.
* **Google Colab:** For providing an accessible platform with GPU support.

