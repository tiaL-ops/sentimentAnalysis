# **Sentiment Analysis from Scratch** (Ongoing)

This project classifies movie reviews from the IMDb dataset as **positive** or **negative**, with a focus on building sentiment analysis models from the ground up using basic tools like **numpy**. Every stage of the NLP pipeline, including data loading, preprocessing, feature extraction, and classification, is manually implemented to provide a deep understanding of AI model construction.

---

## **Key Features and Project Progress:** 
- **Manual Data Loading**: Custom parsing of IMDb movie reviews. ✅
- **Text Preprocessing**: Tokenization, stop word removal, and text cleaning. ✅
- **Feature Extraction**: Implemented both **Bag of Words (BoW)** and **TF-IDF** from scratch. ✅
- **Custom Models**: Built a **Naive Bayes classifier** and a basic **neural network** using only numpy.✅
- **Numerical Computation**: Leveraged numpy for matrix operations and computations.

---
## Knowledge Review (Day 1 - Day 6) 

### 1. **Project Setup (Day 1)**:
   - Established the folder structure for organizing data, code, and outputs.
   - Initialized a GitHub repository for version control and added a `README` to document the project workflow.

### 2. **Dataset Loading and Exploration (Day 2)**:
   - Loaded the IMDb dataset manually by parsing CSV files.
   - Developed functions to load data in smaller batches to handle memory limitations.
   - Printed and inspected a subset of data to understand its structure (text and sentiment labels).

### 3. **Text Preprocessing (Day 3)**:
   - Implemented text preprocessing from scratch, including:
     - **Tokenization**: Splitting text into individual words (tokens).
     - **Lowercasing**: Converting all text to lowercase for consistency.
     - **Punctuation Removal**: Cleaned out unwanted symbols using regular expressions.
     - **Stop Word Removal**: Defined a custom list of stop words and filtered them out.
   - Preprocessing prepares text for vectorization and model training.

### 4. **Bag of Words (BoW) Representation (Day 4)**:
   - Created a Bag of Words model to represent text as vectors based on word frequency.
   - BoW encodes each document as a vector of word counts without considering word order.
   - **Limitation**: BoW doesn't capture the context or importance of words, which can lead to less effective classification in some cases.

### 5. **Naive Bayes Classifier (Day 5)**:
   - Built a simple Naive Bayes classifier for sentiment analysis from scratch.
   - Key steps:
     - **Prior Probability**: Calculated the prior probability of positive and negative classes.
     - **Likelihood**: Calculated the likelihood of words occurring in each class.
     - **Laplace Smoothing**: Applied smoothing to avoid zero probabilities for unseen words.
   - Naive Bayes makes the "naive" assumption that features (words) are independent given the class label.

### 6. **TF-IDF (Term Frequency-Inverse Document Frequency) Representation (Day 6)**:
   - Implemented TF-IDF to improve text representation by weighting word importance.
     - **TF (Term Frequency)**: Counts how often a word appears in a document.
     - **IDF (Inverse Document Frequency)**: Measures how rare a word is across the entire corpus.
   - TF-IDF improves upon BoW by giving more weight to less frequent, but important words, and downweights very common words.
### 7. **Neural Network from Scratch (Day 7)**:
   - Built a simple neural network with an input layer, one hidden layer (using ReLU activation), and an output layer (using sigmoid activation for binary classification).
    - **Forward Propagation**: Passed input data through the network and applied activations.
    - **Binary Cross-Entropy Loss** : Used this loss function to measure prediction error between predicted probabilities and actual labels.
    - **Backpropagation**: Calculated gradients to update weights and biases by determining the contribution of each parameter to the error.
    - **Gradient Descent**: Optimized the network by adjusting weights and biases based on the learning rate.

### 8. **Reccurent Neural Network(RNN) from Scratch (Day 8)**:
- **Purpose**: RNNs (Recurrent Neural Networks) are designed to process sequential data, capturing context and relationships across elements (like words in a sentence) by maintaining a "memory" of previous inputs.
- **Hidden State**: As each element in the sequence is processed, the RNN updates its hidden state, which retains information about prior inputs, allowing it to "remember" context as it moves through the sequence.
- **Weights**: The RNN uses two primary weight matrices, Wx (input-to-hidden) and Wh (hidden-to-hidden), to update the hidden state. These weights help integrate the new input with past information.
- **Output**: After processing each element, the final hidden state acts as a summary of the entire sequence. In applications like sentiment analysis, this final state is typically passed through a dense layer to predict a label (e.g., positive or negative sentiment).
- **Backpropagation Through Time (BPTT)**: RNNs are trained using BPTT, which computes gradients across time steps. However, RNNs can encounter challenges like vanishing/exploding gradients, especially on long sequences, which may require additional techniques like gradient clipping or using more advanced RNN variants (e.g., LSTM or GRU).