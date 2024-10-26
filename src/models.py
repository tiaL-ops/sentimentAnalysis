

import math
from preprocess import preprocess_text
import numpy as np
"""
Step 1: Calculate Class Priors:
Input: A list of labeled reviews (positive or negative).
Task: Calculate the prior probabilities for each class.
This is the probability of a document being positive or negative based on the entire training set.

"""
def classPrior(y_train):
   
    smoothing_factor = 1  # Laplace smoothing factor
    total_samples = len(y_train)
    
    
    pos_count = sum(y_train)
    neg_count = total_samples - pos_count
    
    prior_pos = (pos_count + smoothing_factor) / (total_samples + 2 * smoothing_factor)
    prior_neg = (neg_count + smoothing_factor) / (total_samples + 2 * smoothing_factor)
    
    return prior_pos, prior_neg


"""
Step 2: Calculate Likelihoods:
Input: The Bag of Words vectors for all documents and their corresponding labels (positive or negative).
Task:
For each word in the vocabulary, calculate the likelihood of that word appearing in a positive review and a negative review.
"""
def likelihood(docs, labels, vocab):
    word_pos={word:0 for word in vocab}
    word_neg={word:0 for word in vocab}

    total_word_pos=0
    total_word_neg=0

    for i, tokens in enumerate(docs):
        label=labels[i]

        if label == " positive":
            for token in tokens: 
                word_pos[token]+=1
                total_word_pos+=1
        elif label == " negative":
            for token in tokens: 
                word_neg[token]+=1
                total_word_neg+=1


        vocab_size=len(vocab)
        likelihood_pos={word:(word_pos[word] + 1)/( vocab_size +  total_word_pos) for word in vocab}
        likelihood_neg={word:(word_neg[word] +1) /(vocab_size + total_word_neg) for word in vocab}


    
        return likelihood_pos, likelihood_neg
"""
Step 3: Make Predictions:
Input: A new review (document).
Task:
Convert the review into its Bag of Words vector.
Calculate the posterior probability for each class (positive and negative).

"""


def predict(review, vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg):
    
    log_prob_pos = math.log(prior_pos)
    log_prob_neg = math.log(prior_neg)
    
    
    for token in review:
        if token in vocab:
            log_prob_pos += math.log(likelihood_pos[token])
            log_prob_neg += math.log(likelihood_neg[token])
    
   
    if log_prob_pos > log_prob_neg:
        return 'positive'
    else:
        return 'negative'

    


"""
Step 4: Evaluate the Model:
Input: A test set of reviews and their true labels.
Task:
After making predictions for all reviews in the test set, compare them to the true labels.
Calculate performance metrics like accuracy, precision, and recall.

"""
def evaluate(test_docs, test_labels, review, vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg):
    total=len(test_labels)
    current=0

    for i, review in enumerate(test_docs):
        predicted=predict(review ,vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg)
        if predicted == test_labels[i]:
            current+=1
    
    accuracy= current/total
    return accuracy



def initialize_weights(input_size, hidden_layer_size, output_layer_size):
    """Initialize weights and biases for input, hidden, and output layers."""
    W_input_hidden = np.random.randn(input_size, hidden_layer_size) * 0.01
    b_hidden = np.zeros((1, hidden_layer_size))
    
    W_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * 0.01
    b_output = np.zeros((1, output_layer_size))
    
    return W_input_hidden, b_hidden, W_hidden_output, b_output


def forward_propagation(X, weights, biases, activations):
    """Perform forward propagation through the network."""
    A = X
    caches = []
    
    for i in range(len(weights) - 1):
        Z = np.dot(A, weights[i]) + biases[i]
        A = activations[i](Z)
        caches.append((A, Z))  
    
    Z_output = np.dot(A, weights[-1]) + biases[-1]
    A_output = activations[-1](Z_output)
    caches.append((A_output, Z_output))
    
    return A_output, caches


def backpropagation(X, y_true, caches, weights, activations_derivative):
    """Compute gradients for each layer using backpropagation."""
    m = X.shape[0]
    dW = []
    db = []
    dA = caches[-1][0] - y_true  

    for i in reversed(range(len(weights))):
        dZ = dA * activations_derivative[i](caches[i][1])
        dW.insert(0, (1 / m) * np.dot(caches[i-1][0].T, dZ) if i > 0 else np.dot(X.T, dZ))
        db.insert(0, (1 / m) * np.sum(dZ, axis=0, keepdims=True))
        dA = np.dot(dZ, weights[i].T)
    
    return dW, db

def compute_loss(y_true, y_pred):
    """Binary cross-entropy loss for binary classification."""
    m = y_true.shape[0]
    loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def initialize_rnn_weights(vocab_size, hidden_size, output_size):
    """Initialize weights for an RNN model."""
    np.random.seed(0)  
    Wx = np.random.randn(hidden_size, vocab_size) * 0.01  # Input to hidden weights
    Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden weights
    Wo = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output weights
    hidden_state = np.zeros((hidden_size, 1))  # Initialize hidden state to zeros
    return Wx, Wh, Wo, hidden_state


def one_hot_encoding(vocab):
    return {word: i for i, word in enumerate(vocab)}

def tanh(x):
    return np.tanh(x)


def calculate_accuracy(predictions, true_labels):
  
    predictions = np.array(predictions).flatten()
    true_labels = np.array(true_labels).flatten()
    
    if predictions.shape != true_labels.shape:
        raise ValueError("Shape mismatch: predictions and true_labels must have the same shape.")
    
  
    correct_predictions = np.sum(predictions == true_labels)
    accuracy = (correct_predictions / len(true_labels)) * 100
    return accuracy