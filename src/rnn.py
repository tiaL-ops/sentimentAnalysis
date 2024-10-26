import numpy as np
from activation import tanh

def forward_pass(sentence, one_hot_vectors, Wx, Wh, Wo, hidden_state, vocab_size):
    """Run forward pass through an RNN model."""
    for word in sentence:
        word_vector = np.zeros((vocab_size, 1))  
        word_vector[one_hot_vectors[word]] = 1  # One-hot vector for the word

        # Update hidden state
        hidden_state = tanh(np.dot(Wx, word_vector) + np.dot(Wh, hidden_state))

    # Compute output
    output = np.dot(Wo, hidden_state)
    return output, hidden_state