import numpy as np
from activation import ReLU, sigmoid, ReLU_derivative
from models import initialize_weights,compute_loss


def forward_propagation(X, W_input_hidden, b_hidden, W_hidden_output, b_output):
    """Forward pass through the network."""
    Z_hidden = np.dot(X, W_input_hidden) + b_hidden
    A_hidden = ReLU(Z_hidden)
    Z_output = np.dot(A_hidden, W_hidden_output) + b_output
    A_output = sigmoid(Z_output)
    return A_hidden, A_output
    
def backpropagation(X, y_true, A_hidden, A_output, W_hidden_output, W_input_hidden):
    """Calculate gradients for weights and biases using backpropagation."""
    m = X.shape[0]
    dZ_output = A_output - y_true
    dW_hidden_output = (1/m) * np.dot(A_hidden.T, dZ_output)
    db_output = (1/m) * np.sum(dZ_output, axis=0, keepdims=True)
    
    dA_hidden = np.dot(dZ_output, W_hidden_output.T)
    dZ_hidden = dA_hidden * ReLU_derivative(A_hidden)
    dW_input_hidden = (1/m) * np.dot(X.T, dZ_hidden)
    db_hidden = (1/m) * np.sum(dZ_hidden, axis=0, keepdims=True)
    
    return dW_input_hidden, db_hidden, dW_hidden_output, db_output

def update_weights(W_input_hidden, b_hidden, W_hidden_output, b_output, dW_input_hidden, db_hidden, dW_hidden_output, db_output, learning_rate):
    """Update weights and biases using gradient descent."""
    W_input_hidden -= learning_rate * dW_input_hidden
    b_hidden -= learning_rate * db_hidden
    W_hidden_output -= learning_rate * dW_hidden_output
    b_output -= learning_rate * db_output
    return W_input_hidden, b_hidden, W_hidden_output, b_output

def train_neural_network(X_train, y_train, vocab_size, hidden_layer_size, output_layer_size, epochs=100, learning_rate=0.01):
    """Train the neural network model."""
    W_input_hidden, b_hidden, W_hidden_output, b_output = initialize_weights(vocab_size, hidden_layer_size, output_layer_size)

    for epoch in range(epochs):
        A_hidden, A_output = forward_propagation(X_train, W_input_hidden, b_hidden, W_hidden_output, b_output)
        loss = compute_loss(y_train, A_output)
        dW_input_hidden, db_hidden, dW_hidden_output, db_output = backpropagation(X_train, y_train, A_hidden, A_output, W_hidden_output, W_input_hidden)
        
        W_input_hidden, b_hidden, W_hidden_output, b_output = update_weights(
            W_input_hidden, b_hidden, W_hidden_output, b_output, 
            dW_input_hidden, db_hidden, dW_hidden_output, db_output, learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    
    return W_input_hidden, b_hidden, W_hidden_output, b_output

def predict_neural_network(X, W_input_hidden, b_hidden, W_hidden_output, b_output):
    """Predict classes using the trained neural network."""
    _, A_output = forward_propagation(X, W_input_hidden, b_hidden, W_hidden_output, b_output)
    predictions = np.where(A_output >= 0.5, 1, 0)
    return predictions
