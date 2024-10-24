
import numpy as np

# Define activation functions
def ReLU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases for a simple neural network
def initialize_weights(input_size, hidden_layer_size, output_layer_size):
    W_input_hidden = np.random.randn(input_size, hidden_layer_size) * 0.01
    b_hidden = np.zeros((1, hidden_layer_size))
    
    W_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * 0.01
    b_output = np.zeros((1, output_layer_size))
    
    return W_input_hidden, b_hidden, W_hidden_output, b_output

def forward_propagation(X, W_input_hidden, b_hidden, W_hidden_output, b_output):
    Z_hidden=np.dot(X, W_input_hidden)  + b_hidden
    A_hidden= ReLU(Z_hidden)

    # Hidden layer to output layer
    Z_output = np.dot(A_hidden, W_hidden_output) + b_output
    A_output = sigmoid(Z_output)

    return A_hidden, A_output
    
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
    
    
# Backpropagation
def backpropagation(X, y_true, A_hidden, A_output, W_hidden_output, W_input_hidden):
    m = X.shape[0]
    
    # Gradients for output layer
    dZ_output = A_output - y_true
    dW_hidden_output = (1/m) * np.dot(A_hidden.T, dZ_output)
    db_output = (1/m) * np.sum(dZ_output, axis=0, keepdims=True)
    
    # Gradients for hidden layer
    dA_hidden = np.dot(dZ_output, W_hidden_output.T)
    dZ_hidden = dA_hidden * (A_hidden > 0)  # Derivative of ReLU
    dW_input_hidden = (1/m) * np.dot(X.T, dZ_hidden)
    db_hidden = (1/m) * np.sum(dZ_hidden, axis=0, keepdims=True)
    
    return dW_input_hidden, db_hidden, dW_hidden_output, db_output
    

# Update weights using gradient descent
def update_weights(W_input_hidden, b_hidden, W_hidden_output, b_output, dW_input_hidden, db_hidden, dW_hidden_output, db_output, learning_rate):
    W_input_hidden -= learning_rate * dW_input_hidden
    b_hidden -= learning_rate * db_hidden
    W_hidden_output -= learning_rate * dW_hidden_output
    b_output -= learning_rate * db_output
    
    return W_input_hidden, b_hidden, W_hidden_output, b_output


# Training the neural network
def train_neural_network(X_train, y_train, vocab_size, hidden_layer_size, output_layer_size, epochs=100, learning_rate=0.01):
    input_size = vocab_size
    W_input_hidden, b_hidden, W_hidden_output, b_output = initialize_weights(input_size, hidden_layer_size, output_layer_size)

    for epoch in range(epochs):
        # Forward propagation
        A_hidden, A_output = forward_propagation(X_train, W_input_hidden, b_hidden, W_hidden_output, b_output)
        
        # Compute loss
        loss = compute_loss(y_train, A_output)
        
        # Backpropagation
        dW_input_hidden, db_hidden, dW_hidden_output, db_output = backpropagation(X_train, y_train, A_hidden, A_output, W_hidden_output, W_input_hidden)
        
        # Update weights
        W_input_hidden, b_hidden, W_hidden_output, b_output = update_weights(
            W_input_hidden, b_hidden, W_hidden_output, b_output, 
            dW_input_hidden, db_hidden, dW_hidden_output, db_output, learning_rate)
        
        # Print loss for every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    
    return W_input_hidden, b_hidden, W_hidden_output, b_output

# Prediction function
def predict_neural_network(X, W_input_hidden, b_hidden, W_hidden_output, b_output):
    _, A_output = forward_propagation(X, W_input_hidden, b_hidden, W_hidden_output, b_output)
    predictions = np.where(A_output >= 0.5, 1, 0)
    return predictions
