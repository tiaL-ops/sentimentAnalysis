import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text, create_vocabulary, vectorize_all_docs
from data_loader import read_by_batches
from models import classPrior, likelihood, predict
from tf_idf import compute_tf_idf
from neuralNetwork import train_neural_network, predict_neural_network
from models import initialize_rnn_weights,one_hot_encoding, calculate_accuracy
from rnn import forward_pass


# Data loading and preprocessing
def load_and_preprocess_data(file_path, batch_size, max_batches):
    data_batches = read_by_batches(file_path, batch_size, max_batches)
    flattened_data = [item for batch in data_batches for item in batch]
    reviews = [item[0] for item in flattened_data]
    labels = [item[1] for item in flattened_data]
    
    processed_reviews = [preprocess_text(review) for review in reviews]
    vocab = create_vocabulary(processed_reviews)
    
    label_map = {'positive': 1, 'negative': 0}
    labels = [label_map[label] for label in labels]
    
    return processed_reviews, labels, vocab


# Naive Bayes with BoW
def evaluate_naive_bayes_bow(X_train, X_test, y_train, y_test, vocab):
    prior_pos, prior_neg = classPrior(y_train)
    likelihood_pos, likelihood_neg = likelihood(X_train, y_train, vocab)
    
    predictions = [predict(review, vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg) for review in X_test]
    accuracy = calculate_accuracy(np.array(predictions), np.array(y_test))
    
    print(f"Naive Bayes BoW Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Naive Bayes with TF-IDF
def evaluate_naive_bayes_tfidf(X_train, X_test, y_train, y_test, vocab):
    prior_pos, prior_neg = classPrior(y_train)
    likelihood_pos, likelihood_neg = likelihood(X_train, y_train, vocab)
    
    predictions = [predict(review, vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg) for review in X_test]
    accuracy = calculate_accuracy(np.array(predictions), np.array(y_test))
    
    print(f"Naive Bayes TF-IDF Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Convert TF-IDF dict to vectors
def convert_dicts_to_vectors(tfidf_docs, vocab):
    vectors = []
    for doc in tfidf_docs:
        vector = [doc.get(word, 0) for word in vocab]
        vectors.append(vector)
    return np.array(vectors)


# Neural Network Evaluation
def evaluate_neural_network(X_train, X_test, y_train, y_test, vocab_size, hidden_layer_size=32, output_layer_size=1, epochs=100, learning_rate=0.01):
    W_input_hidden, b_hidden, W_hidden_output, b_output = train_neural_network(X_train, y_train, vocab_size, hidden_layer_size, output_layer_size, epochs, learning_rate)
    predictions = predict_neural_network(X_test, W_input_hidden, b_hidden, W_hidden_output, b_output)
    
    accuracy = calculate_accuracy(predictions, y_test)
    print(f"Neural Network Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# RNN Evaluation
def evaluate_rnn(processed_reviews, labels, vocab, hidden_size=5):
    one_hot_vectors = one_hot_encoding(vocab)
    vocab_size = len(vocab)
    Wx, Wh, Wo, hidden_state = initialize_rnn_weights(vocab_size, hidden_size, output_size=1)

    predictions = []
    for review in processed_reviews:
        output, _ = forward_pass(review, one_hot_vectors, Wx, Wh, Wo, hidden_state, vocab_size)
        prediction = 1 if output >= 0.5 else 0
        predictions.append(prediction)

    accuracy = calculate_accuracy(np.array(predictions), np.array(labels))
    print(f"RNN Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    # Parameters
    file_path = '../data/IMDB Dataset.csv'
    batch_size = 10
    max_batches = 5

    # Load and preprocess data
    processed_reviews, labels, vocab = load_and_preprocess_data(file_path, batch_size, max_batches)
    X = vectorize_all_docs(processed_reviews, vocab)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Naive Bayes BoW
    print("Evaluating Naive Bayes with BoW...")
    bow_accuracy = evaluate_naive_bayes_bow(X_train, X_test, y_train, y_test, vocab)
    
    # TF-IDF Transformation and Naive Bayes
    print("Evaluating Naive Bayes with TF-IDF...")
    X_tfidf = compute_tf_idf(processed_reviews)
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
    X_train_tfidf = convert_dicts_to_vectors(X_train_tfidf, vocab)
    X_test_tfidf = convert_dicts_to_vectors(X_test_tfidf, vocab)
    tfidf_accuracy = evaluate_naive_bayes_tfidf(X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, vocab)
    
    # Neural Network Evaluation
    print("Evaluating Neural Network...")
    X_train_nn = np.array(X_train_tfidf)
    y_train_nn = np.array(y_train_tfidf, dtype=float).reshape(-1, 1)
    X_test_nn = np.array(X_test_tfidf)
    y_test_nn = np.array(y_test_tfidf, dtype=float).reshape(-1, 1)
    nn_accuracy = evaluate_neural_network(X_train_nn, X_test_nn, y_train_nn, y_test_nn, len(vocab))

    # RNN Evaluation
    print("Evaluating RNN...")
    rnn_accuracy = evaluate_rnn(processed_reviews, labels, vocab)

    # Summary of results
    print(f"\nModel Performance Summary:")
    print(f"BoW Accuracy: {bow_accuracy * 100:.2f}%")
    print(f"TF-IDF Accuracy: {tfidf_accuracy * 100:.2f}%")
    print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")
    print(f"RNN Accuracy: {rnn_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
