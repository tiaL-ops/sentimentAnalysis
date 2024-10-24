#testing purpoes we iwill sklearn to split the text
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text,create_vocabulary,vectorize_all_docs
from data_loader import load_imdb_data,read_by_batches
from models import classPrior,likelihood,predict
from tf_idf import compute_tf_idf
import numpy as np
from neuralNetwork import train_neural_network, predict_neural_network


# Load the data in batches because my computer is old
file_path = '../data/IMDB Dataset.csv'  
batch_size = 10 
max_batches = 5 

data_batches = read_by_batches(file_path, batch_size, max_batches)


flattened_data = [item for batch in data_batches for item in batch]


reviews = [item[0] for item in flattened_data]  

labels = [item[1] for item in flattened_data]  


processed_reviews = [preprocess_text(review) for review in reviews]


vocab = create_vocabulary(processed_reviews)



X = vectorize_all_docs(processed_reviews, vocab)
label_map = {'positive': 1, 'negative': 0}
labels = [label_map[label] for label in labels]

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

prior_pos, prior_neg = classPrior(y_train)
likelihood_pos, likelihood_neg = likelihood(X_train, y_train, vocab)

predictions = []
for review in X_test:
    prediction = predict(review, vocab, likelihood_pos, likelihood_neg, prior_pos, prior_neg)
    predictions.append(prediction)

correct_predictions = sum([1 for i in range(len(predictions)) if predictions[i] == y_test[i]])
accuracy = correct_predictions / len(y_test)

print(f"Accuracy of Naive Bayes model: {accuracy * 100:.2f}%")


#Tfidt Approach
X_tfidf= compute_tf_idf(processed_reviews)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

def convert_dicts_to_vectors(tfidf_docs, vocab):
    vectors = []
    for doc in tfidf_docs:
        vector = [doc.get(word, 0) for word in vocab]
        vectors.append(vector)
    return vectors

X_train_tfidf = convert_dicts_to_vectors(X_train_tfidf, vocab)
X_test_tfidf = convert_dicts_to_vectors(X_test_tfidf, vocab)


prior_pos_tfidf, prior_neg_tfidf = classPrior(y_train_tfidf)
likelihood_pos_tfidf, likelihood_neg_tfidf = likelihood(X_train_tfidf, y_train_tfidf, vocab)

predictions_tfidf = [predict(review, vocab, likelihood_pos_tfidf, likelihood_neg_tfidf, prior_pos_tfidf, prior_neg_tfidf) for review in X_test_tfidf]

correct_predictions_tfidf = sum([1 for i in range(len(predictions_tfidf)) if predictions_tfidf[i] == y_test_tfidf[i]])
accuracy_tfidf = correct_predictions_tfidf / len(y_test_tfidf)



## NeuralNetwork:
# Set the neural network parameters
hidden_layer_size = 32
output_layer_size = 1
epochs = 100
learning_rate = 0.01
# Convert the dataset into numpy arrays
X_train = np.array(X_train_tfidf)
y_train = np.array(y_train, dtype=float).reshape(-1, 1) 

# Set the neural network parameters
hidden_layer_size = 32
output_layer_size = 1
epochs = 100
learning_rate = 0.01

# Train the neural network
W_input_hidden, b_hidden, W_hidden_output, b_output = train_neural_network(X_train, y_train, len(vocab), hidden_layer_size, output_layer_size, epochs, learning_rate)


# Convert test set to numpy array
X_test = np.array(X_test_tfidf)

# Predict sentiment on test data
predictions = predict_neural_network(X_test, W_input_hidden, b_hidden, W_hidden_output, b_output)

# Calculate accuracy
y_test = np.array(y_test, dtype=float).reshape(-1, 1)
correct_predictions = np.sum(predictions == y_test)
accuracy = correct_predictions / len(y_test)



print(f"Comparison of Naive Bayes Model Performance:")
print(f"BoW Accuracy: {accuracy* 100:.2f}%")
print(f"TF-IDF Accuracy: {accuracy_tfidf * 100:.2f}%")
print(f"Neural Network Accuracy: {accuracy * 100:.2f}%")




