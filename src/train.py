#testing purpoes we iwill sklearn to split the text
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text,create_vocabulary,vectorize_all_docs
from data_loader import load_imdb_data,read_by_batches
from models import classPrior,likelihood,predict


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

