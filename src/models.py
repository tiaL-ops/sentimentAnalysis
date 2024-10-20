

import math
from preprocess import preprocess_text
"""
Step 1: Calculate Class Priors:
Input: A list of labeled reviews (positive or negative).
Task: Calculate the prior probabilities for each class.
This is the probability of a document being positive or negative based on the entire training set.

"""
def classPrior(labels):
    count_positive = 0
    count_negative = 0
    
    for label in labels:
        if label == 'positive':
            count_positive += 1
        elif label == 'negative':
            count_negative += 1

    total = len(labels)
    
    prior_pos = count_positive / total  
    prior_neg = count_negative / total  
    
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


