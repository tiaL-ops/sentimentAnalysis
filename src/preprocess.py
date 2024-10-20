import re
def tokenize(text):
    return text.split()


def to_lowerCase(tokens):
    return [token.lower() for token in tokens]

def remove_punctuation(tokens):
    clean_token= [re.sub(r'[^\w\s]','',token) for token in tokens]     
    return clean_token

stop_words = {"i", "this", "so", "it's", "the", "is", "a", "an", "and"}

def remove_stopWords(tokens):
    return[token for token in tokens if token not in stop_words]

def preprocess_text(text):
    tokens=tokenize(text)
    tokens=to_lowerCase(tokens)
    tokens=remove_punctuation(tokens)
    tokens=remove_stopWords(tokens)

    return tokens

#Bacth of Word
def create_vocabulary(docs):
    vocab = set() 
    for tokens in docs:
        vocab.update(tokens)
    return sorted(vocab)


def vectorize_doc(tokens, vocab):
    vector = [0] * len(vocab)
    for token in tokens:
        if token in vocab:
            index = vocab.index(token)  
            vector[index] += 1 
    return vector

def vectorize_all_docs(docs, vocab):
    vectors = []
    for tokens in docs:
        vectors.append(vectorize_doc(tokens, vocab))
    return vectors

