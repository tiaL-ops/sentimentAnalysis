import re
def tokenize(str):
    return str.split()


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


# Example usage:
sample_texts = [
    "I loved this movie! It's so exciting.",
    "This movie was terrible, I hated it!",
    "The plot was great, but the acting was bad."
]

# Step 1: Preprocess all the sample texts
preprocessed_docs = [preprocess_text(text) for text in sample_texts]

# Step 2: Create the vocabulary
vocab = create_vocabulary(preprocessed_docs)
print(f"Vocabulary: {vocab}")

# Step 3 & 4: Vectorize all the documents
vectors = vectorize_all_docs(preprocessed_docs, vocab)
print("BoW Representation (Vectors):")
for i, vector in enumerate(vectors):
    print(f"Document {i+1}: {vector}")