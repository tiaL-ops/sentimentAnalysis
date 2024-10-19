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

sample_text = "I loved this movie! It's so exciting."
processed_tokens = preprocess_text(sample_text)
print(processed_tokens)