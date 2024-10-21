import math

# Step 1: Term Frequency (TF)
def compute_tf(document):
    
    tf_dict = {}
    total_terms = len(document)
    
    for word in document:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    
   
    for word in tf_dict:
        tf_dict[word] /= total_terms
    
    return tf_dict

# Step 2: Inverse Document Frequency (IDF)
def compute_idf(documents):

    N = len(documents)  
    idf_dict = {}
    word_doc_count = {}
    
  
    for document in documents:
        for word in set(document):  
            word_doc_count[word] = word_doc_count.get(word, 0) + 1
    
    
    for word, doc_count in word_doc_count.items():
        idf_dict[word] = math.log(N / (doc_count + 1)) 
    
    return idf_dict

# Step 3: TF-IDF Calculation
def compute_tf_idf(documents):
   
    tf_idf_documents = []
    idf = compute_idf(documents)  
    
    for document in documents:
        tf = compute_tf(document) 
        tf_idf = {}
        
       
        for word in tf:
            tf_idf[word] = tf[word] * idf.get(word, 0)
        
        tf_idf_documents.append(tf_idf)
    
    return tf_idf_documents

# Example Usage
documents = [
    "this is a sample document".split(),
    "this document is the second document".split(),
    "and this is the third one".split(),
]

tf_idf_documents = compute_tf_idf(documents)

for doc_index, tf_idf in enumerate(tf_idf_documents):
    print(f"Document {doc_index+1} TF-IDF:\n", tf_idf)
