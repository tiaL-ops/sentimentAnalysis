import csv

import csv

def read_by_batches(file_path, batch_size, max_batches=None):
    """
    Reads data from a CSV file in batches and optionally prints them.
    
    Args:
    - file_path (str): Path to the CSV file.
    - batch_size (int): Number of rows to read in each batch.
    - max_batches (int, optional): Maximum number of batches to process. Default is None, meaning no limit.
    
    Returns:
    - List of batches, where each batch is a list of (review, sentiment) tuples.
    """
    batches = [] 
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  

        batch = []
        batch_count = 0

        for i, row in enumerate(reader):
            review = row[0]  
            sentiment = row[1]  
            
            clean_review = review.replace("<br />", " ").strip() 
            batch.append((clean_review, sentiment))  

            
            if (i + 1) % batch_size == 0:
                batch_count += 1
                batches.append(batch) 
                
              
                if max_batches is None or batch_count <= max_batches:
                    print(f"\n--- Batch {batch_count} ---")
                    for review, sentiment in batch:
                        print(f"Review: {review}")
                        print(f"Sentiment: {sentiment}\n")
                
                batch = []  

               
                if max_batches is not None and batch_count >= max_batches:
                    break

       
        if batch:
            batch_count += 1
            batches.append(batch)  
            if max_batches is None or batch_count <= max_batches:
                print(f"\n--- Final Batch {batch_count} ---")
                for review, sentiment in batch:
                    print(f"Review: {review}")
                    print(f"Sentiment: {sentiment}\n")

    return batches  



def load_imdb_data(file_path):
    reviews = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) 

        for row in reader:
            review = row[0]
            sentiment = row[1]
            reviews.append(review)
            labels.append('positive' if sentiment == 'positive' else 'negative')

    return reviews, labels

