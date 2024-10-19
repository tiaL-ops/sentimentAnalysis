import csv

def read_by_batches(file_path, batch_size, max_batches=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  

        batch = []
        batch_count = 0 
        
        for i, row in enumerate(reader):
            review = row[0]  # First column is the review text
            sentiment = row[1]  # Second column is the sentiment (positive/negative)
            
            clean_review = review.replace("<br />", " ").strip()  
            batch.append((clean_review, sentiment))  #

            # Process the batch when it reaches the batch size
            if (i + 1) % batch_size == 0:
                batch_count += 1
           
                if max_batches is None or batch_count <= max_batches:
                    print(f"\n--- Batch {batch_count} ---")
                    for review, sentiment in batch:
                        print(f"Review: {review}")
                        print(f"Sentiment: {sentiment}\n")

                batch = [] 

        if batch:
            batch_count += 1
            if max_batches is None or batch_count <= max_batches:
                print(f"\n--- Final Batch {batch_count} ---")
                for review, sentiment in batch:
                    print(f"Review: {review}")
                    print(f"Sentiment: {sentiment}\n")


batches = 10
file_path = '../data/IMDB Dataset.csv'
read_by_batches(file_path, batches, max_batches=2)  # Limit to print only 2 batches
