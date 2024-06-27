import nltk
from sacrebleu import sentence_chrf
import csv
import pandas as pd

def calculate_chrf_score(file1_path, file2_path):
    file1_data = []
    with open(file1_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                file1_data.append((row[0], nltk.word_tokenize(row[1])))

    file2_data = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                file2_data.append((row[0], nltk.word_tokenize(row[1])))

    chrF_score = []
    totalscore = 0.0
    print(file1_data)
    for hyp, ref in zip(file1_data, file2_data):
        print("ggggggggg")
        hyp_text = ' '.join(hyp)
        ref_text = ' '.join(ref)
        scores = sentence_chrf(hyp_text, [ref_text])
        chrF_score.append(scores.score)
        totalscore = totalscore + scores.score

    return (totalscore/len(chrF_score))

if __name__ == "__main__":
#     file1_path = '../Hindi_pred.csv'  # Replace with the actual path to file1.txt
#     file2_path = '../Hindi_true.csv'  # Replace with the actual path to file2
    
#     dataframe1 = pd.read_csv("../Hindi_pred.txt",header = None)
#     dataframe2 = pd.read_csv("../Hindi_true.txt",header = None)
# # storing this dataframe in a csv file
#     dataframe1.to_csv(file1_path, index = None)
#     dataframe2.to_csv(file2_path, index = None)



    input_file1 = '../Hindi_pred.txt'
    file1_path = '../Hindi_pred.csv'

    # Open the input text file in read mode and the output CSV file in write mode
    with open(input_file1, 'r') as text_file, open(file1_path, 'w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Iterate through the lines in the text file
        for line in text_file:
            # Remove leading and trailing whitespace from each line and split it by newline
            rows = line.strip().split('\n')
            
            # Write each row to the CSV file
            for row in rows:
                csv_writer.writerow([row])

    input_file2 = '../Hindi_true.txt'
    file2_path = '../Hindi_true.csv'

    # Open the input text file in read mode and the output CSV file in write mode
    with open(input_file2, 'r') as text_file, open(file2_path, 'w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Iterate through the lines in the text file
        for line in text_file:
            # Remove leading and trailing whitespace from each line and split it by newline
            rows = line.strip().split('\n')
            
            # Write each row to the CSV file
            for row in rows:
                csv_writer.writerow([row])


    score = calculate_chrf_score(file1_path, file2_path)
    print(score)
