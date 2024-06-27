import nltk
import io
import csv
import subprocess
subprocess.run(["pip", "install", "rouge"])
import pandas as pd

from rouge import Rouge

nltk.download('punkt')

def calculate_rouge_scores(file1_path, file2_path):
    file1_data = []
    with io.open(file1_path, 'r', encoding='utf-8') as f:
        
        for input_line in f:
            row = [column.strip('"') for column in input_line.split(maxsplit=1)]
            if len(row) >= 2:
                file1_data = pd.concat(file1_data,(row[0], nltk.word_tokenize(row[1])))
                #print(nltk.word_tokenize(row[1]))

    file2_data = []
    with io.open(file2_path, 'r', encoding='utf-8') as f:
        
        for input_line in f:
            row = [column.strip('"') for column in input_line.split(maxsplit=1)]
            if len(row) >= 2:
                file2_data = pd.concat(file2_data,(row[0], nltk.word_tokenize(row[1])))

    # if len(file1_data) != len(file2_data):
    #     raise ValueError(f"Number of data points in file1({len(file1_data)}) and file2 must be the same ({len(file1_data)})")

    rouge_scorer = Rouge()
    rouge_l_scores = []

    totalscore = 0.0
    total_itrs =0
    for (hyp), (ref) in zip(file1_data, file2_data):
        if total_itrs==1:
            continue

        total_itrs = total_itrs+1
        # if answer_id != ref_id:
        #     print(f"Warning: Answer ID ({answer_id}) and Ref ID ({ref_id}) do not match. Skipping this iteration.")
        #     continue        
        hyp_text = ' '.join(hyp)
        ref_text = ' '.join(ref)
        
        #print("hyptext")
        #print(hyp_text)
        scores = rouge_scorer.get_scores(hyp_text, ref_text)
        rouge_l_scores.append(scores[0]['rouge-l']['f'])
        totalscore = totalscore + scores[0]['rouge-l']['f']

    return (totalscore/total_itrs)

if __name__ == "__main__":
    file1_path = '../Hindi_pred.txt'  # Replace with the actual path to file1.txt
    file2_path = '../Hindi_true.txt' # Replace with the actual path to file2

    score = calculate_rouge_scores(file1_path, file2_path)
    print(score)
