import nltk

def calculate_bleu_score(file1_path, file2_path):
    nltk.download('punkt')

    truth = []
    with open(file1_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = nltk.word_tokenize(line)
            truth.append([line])

    submission_answer = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = nltk.word_tokenize(line)
            #line = [token for token in line if token != '"']
            submission_answer.append(line)
            #print(newline)

    score = nltk.translate.bleu_score.corpus_bleu(truth, submission_answer)
    return score

if __name__ == "__main__":
    file1_path = '../Hindi_true.txt'  # Replace with the actual path to file1.txt
    file2_path = '../Hindi_pred.txt'  # Replace with the actual path to file2.txt

    score = calculate_bleu_score(file1_path, file2_path)
    print(score)
    
    
    

