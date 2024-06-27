import pandas as pd
import csv 
import json

def load_data(lang):
    #loading data from the desired directory
    DATA_PATH = 'data/English_'+lang+'.csv'
    # TEST_PATH = '/kaggle/input/cs779-mt/eng_Hindi_data_dev_X.csv'
    FINAL_TEST_DATA = 'data/English_'+lang+'_Test.csv'
    data = pd.read_csv(DATA_PATH, header = None)
    data.columns = ['hindi', 'english']

    # test = pd.read_csv(TEST_PATH, header = None)
    # test.columns = ['sentence']
    final_test_data = pd.read_csv(FINAL_TEST_DATA, header = None)
    final_test_data.columns = ['sentence']

    def swap_columns(df, col1, col2):
        col_list = list(df.columns)
        x, y = col_list.index(col1), col_list.index(col2)
        col_list[y], col_list[x] = col_list[x], col_list[y]
        df = df[col_list]
        return df

    data = swap_columns(data, 'hindi', 'english')


    data['english'] = data['english'].apply(str)
    data['hindi'] = data['hindi'].apply(str)
    final_test_data['sentence'] = final_test_data['sentence'].apply(str)

    return data, final_test_data


def json2csv_train_data():

    with open('train_data1.json') as user_file:
        parsed_json = json.load(user_file)

    filename = "English_Malayalam.csv"
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            

        for i in range(812759+1-758703):
            x = 758703+i
            csvwriter.writerows([[parsed_json["English-Malayalam"]["Train"][str(x)]["source"],
                                  parsed_json["English-Malayalam"]["Train"][str(x)]["target"]]])

    return None

def json2csv_test_data():

    with open('test_data1.json') as user_file:
        parsed_json = json.load(user_file)        
    # field names 
    
    filename = "English_Bengali_test.csv"
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
            
        # writing the data rows 
        for i in range(196710-177039+1):
            x = 177039+i
            csvwriter.writerows([[parsed_json["English-Bengali"]["Test"][str(x)]["source"]]])

    return None








# Function to check if a text contains Malayalam script characters
def has_malayalam_script(text):
    malayalam_script_range = range(0x0D00, 0x0D7F)  # Unicode range for Malayalam script
    
    for char in text:
        if ord(char) in malayalam_script_range:
            return True
    return False

# Function to check if a text contains Kannada script characters
def has_kannada_script(text):
    kannada_script_range = range(0x0C80, 0x0CFF)  # Unicode range for Kannada script
    
    for char in text:
        if ord(char) in kannada_script_range:
            return True
    return False

# Function to check if a text contains Telugu script characters
def has_telugu_script(text):
    telugu_script_range = range(0x0C00, 0x0C7F)  # Unicode range for Telugu script
    
    for char in text:
        if ord(char) in telugu_script_range:
            return True
    return False

# Function to check if a text contains Tamil script characters
def has_tamil_script(text):
    tamil_script_range = range(0x0B80, 0x0BFF)  # Unicode range for Tamil script
    
    for char in text:
        if ord(char) in tamil_script_range:
            return True
    return False

# Function to check if a text contains Gujarati script characters
def has_gujarati_script(text):
    gujarati_script_range = range(0x0A80, 0x0AFF)  # Unicode range for Gujarati script
    
    for char in text:
        if ord(char) in gujarati_script_range:
            return True
    return False

def has_hindi_script(text):
    hindi_script_range = range(0x0900, 0x097F)  # Unicode range for Hindi script
    
    for char in text:
        if ord(char) in hindi_script_range:
            return True
    return False

def has_bengali_script(text):
    bengali_script_range = range(0x0980, 0x09FF)  # Unicode range for Bengali script
    
    for char in text:
        if ord(char) in bengali_script_range:
            return True
    return False

def is_english_text(sentence):
    # Check if all characters in the sentence are within the ASCII range (English characters)
    return all(ord(char) < 128 for char in sentence)



def detect_script(filename):
    if "Bengali" in filename:
        return has_bengali_script
    elif "Hindi" in filename:
        return has_hindi_script
    elif "Gujarati" in filename:
        return has_gujarati_script
    elif "Kannada" in filename:
        return has_kannada_script
    elif "Telugu" in filename:
        return has_telugu_script
    elif "Tamil" in filename:
        return has_tamil_script
    elif "Malayalam" in filename:
        return has_malayalam_script
    else:
        return None
    

def clean_7_data():
    # Initialize a dictionary to store counts for each script
    script_counts = {
        'Bengali': 0,
        'Hindi': 0,
        'Gujarati': 0,
        'Kannada': 0,
        'Telgu': 0,
        'Tamil': 0,
        'Malayalam': 0
    }

    # Process each CSV file
    for language in script_counts.keys():
        filename = f'data/English_{language}_test.csv'
        script_detection_function = detect_script(filename)
        
        if script_detection_function:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                
                # Assuming the sentences are in the first column (index 0)
                for row in csvreader:
                    sentence = row[0]
                    
                    if script_detection_function(sentence):
                        print(sentence)
                        print("-----------------------------------")
                        script_counts[language] += 1

    # Print the script counts
    for language, count in script_counts.items():
        print(f"Number of {language} script sentences: {count}")

    return None    

def detect_script_eng(filename):
    if "Bengali" in filename:
        return is_english_text
    elif "Hindi" in filename:
        return is_english_text
    elif "Gujarati" in filename:
        return is_english_text
    elif "Kannada" in filename:
        return is_english_text
    elif "Telugu" in filename:
        return is_english_text
    elif "Tamil" in filename:
        return is_english_text
    elif "Malayalam" in filename:
        return is_english_text
    else:
        return None
    


def clean_test_data_eng():
    # Initialize a dictionary to store counts for each script
    script_counts = {
        'Bengali': 0,
        'Hindi': 0,
        'Gujarati': 0,
        'Kannada': 0,
        'Telgu': 0,
        'Tamil': 0,
        'Malayalam': 0
    }

    # Process each CSV file
    for language in script_counts.keys():
        filename = f'data/English_{language}.csv'
        script_detection_function = detect_script(filename)
        
        if script_detection_function:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                row_number = 0
                # Assuming the sentences are in the first column (index 0)
                for row in csvreader:
                    sentence = row[1]
                    row_number += 1
                    if script_detection_function(sentence):
                        
                        # print(sentence,row_number,csvfile)
                        # print("-----------------------------------")
                        script_counts[language] += 1

    # Print the script counts
    for language, count in script_counts.items():
        print(f"Number of {language} script sentences: {count}")

    return None

