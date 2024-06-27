import pandas as pd
import csv 
import json

fields = ["ID","Translation\n"] 
language = ["Bengali","Gujarati","Hindi","Kannada","Malayalam","Tamil","Telgu"]
Index = [147532,298458,505511,663498,812760,976433,1114345]
for i in range(7):
    filename = "Shit_submission3/"+language[i]+".csv"
    with open(filename, 'w',newline='') as csvfile: 
        # creating a csv writer object 
        # csvwriter = csv.writer(csvfile,lineterminator='') 
            
        # writing the fields 
        # csvwriter.writerow(fields) 
        # csvfile.write('\"ID\"\t\"Translation\"\n')
            
        # writing the data rows 
        f = open("Shit_submission3/answer_"+language[i]+".txt", "r")
        j=0
        for x in f:
            
            ID = Index[i]+j
            x = x[:len(x)-1]
            x = str(ID)+"\t"+"\"" + x + "\"" + '\n'
            # csvwriter.writerows([[x]])  
            csvfile.write(x)
            j=j+1


# List of CSV file paths to concatenate
csv_files = ['Shit_submission3/Bengali.csv', 'Shit_submission3/Gujarati.csv','Shit_submission3/Hindi.csv','Shit_submission3/Kannada.csv','Shit_submission3/Malayalam.csv','Shit_submission3/Tamil.csv','Shit_submission3/Telgu.csv']

# Create an empty list to store the concatenated lines
concatenated_lines = []

# Loop through each CSV file
for file_path in csv_files:
    with open(file_path, 'r') as file:
        # Read all lines from the current file and append them to the concatenated_lines list
        concatenated_lines.extend(file.readlines())

# Create a new CSV file and write the concatenated lines to it
output_file = 'Shit_submission3/answer.csv'
with open(output_file, 'w') as outfile:
    outfile.writelines(concatenated_lines)

print(f"CSV files have been concatenated and saved as {output_file}.")