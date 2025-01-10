import os
import csv
import json


# change this path to the directoy containing the csv files with the annotations
CSV_DATA_PATH = r"C:\Users\anasn\Downloads\OPP-115_\OPP-115\consolidation\threshold-0.75-overlap-similarity"

# change to path of where you want text files to be outputted. 
OUTPUT_DIR= r"C:\Users\anasn\OneDrive\Desktop\files_categories"

counter = 0

for filename in os.listdir(CSV_DATA_PATH):

    # defined new dictionary for every file
    categories = {
    'Other' : False,
    'Policy Change': False,
    'First Party Collection/Use': False,
    'Data Retention': False,
    'International and Specific Audiences': False,
    'Third Party Sharing/Collection': False,
    'User Choice/Control': False,
    'User Access, Edit and Deletion': False,
    'Data Security': False,
    'Do Not Track': False}


    print(f"processing {filename}")
    new_filename = filename.replace(".csv", ".json")
    with open(os.path.join(CSV_DATA_PATH, filename), mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[5] in categories:
                categories[row[5]] = True
    with open(os.path.join(OUTPUT_DIR, new_filename), "w", encoding="utf-8") as json_file:
        json.dump(categories, json_file, indent=4, ensure_ascii=False)
    
    counter += 1


print(f"Processed {counter} files")
