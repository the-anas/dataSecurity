import os
import pandas as pd
from utils import get_identifier, get_segments, get_file, preprocess_json, keep_keys, json_to_columns

# THIS FILE PROCESSES DATA TO TRAIN THE second set of classifiers ONLY

try:
    CSV_DATA_PATH = r"...\OPP-115\annotations" # include your path to OPP-115\annotations of OPP-115 dataset
    TEXT_DATA_PATH = r"...\OPP-115\sanitized_policies" # include your path to OPP-115\sanitized_policies of OPP-115 dataset
except:
    print("Update the paths to the OPP-115 dataset to your own path.")

COL_NAMES = [ "annotationID",  "batchID", "annotatorID", "policyID", "segmentID", "category", "attributes",
             "URL", "date"]
COL_TO_KEEP = ["annotationID", "segmentID", "category", "attributes"]
PATH_TO_SAVE = r".\seperated_categories"

# Keys to keep for each category
Data_Retention_keys = ["Retention Period", "Retention Purpose", "Personal Information Type"]
First_Party_Collection_or_Use_keys = ["Collection Mode", "Personal Information Type", "Purpose"]
Third_Party_Sharing_or_Collection_keys = ["Action Third Party", "Personal Information Type", "Purpose"]
Do_Not_Track_keys =["Do Not Track policy"]
Policy_Change_keys = ["Change Type", "User Choice", "Notification Type"]
User_Access_Edit_and_Deletion_keys = [ "Access Type", "Access Scope"]
User_Choice_or_Control_keys = ["Choice Type", "Choice Scope"]
Data_Security_keys = ["Security Measure"]	
Other_keys = ["Other Type"]
International_and_Specific_Audiences_keys = ["Audience Type"]

# counter for logs
counter = 0

# Empty dataframe
combined_df = pd.DataFrame(columns=COL_TO_KEEP + ["segment"])	

for filename in os.listdir(CSV_DATA_PATH):
    identifier = get_identifier(filename) # needded to match the csv file to the text file, since the number in the id alone fails
    text_file = get_file(identifier, TEXT_DATA_PATH)
    segments = get_segments(text_file) # returns list of segements 
    file_path = os.path.join(CSV_DATA_PATH, filename)
    print(f"Processing {file_path}")

    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, header=None, names=COL_NAMES)
        # take columns you need only
        df = df[COL_TO_KEEP]

        # Add the text to the dataframe
        df['segment'] = df['segmentID'].map(lambda x: segments[x])
    

        # reformat the json column (attributes column)
        df['attributes'] = df['attributes'].apply(preprocess_json)
        combined_df = pd.concat([combined_df, df], ignore_index=True)


        counter += 1 

print(f"Processed {counter} files.")

# Split files per there categories

dataframes_names = list()
# Get unique categories
unique_categories = combined_df['category'].unique()

# Loop through each unique category
for category in unique_categories:
    # Filter DataFrame for the current category
    category_df = combined_df[combined_df['category'] == category]
    sanitized_category = category.replace(' ', '_').replace('/', '_or_').replace(",","")
    filename = f"{sanitized_category}.csv"

    globals()[sanitized_category] = category_df
    dataframes_names.append(sanitized_category)

for dataframe in dataframes_names:
    print(f"Processing {dataframe} DataFrame")
    # Apply the function to the datafames
    globals()[dataframe]['attributes'] = globals()[dataframe]['attributes'].apply(lambda x: keep_keys(x, globals()[dataframe + "_keys"]))
    new_dataframe = json_to_columns(globals()[dataframe])
    # Save the dataframes
    new_dataframe.to_csv(os.path.join(PATH_TO_SAVE, dataframe + ".csv"), index=False)

print("Done.")