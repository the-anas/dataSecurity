# Useful functions for data preprocessing

import os
import re
import pandas as pd
import json

# function to get the identifier of a file
def get_identifier(filename):
    # Regular expression to match a number followed by an underscore at the start of the filename
    match = re.match(r"\d+_", filename)
    if match:
        return match.group()  # Extract the matched number
    return None 


# function that returns defined a list of defined segments from html file, used to match segments to a csv file
def get_segments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        segments = data.split("|||")
        return segments
    

# function that gets a file given an identifier and a directory
def get_file(identifier, directory):
    for filename in os.listdir(directory):
        if filename.startswith(identifier):
            return os.path.join(directory, filename)
    return None

# function that processes the attributes of an annotation, these attributes follow a json format
def preprocess_json(data):
    """
    Preprocess the JSON data to extract the element names and their values.

    Args:
    data (dict): JSON data to preprocess.

    Returns:
    dict: A dictionary with element names as keys and their values.
    """
    data = json.loads(data)
    result = {}
    for key, details in data.items():
        if isinstance(details, dict) and 'value' in details:
            result[key] = details['value']
    return result


# Function to keep only specified keys in a dictionary
def keep_keys(d, keys_to_keep):
    #d = eval(d) # convert column from str to dict
    return {k: v for k, v in d.items() if k in keys_to_keep}


# function that takes in dataframe and turns json column into dataframe columns
def json_to_columns(df):

    json_df = pd.json_normalize(df['attributes'])
    df = df.drop(columns=['attributes'])
    json_df = json_df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    # Merge the new columns back into the original DataFrame
    df = pd.concat([df, json_df], axis=1)

    return df
