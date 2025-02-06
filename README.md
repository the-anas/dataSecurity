# dataSecurity

This repository contains the code related to automatically annotating privacy policy datasets through different text classifiers based on the OPP-115 dataset.

# Replicating project
follow below instructions to replicate project

### Set up
1. Install requirements find in `Requirements.txt`. 
2. Download OPP-115 dataset found [here](https://usableprivacy.org/data). 

### Performing data transformations from OPP-115
Note that data for training the first stage classifier is provided here already in the file `seperated_categories/agg_data.pkl´
Data for training the second stage classifiers is also provided here under `seperated_categories/´ but, in case you want to replicate it follow these steps:
1. Link your OPP-115 dataset path in to the complete the path  in `data_transformations/seperate_categories.py´.
2. run `data_transformations/seperate_categories.py´, you will get seperate dataframes for all text segments in all csv files in OPP-115 seperated per category. 
3. Use the `data_transformations/Transform_to_multilabel.ipynb´ notebook to transform every dataframe seperately into one adapted for multi-label classification. 

### how to rerun experiments and retrain models
1. Make sure you have all the necessary data (present in repo all ready). This the `seperated_categories/agg_data.pkl´ file for the first classifier and files under `seperated_categories/´ 
### information about notebooks

### information about what is in each directory