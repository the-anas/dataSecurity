# Data Security Project

## IMPORTANT NOTE
This repository uses git submodules. When cloning make sure you take that into account. 
You can clone via `git clone --recurse-submodules repo-web-url`

## About the Project
This repository contains our Code (Nicola and Anas) for the Data Security Project of the ESG Data Science Master. Here we attempt to test and compare two different approaches to the automatic annotation of privacy policies: Text classification through fine-tuning small transformer models and n-shot prompting with LLMs. 
Since this project is divided into two parts, you can see two different folders here. 
Under `Text Classification/`, you will find files related to using transformers for text classification. You will find scripts we used for data transformation along with training scripts, training logs, and notebooks we used for testing various things.
Under `LLMs/`, you will find repositories for testing In-Context Learning, Chain-Of-Thought, and for computing the metrics with different LLMs. This folder contains also copy of the OPP-115 policies used for the classification, for the complete dataset you may have a look at https://www.usableprivacy.org/data (credits to the authors).

You will find a separate README.MD file in each folder with information about the code and instructions on how to run it.  
