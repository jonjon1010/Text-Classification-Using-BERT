# CompSci 723 - Intro to NLP 
# Assignment 3 - Text Classifcaiton Using BERT
# Name: Jonathan Nguyen 
# Date: 12/02/2021
# Version: 1.0 

# Setup: 
# 1. Install transformers and tensorflow 
# 2. Try things from the slides. Make sure you can run code in A3.py 

# Task 1: 
# Create your own small test dataset with at least 10 positive and at 10 negative examples of movie reivews. You could create your own examples 
# (including a few tricky ones if you want) and/or take some from popular movie reivew websites. Put them in a folder just like the acllmdb/test folder. 

# Task 2: 
# Train at least two differnt BERT fine-tuning models. They could differ in the classification architecture (e.g. the alternate model from the slides),
# number of traning examples, number of epochs, BERT's cased vs. uncased models, two differnt BERT models, etc. After training, evaluate each model on 
# test data from aclImdb (use same number of test examples for evaluting each model). Next, evaluate each model on your samll test dataset. Also look at
# the models' predictions on your small dataset. 

# Submit: 
# 1. Report: Write a report in which you include: 
# (4 points) Describe the configurations of the models and how they differ.Mention whatwas your expectation with their performances. 
# (3 points) Show evalution results on the test examples of acllmdb.Give comments on your results. 
# (3 points) Show evaluation result on your small test dataset.Give comments on your results. Also give a few comments of the predictions. 
# 2. Code (3 points): Submit the code your wrote for doing this assignment as an executable .py file . The grader should be able to replicate your results.
# 3. Dataset (2 points): Submit the small test dataset you created as a compressed folder file. 

from transformers import AutoTokenizer, TFAutoModel 
model = TFAutoModel. from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                                          
maxTrEg=1000 # maximum number of pos & neg training examples
maxTeEg=1000# maximum number of pos & neg test examples                    
