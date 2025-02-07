import comet_ml                                        #initialize the coment_ml library
from  transformers import AutoTokenizer, Trainer, TrainingArguments # import all the required libraries
from datasets import load_dataset #importing the dataset library
import torch           
import numpy as np
import random
from transformers import DataCollatorWithPadding # Data collator is used to collate the data in the form of batches(splitting the data into batches)
from transformers import AutoModelForSequenceClassification #importing the transformer model


experiment = comet_ml.Experiment(                   #initialize the experiment     #code from chatgpt                                     
    project_name="IMBD_distilbart",  # Replace with your desired project name
    api_key = "rimsLS7ePUrxbDV1pY10TsSyR" #got the free api key from comet website in account and apikeys 
)

pre_trained_model = "sshleifer/distilbart-cnn-12-6"  #Initializing pretrained model

tokenizer = AutoTokenizer.from_pretrained(pre_trained_model) # initailizing the tokenizer, it helps input data to be converted into tokens for better understanding
#making it easier for machine learning models to analyze and learn from the data.

raw_datasets = load_dataset("imdb")  #loading the dataset

SEED = 20      #setting the seed value

def tokenize_function(examples): #function to tokenize the data #Passig the raw data to the tokenizer in order to get tokenizedd daata
    return tokenizer(examples["text"], padding="max_length", truncation=True) #padding is used to make all the input data of same length, 
#truncation is used to truncate the data if it is too long, so that it can be processed easily by the model

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) #mapping the tokenized data to the dataset

Data_collator = DataCollatorWithPadding(tokenizer = tokenizer) #initializing the data collator # creatiing a new sample of the dataset by sing tokenized dataset
# then we split the data into batches of training data and evaluation data 

train_dataset = tokenized_datasets["train"].shuffle(SEED).select(range(200))#selecting the training data from the tokenized dataset, shuffling the data and selecting the first 200 samples
eval_dataset = tokenized_datasets["test"].shuffle(SEED).select(range(200)) #selecting the evaluation data from the tokenized dataset, shuffling the data and selecting the first 200 samples

model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=2) #initializing the model #loading the pretrained model and setting the number of labels to 2
# The num_labels=2 means we are setting up a binary classification task (e.g., positive/negative) This means your task involves categorizing input text into predefined labels.











































