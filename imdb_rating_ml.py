import comet_ml                                        #initialize the coment_ml library
from  transformers import AutoTokenizer, Trainer, TrainingArguments # import all the required libraries
from datasets import load_dataset #importing the dataset library
import torch           
import numpy as np
import random
from transformers import DataCollatorWithPadding # Data collator is used to collate the data in the form of batches(splitting the data into batches)
from transformers import AutoModelForSequenceClassification #importing the transformer model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support #importing the required metrics, accuracy, precision, recall, f1 score, support, etc., 
import os

os.environ["COMET_MODE"] = "ONLINE" #setting the comet mode to online , so that the results can be viewed online
os.environ["COMET_LOG_ASSETS"] = "True" #setting the comet log assets to true, so that the assets can be logged

experiment = comet_ml.Experiment(                   #initialize the experiment     #code from chatgpt                                     
    project_name="IMBD_distilbart",  # Replace with your desired project name
    api_key = "rimsLS7ePUrxbDV1pY10TsSyR" #got the free api key from comet website in account and apikeys 
)

api_key = os.getenv("COMET_API_KEY") #getting the comet api key from the environment variable
experiment = comet_ml.Experiment(project_name="IMBD_distilbart", api_key=api_key)


pre_trained_model = "distilbert-base-uncased-finetuned-sst-2-english" #Initializing pretrained model

tokenizer = AutoTokenizer.from_pretrained(pre_trained_model) # initailizing the tokenizer, it helps input data to be converted into tokens for better understanding
#making it easier for machine learning models to analyze and learn from the data.

raw_datasets = load_dataset("imdb")  #loading the dataset

SEED = 20      #setting the seed value

def tokenize_function(examples): #function to tokenize the data #Passig the raw data to the tokenizer in order to get tokenizedd daata
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt") #padding is used to make all the input data of same length, 
#truncation is used to truncate the data if it is too long, so that it can be processed easily by the model

def get_example(index):
    return eval_dataset[index]["text"] #getting the example from the evaluation dataset at the given index

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) #mapping the tokenized data to the dataset

Data_collator = DataCollatorWithPadding(tokenizer = tokenizer) #initializing the data collator # creatiing a new sample of the dataset by sing tokenized dataset
# then we split the data into batches of training data and evaluation data 

train_dataset = tokenized_datasets["train"].shuffle(SEED).select(range(200))#selecting the training data from the tokenized dataset, shuffling the data and selecting the first 200 samples
eval_dataset = tokenized_datasets["test"].shuffle(SEED).select(range(200)) #selecting the evaluation data from the tokenized dataset, shuffling the data and selecting the first 200 samples

model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=2) #initializing the model #loading the pretrained model and setting the number of labels to 2
# The num_labels=2 means we are setting up a binary classification task (e.g., positive/negative) This means your task involves categorizing input text into predefined labels.

# ---------------------------------------------------------------------------------------
# get_example(index):
#   1) Takes an integer index.
#   2) Returns the text (review) from `eval_dataset` at that index.
#   3) Used by Comet ML to display which exact piece of text was incorrectly/correctly predicted.
# ---------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------
# compute_metrics(pred):
#   1) Retrieves the Comet ML experiment (if running) to log various details.
#   2) Extracts true labels (label) from pred.label_ids.
#   3) Converts raw model outputs (pred.predictions) into predicted labels (preds) 
#      by taking argmax across the last dimension.
#   4) Calculates classification metrics: precision, recall, f1 score, and accuracy.
#   5) If a Comet experiment is active:
#      - Grabs the current epoch (or defaults to 0).
#      - Logs a confusion matrix (actual vs. predicted) to Comet:
#         * Uses 'index_to_example_function' = get_example so we can see which exact text
#           corresponds to each entry in the confusion matrix.
#      - Logs the first 20 text samples (and their true labels) to Comet 
#        for easy inspection and debugging.
#   6) Returns a dictionary of metrics (accuracy, f1, precision, recall) 
#      for the Trainer or any other part of the code that needs them.
# ---------------------------------------------------------------------------------------
def compute_metrics(pred):
    experiment = comet_ml.get_global_experiment()

    # True labels for each example
    label = pred.label_ids
    
    # Predicted labels (0 or 1) after taking argmax of model logits
    preds = pred.predictions.argmax(-1)
    
    # Calculate precision, recall, f1 score (macro-averaged), and ignore 'support' (the fourth return)
    precision, recall, f1, _ = precision_recall_fscore_support(label, preds, average="macro")
    
    # Calculate overall accuracy
    acc = accuracy_score(label, preds)

    # If Comet experiment is available, log additional info
    if experiment:
        # Get current epoch from Comet if defined, else default to 0
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
        
        # Inform Comet which epoch we are logging
        experiment.set_epoch(epoch)

        # Log a confusion matrix to show how often we got negative vs positive correct/incorrect
        experiment.log_confusion_matrix(
            y_true = label,
            y_predicted = preds,
            file_name = f"confusion-matrix-epoch-{epoch}.json",
            labels = ["negative", "positive"],
            index_to_example_function = get_example  # This lets Comet show the original text
        )

        # Log the first 20 examples (reviews) to Comet with their true labels
        for i in range(20):
            experiment.log_text(get_example(i), metadata={"label": label[i].item()})
    
    # Finally, return the calculated metrics as a dictionary
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}





training_args = TrainingArguments( #initializing the training arguments
    seed  = SEED, #setting the seed value
    output_dir="results", #setting the output directory
    overwrite_output_dir = True, #setting the overwrite output directory to true
    num_train_epochs=3 , #setting the number of training epochs to 3
    do_train = True,
    do_eval = True,
    evaluation_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_total_limit=10,
    save_steps = 25,
    per_device_train_batch_size=8, #setting the batch size  
)

trainer = Trainer(     #initializing the trainer
    model = model, #setting the model
    args = training_args, #setting the training arguments
    train_dataset = train_dataset, #setting the training dataset
    eval_dataset = eval_dataset, #setting the evaluation dataset
    compute_metrics = compute_metrics, #setting the compute metrics
    data_collator= Data_collator, #setting the data collator
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #setting the device to cuda if available else to cpu 
model.to(device) #moving the model to the device

trainer.train() #training the model






















