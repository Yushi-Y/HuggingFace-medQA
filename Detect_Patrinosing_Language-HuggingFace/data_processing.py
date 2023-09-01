from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib import request
from collections import Counter
import nltk
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import logging
import torch
import random


"""## Data Augmentation"""

# Download wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

# The stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

# Data Augmentation with synonyms from wordnet 
# Only for positive labeled samples


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def augment_data(train_df, factor):
    df_pos = train_df[train_df['label']==1]

    for index, row in df_pos.iterrows():
        new_text = apply_synonyms(row['text'], factor)
        new_row = row
        new_row.loc['text'] = new_text
        train_df = train_df.append(new_row)

    return train_df

def apply_synonyms(original_paragraph, number_of_replacements):
    list_of_sentences =  original_paragraph.split('.')
    new_list_of_sentences = []

    for idx_sentence, sentence in enumerate(list_of_sentences):
        words = sentence.lower().split()
        new_words = synonym_replacement(words, number_of_replacements)
        new_sentence = ' '.join(new_words)
        new_list_of_sentences += [new_sentence]
        
    return ". ".join(new_list_of_sentences)
	
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
		
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: # Only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


"""## Data Downsampling"""

# Randomly downsample negative instances to 2 times the number of positive instances
def downsample(train_df):
    train_neg_downsample = resample(train_df[train_df.label==0],
    replace=True,
    n_samples=2*len(train_df[train_df.label==1]),
    random_state=42)
 
    train_pos = train_df[train_df.label==1]
    train_downsample = pd.concat([train_neg_downsample, train_pos])

    return train_downsample



"""## Data Upsampling"""

# Randomly upsample positive instances to double its original number - risk of overfitting
def upsample(train_df):
    train_pos_upsample = resample(train_df[train_df.label==1],
              replace=True,
              n_samples=2*len(train_df[train_df.label==1]),
              random_state=42)

    train_neg = train_df[train_df.label==0]
    train_upsample = pd.concat([train_neg, train_pos_upsample, ])

    return train_upsample



"""## Change Class Weights"""

# Define the model with equal class weights 

# model_args = ClassificationArgs(num_train_epochs=1, 
#                                       no_save=True, 
#                                       no_cache=True, 
#                                       overwrite_output_dir=True)

# unweighted_model = ClassificationModel("roberta", 
#                                  'roberta-base', 
#                                   args = model_args, 
#                                   num_labels=2, 
#                                   use_cuda=cuda_available) 

def create_weighted_model(train_df):
    # Compute the class weights so that two classes are balanced
    labels = train_df['label']
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
    class_weights = class_weights.tolist()

    # Define the model with the balanced class Weights 
    weighted_model = ClassificationModel("roberta", 
                                     'roberta-base', 
                                      args = model_args, 
                                      num_labels=2, 
                                      use_cuda=cuda_available,
                                      weight = class_weights)
    
    return weighted_model



"""## Cross Validation"""

def k_fold_cv(k, data, augment=False, preprocess=True, preprocess_function=downsample, weighted_model=False):

    kf = KFold(n_splits=k, random_state=1000, shuffle=True)
    
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, val_index in kf.split(data[['text', 'label']]):
        # Split the dataframe 
        train_df = data.iloc[train_index]
        val_df = data.iloc[val_index]
        # Data processing - augmentation
        if augment==True:
            train_df = augment_data(train_df, 2) # Replace 2 words randomly from each sentence
        # Data processing - downsampling, upsampling
        if preprocess==True: 
            train_df = preprocess_function(train_df)
        # Train the model
        if weighted_model == True: 
            model = create_weighted_model(train_df)
            model.train_model(train_df)
        else:
            model = unweighted_model
            model.train_model(train_df)
        # Validate the model
        result, model_outputs, wrong_predictions = model.eval_model(val_df, acc = accuracy_score, 
                                                                    precision = precision_score, 
                                                                    recall = recall_score, 
                                                                    f1_score = f1_score)
        # result, model_outputs, wrong_predictions = model.eval_model(val_df, f1_score = f1_score)
        # Append model scores
        accuracy.append(result['acc'])
        precision.append(result['precision'])
        recall.append(result['recall'])
        f1.append(result['f1_score'])

    print("Average accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("Average precision for positive class: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("Average recall for positive class: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("Average f1 score for positive class: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))



"""# RoBERTa Baseline Results (Cross Validation)

Best processing methods: 

*   Augmentation + Downsampling (Average F1 score on CV: 0.68)
*   Augmentation + Upsampling (Average F1 score on CV: 0.67)
"""

if __name__ == "__main__":
    # Try downsample only
    k_fold_cv(5, training_set, augment=False, preprocess=True, preprocess_function=downsample, weighted_model=False)

    # Try upsampling only
    k_fold_cv(5, training_set, augment=False, preprocess=True, preprocess_function=upsample, weighted_model=False)

    # Try downsampling + balanced class weights
    k_fold_cv(5, training_set, augment=False, preprocess=True, preprocess_function=downsample, weighted_model=True)

    # Try augmenting + downsampling
    k_fold_cv(5, training_set, augment=True, preprocess=True, preprocess_function=downsample, weighted_model=False)

    # Try augmenting + upsampling
    k_fold_cv(5, training_set, augment=True, preprocess=True, preprocess_function=upsample, weighted_model=False)

    # Try augmenting + balance class weights
    k_fold_cv(5, training_set, augment=True, preprocess=False, preprocess_function=None, weighted_model=True)

    # Try augmenting + downsampling + balanced class weights
    k_fold_cv(5, training_set, augment=True, preprocess=True, preprocess_function=downsample, weighted_model=True)

    # Try augmenting + upsampling + balanced class weights
    k_fold_cv(5, training_set, augment=True, preprocess=True, preprocess_function=upsample, weighted_model=True)
