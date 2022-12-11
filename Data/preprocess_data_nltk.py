##################################################
# Reads data from csv, tokenizes it using NLTK and
# aligns labels with each token
##################################################

import pandas as pd
from nltk.tokenize import wordpunct_tokenize
import ast
import numpy as np
import re
import tensorflow as tf

# list of unique labels with ids
unique_labels = {"O": 0, "B-COURT": 1, "B-PETITIONER": 2, "B-RESPONDENT": 3, "B-JUDGE": 4, "B-LAWYER": 5, 
                "B-DATE": 6, "B-ORG": 7, "B-GPE": 8, "B-STATUTE": 9, "B-PROVISION": 10, "B-PRECEDENT": 11, 
                "B-CASE_NUMBER": 12, "B-WITNESS": 13, "B-OTHER_PERSON": 14, "I-COURT": 15, "I-PETITIONER": 16, 
                "I-RESPONDENT": 17, "I-JUDGE": 18, "I-LAWYER": 19, "I-DATE": 20, "I-ORG": 21, "I-GPE": 22, 
                "I-STATUTE": 23, "I-PROVISION": 24, "I-PRECEDENT": 25, "I-CASE_NUMBER": 26, "I-WITNESS": 27, 
                "I-OTHER_PERSON": 28, "<PAD>":29}

# read the data in
train_judgement_df = pd.read_csv('./cleandata/NER_TRAIN_JUDGEMENT.csv')
train_preamble_df = pd.read_csv('./cleandata/NER_TRAIN_PREAMBLE.csv')
dev_judgement_df = pd.read_csv('./cleandata/NER_DEV_JUDGEMENT.csv')
dev_preamble_df = pd.read_csv('./cleandata/NER_DEV_PREAMBLE.csv')

train_judgement_df = train_judgement_df.drop(index=514)# drop row 514 because it contains an invalid label

# combine judgement and preamble into a single dataframe
train_df = pd.concat([train_judgement_df, train_preamble_df])
dev_df = pd.concat([dev_judgement_df, dev_preamble_df])

# convert string lists to actual lists
train_entity_list = train_df['text'].values.tolist()
train_entity_list_clean = []
for entity in train_entity_list:
    entity_clean = entity.replace(".", ". ")
    entity_clean = entity_clean.replace(",", ", ")
    entity_clean = entity_clean.replace(")", ") ")
    entity_clean = entity_clean.replace("(", "( ")
    entity_clean = entity_clean.replace("[", "[ ")
    entity_clean = entity_clean.replace("]", "] ")
    entity_clean = entity_clean.replace("{", "{ ")
    entity_clean = entity_clean.replace("}", "} ")
    entity_clean = entity_clean.replace("'", "' ")
    entity_clean = entity_clean.replace("‟", "‟ ")
    entity_clean = entity_clean.replace("-", "- ")
    entity_clean = re.sub(' +', ' ', entity_clean)
    train_entity_list_clean.append(ast.literal_eval(entity_clean))

dev_entity_list = dev_df['text'].values.tolist()
dev_entity_list_clean = []
for entity in dev_entity_list:
    entity_clean = entity.replace(".", ". ")
    entity_clean = entity_clean.replace(",", ", ")
    entity_clean = entity_clean.replace(")", ") ")
    entity_clean = entity_clean.replace("(", "( ")
    entity_clean = entity_clean.replace("[", "[ ")
    entity_clean = entity_clean.replace("]", "] ")
    entity_clean = entity_clean.replace("{", "{ ")
    entity_clean = entity_clean.replace("}", "} ")
    entity_clean = entity_clean.replace("'", "' ")
    entity_clean = entity_clean.replace("‟", "‟ ")
    entity_clean = entity_clean.replace("-", "- ")
    entity_clean = re.sub(' +', ' ', entity_clean)
    dev_entity_list_clean.append(ast.literal_eval(entity_clean))

train_label_list = train_df['label'].values.tolist()
train_label_list_clean = []
for label in train_label_list:
    train_label_list_clean.append(ast.literal_eval(label))

dev_label_list = dev_df['label'].values.tolist()
dev_label_list_clean = []
for label in dev_label_list:
    dev_label_list_clean.append(ast.literal_eval(label))

# ---tokenize all the data using BertTokenizer--- #

# fix bracket punctuation
train_data_clean = []
for row in train_df['overall_text'].values.tolist():
    row_clean = row.replace(".", ". ")
    row_clean = row_clean.replace(",", ", ")
    row_clean = row_clean.replace(")", ") ")
    row_clean = row_clean.replace("(", "( ")
    row_clean = row_clean.replace("[", "[ ")
    row_clean = row_clean.replace("]", "] ")
    row_clean = row_clean.replace("{", "{ ")
    row_clean = row_clean.replace("}", "} ")
    row_clean = row_clean.replace("'", "' ")
    row_clean = row_clean.replace("‟", "‟ ")
    row_clean = row_clean.replace("-", "- ")
    row_clean = re.sub(' +', ' ', row_clean)
    train_data_clean.append(row_clean)

dev_data_clean = []
for row in dev_df['overall_text'].values.tolist():
    row_clean = row.replace(".", ". ")
    row_clean = row_clean.replace(",", ", ")
    row_clean = row_clean.replace(")", ") ")
    row_clean = row_clean.replace("(", "( ")
    row_clean = row_clean.replace("[", "[ ")
    row_clean = row_clean.replace("]", "] ")
    row_clean = row_clean.replace("{", "{ ")
    row_clean = row_clean.replace("}", "} ")
    row_clean = row_clean.replace("'", "' ")
    row_clean = row_clean.replace("‟", "‟ ")
    row_clean = row_clean.replace("-", "- ")
    row_clean = re.sub(' +', ' ', row_clean)
    dev_data_clean.append(row_clean)
    
# tokenize dataframes 
train_tokenized = []
for row in train_data_clean:
    train_tokenized.append(wordpunct_tokenize(row))

dev_tokenized = []
for row in dev_data_clean:
    dev_tokenized.append(wordpunct_tokenize(row))

# pad dataframes (we set max lenth to 5379, the length of the longest text)
train_tokenized = tf.keras.preprocessing.sequence.pad_sequences(train_tokenized, padding='post', truncating='post', value='<PAD>', maxlen=5379, dtype=object)
dev_tokenized = tf.keras.preprocessing.sequence.pad_sequences(dev_tokenized, padding='post', truncating='post', value='<PAD>', maxlen=5379, dtype=object)

df_train_tokenized = pd.DataFrame(train_tokenized.tolist())
df_dev_tokenized = pd.DataFrame(dev_tokenized.tolist())

# tokenize training entities
train_entities_tokenized = []
for row_entities in train_entity_list_clean:# iterate through row
    if not row_entities:# if row is empty then do not tokenize and just add an empty list
        train_entities_tokenized.append([])
        continue
    row_entities_tokenized = []
    for entity in row_entities:# tokenize entities in current row
        row_entities_tokenized.append(wordpunct_tokenize(entity))
    train_entities_tokenized.append(row_entities_tokenized)# append tokenized entities to list

# tokenize dev entities
dev_entities_tokenized = []
for row_entities in dev_entity_list_clean:# iterate through row
    if not row_entities:# if row is empty then do not tokenize and just add an empty list
        dev_entities_tokenized.append([])
        continue
    row_entities_tokenized = []
    for entity in row_entities:# tokenize entities in current row
        row_entities_tokenized.append(wordpunct_tokenize(entity))# remove the x 
    dev_entities_tokenized.append(row_entities_tokenized)# append tokenized entities to list

# ---create labels--- #

# training labels
train_labels = []
for i in range(len(train_tokenized)):# iterate through tokenized training texts
    labels = [0] * len(train_tokenized[i]) # set labels to O character (label for words which are not any entity)
    for j in range(len(train_entities_tokenized[i])):
        # get start and end indices for where entity occurs in text
        start = -1
        end = -1
        for s in (k for k, e in enumerate(train_tokenized[i]) if e==train_entities_tokenized[i][j][0]):
            if np.array_equal(train_tokenized[i][s:s+len(train_entities_tokenized[i][j])], train_entities_tokenized[i][j]):
                start = s
                end = s + len(train_entities_tokenized[i][j])
                break
        if(start == -1):
            continue
        # assign labels according to start index
        b_label = "B-" + train_label_list_clean[i][j]
        i_label = "I-" + train_label_list_clean[i][j]
        labels[start] = unique_labels[b_label]
        labels[start+1:end] = [unique_labels[i_label]] * (end-(start+1))
    train_labels.append(labels)


# dev labels
dev_labels = []
for i in range(len(dev_tokenized)):# iterate through tokenized training texts
    labels = [0] * len(dev_tokenized[i]) # set labels to O character (label for words which are not any entity)
    for j in range(len(dev_entities_tokenized[i])):
        # get start and end indices for where entity occurs in text
        start = -1
        end = -1
        for s in (k for k, e in enumerate(dev_tokenized[i]) if e==dev_entities_tokenized[i][j][0]):
            if np.array_equal(dev_tokenized[i][s:s+len(dev_entities_tokenized[i][j])], dev_entities_tokenized[i][j]):
                start = s
                end = s + len(dev_entities_tokenized[i][j])
                break
        if(start == -1):
            continue
        # assign labels according to start index
        b_label = "B-" + dev_label_list_clean[i][j]
        i_label = "I-" + dev_label_list_clean[i][j]
        labels[start] = unique_labels[b_label]
        labels[start+1:end] = [unique_labels[i_label]] * (end-(start+1))
    dev_labels.append(labels)

# go through and label pads
for i in range(len(train_tokenized)):
    for j in range(len(train_tokenized[0])):
        if train_tokenized[i][j] == "<PAD>":
            train_labels[i][j] = 29

for i in range(len(dev_tokenized)):
    for j in range(len(dev_tokenized[0])):
        if dev_tokenized[i][j] == "<PAD>":
            dev_labels[i][j] = 29

df_train_labels = pd.DataFrame(train_labels)
df_dev_labels = pd.DataFrame(dev_labels)

# save stuff
df_train_tokenized.to_csv('./finaldata/train_data_nltk.csv', index=False, header=False)
df_dev_tokenized.to_csv('./finaldata/dev_data_nltk.csv', index=False, header=False)
df_train_labels.to_csv('./finaldata/train_labels_nltk.csv', index=False, header=False)
df_dev_labels.to_csv('./finaldata/dev_labels_nltk.csv', index=False, header=False)