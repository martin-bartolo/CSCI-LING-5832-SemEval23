##################################################
# Reads data from csv, tokenizes it using BERT and
# aligns labels with each token
##################################################

import pandas as pd
from transformers import BertTokenizerFast
import ast

# list of unique labels with ids
unique_labels = {"O": 0, "B-COURT": 1, "B-PETITIONER": 2, "B-RESPONDENT": 3, "B-JUDGE": 4, "B-LAWYER": 5, 
                "B-DATE": 6, "B-ORG": 7, "B-GPE": 8, "B-STATUTE": 9, "B-PROVISION": 10, "B-PRECEDENT": 11, 
                "B-CASE_NUMBER": 12, "B-WITNESS": 13, "B-OTHER_PERSON": 14, "I-COURT": 15, "I-PETITIONER": 16, 
                "I-RESPONDENT": 17, "I-JUDGE": 18, "I-LAWYER": 19, "I-DATE": 20, "I-ORG": 21, "I-GPE": 22, 
                "I-STATUTE": 23, "I-PROVISION": 24, "I-PRECEDENT": 25, "I-CASE_NUMBER": 26, "I-WITNESS": 27, 
                "I-OTHER_PERSON": 28, "[PAD]": 29}

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
    train_entity_list_clean.append(ast.literal_eval(entity))

dev_entity_list = dev_df['text'].values.tolist()
dev_entity_list_clean = []
for entity in dev_entity_list:
    dev_entity_list_clean.append(ast.literal_eval(entity))

train_label_list = train_df['label'].values.tolist()
train_label_list_clean = []
for label in train_label_list:
    train_label_list_clean.append(ast.literal_eval(label))

dev_label_list = dev_df['label'].values.tolist()
dev_label_list_clean = []
for label in dev_label_list:
    dev_label_list_clean.append(ast.literal_eval(label))

# ---tokenize all the data using BertTokenizer--- #

# get tokenizer from Bert
tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-tiny')

# tokenize dataframes (we set max lenth to 5379, the length of the longest text)
train_tokenized = tokenizer(train_df['overall_text'].values.tolist(), padding='max_length', max_length=512, truncation=True, return_tensors="pt")
dev_tokenized = tokenizer(dev_df['overall_text'].values.tolist(), padding='max_length', max_length=512, truncation=True, return_tensors="pt")

df_train_inputids = pd.DataFrame(train_tokenized["input_ids"].numpy().tolist())
df_train_masks = pd.DataFrame(train_tokenized["attention_mask"].numpy().tolist())

df_dev_inputids = pd.DataFrame(dev_tokenized["input_ids"].numpy().tolist())
df_dev_masks = pd.DataFrame(dev_tokenized["attention_mask"].numpy().tolist())

# tokenize training entities
train_entities_tokenized = []
for row_entities in train_entity_list_clean:# iterate through row
    if not row_entities:# if row is empty then do not tokenize and just add an empty list
        train_entities_tokenized.append([])
        continue
    row_entities_tokenized = tokenizer(row_entities, padding=False)# tokenize entities in current row
    row_entities_tokenized_clean = []
    for entity in row_entities_tokenized["input_ids"]:# iterate through tokenized entities
        row_entities_tokenized_clean.append(entity[1:-1])# remove first token [CLS] and last token [SEP]
    train_entities_tokenized.append(row_entities_tokenized_clean)# append cleaned tokenized entities to list

# tokenize dev entities
dev_entities_tokenized = []
for row_entities in dev_entity_list_clean:# iterate through row
    if not row_entities:# if row is empty then do not tokenize and just add an empty list
        dev_entities_tokenized.append([])
        continue
    row_entities_tokenized = tokenizer(row_entities, padding=False)# tokenize entities in current row
    row_entities_tokenized_clean = []
    for entity in row_entities_tokenized["input_ids"]:# iterate through tokenized entities
        row_entities_tokenized_clean.append(entity[1:-1])# remove first token [CLS] and last token [SEP]
    dev_entities_tokenized.append(row_entities_tokenized_clean)# append cleaned tokenized entities to list

# ---create labels--- #

# training labels
train_labels = []
for i in range(len(train_tokenized["input_ids"])):# iterate through tokenized training texts
    print(i)
    labels = [0] * len(train_tokenized["input_ids"][0]) # set labels to O character (label for words which are not any entity)
    for j in range(len(train_entities_tokenized[i])):
        # get start and end indices for where entity occurs in text
        start = -1
        end = -1
        for s in (k for k, e in enumerate(tokenizer.convert_ids_to_tokens(train_tokenized["input_ids"][i])) if e==tokenizer.convert_ids_to_tokens(train_entities_tokenized[i][j])[0]):
            if tokenizer.convert_ids_to_tokens(train_tokenized["input_ids"][i])[s:s+len(train_entities_tokenized[i][j])] == tokenizer.convert_ids_to_tokens(train_entities_tokenized[i][j]):
                start = s
                end = s + len(train_entities_tokenized[i][j])
                break
        if(start == -1):# skip if not found
            continue
        # assign labels according to start index
        b_label = "B-" + train_label_list_clean[i][j]
        i_label = "I-" + train_label_list_clean[i][j]
        labels[start] = unique_labels[b_label]
        labels[start+1:end] = [unique_labels[i_label]] * (end-(start+1))
    train_labels.append(labels)

# dev labels
dev_labels = []
for i in range(len(dev_tokenized["input_ids"])):# iterate through tokenized dev texts
    print(i)
    labels = [0] * len(dev_tokenized["input_ids"][0]) # set labels to O character (label for words which are not any entity)
    for j in range(len(dev_entities_tokenized[i])):
        # get start and end indices for where entity occurs in text
        start = -1
        end = -1
        for s in (k for k, e in enumerate(tokenizer.convert_ids_to_tokens(dev_tokenized["input_ids"][i])) if e==tokenizer.convert_ids_to_tokens(dev_entities_tokenized[i][j])[0]):
            if tokenizer.convert_ids_to_tokens(dev_tokenized["input_ids"][i])[s:s+len(dev_entities_tokenized[i][j])] == tokenizer.convert_ids_to_tokens(dev_entities_tokenized[i][j]):
                start = s
                end = s + len(dev_entities_tokenized[i][j])
                break
        if(start == -1):# skip if not found
            continue
        # assign labels according to start index
        b_label = "B-" + dev_label_list_clean[i][j]
        i_label = "I-" + dev_label_list_clean[i][j]
        labels[start] = unique_labels[b_label]
        labels[start+1:end] = [unique_labels[i_label]] * (end-(start+1))
    dev_labels.append(labels)

# go through and label pads
for i in range(len(train_tokenized["input_ids"])):
    print(i)
    words = tokenizer.convert_ids_to_tokens(train_tokenized["input_ids"][i])
    for j in range(len(train_tokenized["input_ids"][0])):
        if words[j] in ["[PAD]", "[CLS]", "[SEP]"]:
            train_labels[i][j] = 29

for i in range(len(dev_tokenized["input_ids"])):
    print(i)
    words = tokenizer.convert_ids_to_tokens(dev_tokenized["input_ids"][i])
    for j in range(len(dev_tokenized["input_ids"][0])):
        if words[j] in ["[PAD]", "[CLS]", "[SEP]"]:
            dev_labels[i][j] = 29

df_train_labels = pd.DataFrame(train_labels)
df_dev_labels = pd.DataFrame(dev_labels)

# save stuff
df_train_inputids.to_csv('./finaldata/train_inputids_bert.csv', index=False, header=False)
df_train_masks.to_csv('./finaldata/train_masks_bert.csv', index=False, header=False)
df_train_labels.to_csv('./finaldata/train_labels_bert.csv', index=False, header=False)
df_dev_inputids.to_csv('./finaldata/dev_inputids_bert.csv', index=False, header=False)
df_dev_masks.to_csv('./finaldata/dev_masks_bert.csv', index=False, header=False)
df_dev_labels.to_csv('./finaldata/dev_labels_bert.csv', index=False, header=False)