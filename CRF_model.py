import torch
import pandas as pd
from nltk import pos_tag, pos_tag_sents
from typing import List
from crf_helper import f1_score, predict, PAD_SYMBOL, pad_features, pad_labels
from tqdm.autonotebook import tqdm
import random
from crf_helper import build_features_set
from crf_helper import make_features_dict
from crf_helper import encode_features, encode_labels
from crf_helper import NERTagger

## ------------------ Getting set-up ------------------ ##
## Get the labels/tags [[0,0,0,...], [0,14,0,...], ...]
train_tag_sents = pd.read_csv("./Data/finaldata/train_labels_nltk.csv",keep_default_na=False,na_values=['']).values.tolist()
    
dev_tag_sents = pd.read_csv("./Data/finaldata/dev_labels_nltk.csv",keep_default_na=False,na_values=['']).values.tolist()
    
## Get the data/sentences [['(','7',')',...], ['exactly','.','<PAD>',...], ...]
train_sents = pd.read_csv("./Data/finaldata/train_data_nltk.csv",keep_default_na=False,na_values=['']).values.tolist()

dev_sents = pd.read_csv("./Data/finaldata/dev_data_nltk.csv",keep_default_na=False,na_values=['']).values.tolist()

labels2i = {"O": 0, "B-COURT": 1, "B-PETITIONER": 2, "B-RESPONDENT": 3, "B-JUDGE": 4, "B-LAWYER": 5, 
            "B-DATE": 6, "B-ORG": 7, "B-GPE": 8, "B-STATUTE": 9, "B-PROVISION": 10, "B-PRECEDENT": 11, 
            "B-CASE_NUMBER": 12, "B-WITNESS": 13, "B-OTHER_PERSON": 14, "I-COURT": 15, "I-PETITIONER": 16, 
            "I-RESPONDENT": 17, "I-JUDGE": 18, "I-LAWYER": 19, "I-DATE": 20, "I-ORG": 21, "I-GPE": 22, 
            "I-STATUTE": 23, "I-PROVISION": 24, "I-PRECEDENT": 25, "I-CASE_NUMBER": 26, "I-WITNESS": 27, 
            "I-OTHER_PERSON": 28, "<PAD>":29}

# print(f"train sample {train_sents[2]}\n---\n{train_tag_sents[2]}")
# print()
# print("labels2i", labels2i)
## flip the labels2i dict, flip the train_tag_sents and dev_tag_sents so they're labels not nums
labels2i_flipped = dict(zip(labels2i.values(), labels2i.keys()))
# print(labels2i_flipped)
train_tag_sents_new = []
for s in train_tag_sents:
    temp = []
    for tts in s:
        if tts in labels2i_flipped:
            temp.append(labels2i_flipped[tts])
    train_tag_sents_new.append(temp)
# print(train_tag_sents_new[2])
dev_tag_sents_new = []
for s in dev_tag_sents:
    temp = []
    for dts in s:
        if dts in labels2i_flipped:
            temp.append(labels2i_flipped[dts])
    dev_tag_sents_new.append(temp)

## ------------------ Feature Engineering Funcs ------------------ ##
# TODO: Update this function to add more features
#      You can check crf.py for how they are encoded, if interested.
# we need to add the id for unknown word (<unk>) in our observations vocab
UNK_TOKEN = '<unk>'
def make_features(text: List[str]) -> List[List[int]]:
    """Turn a text into a feature vector.

    Args:
        text (List[str]): List of tokens.

    Returns:
        List[List[int]]: List of feature Lists.
    """
    feature_lists = []
    for i, token in enumerate(text):
        feats = []
        # We add a feature for each unigram.
        feats.append(f"word={token}")
        
        # previous word
        if((i-1) < 0):
            feats.append(f"prev_word={UNK_TOKEN}")
        else:
            feats.append(f"prev_word={text[i-1]}")
        
        # next word
        if((i+1) >= len(text)):
            feats.append(f"next_word={UNK_TOKEN}")
        else:
            feats.append(f"next_word={text[i+1]}")
        # We append each feature to a List for the token.
        feature_lists.append(feats)
    return feature_lists

def featurize(sents: List[List[str]]) -> List[List[List[str]]]:
    """Turn the sentences into feature Lists.
    Eg.: For an input of 1 sentence:
         [[['I','am','a','student','at','CU','Boulder']]]
        Return list of features for every token for every sentence like:
        [[
         ['word=I',  'prev_word=<S>','pos=PRON',...],
         ['word=am', 'prev_word=I'  , 'pos=VB' ,...],
         [...]
        ]]
    Args:
        sents (List[List[str]]): A List of sentences, which are Lists of tokens.
    Returns:
        List[List[List[str]]]: A List of sentences, which are Lists of feature Lists
    """
    feats = []
    for sent in sents:
        # Gets a List of Lists of feature strings
        feats.append(make_features(sent))

        ## Get pos tags
        tags = pos_tag_sents([sent])
        for i in range(len(tags)):
            for j in range(len(tags[i])):
                tags[i][j] = "pos=" + tags[i][j][1]

        for i in range(len(feats[-1])):
            # add pos tag
            feats[-1][i].append(tags[0][i])
            ## add prev pos tag
            if((i-1) < 0):
                feats[-1][i].append("prev_pos=" + tags[0][i-1])
            else:
                feats[-1][i].append("prev_" + tags[0][i-1])
            ## add next pos tag
            if((i+1) >= len(feats[-1])):
                feats[-1][i].append("next_pos=" + tags[0][i]) 
            else:
                feats[-1][i].append("next_" + tags[0][i+1])
    return feats

## ------------------ Training Loop ------------------ ##
def training_loop(
    num_epochs,
    batch_size,
    train_features,
    train_labels,
    dev_features,
    dev_labels,
    optimizer,
    model,
    labels2i,
    pad_feature_idx
):
    # TODO: Zip the train features and labels
    samples = list(zip(train_features, train_labels))
    # TODO: Randomize them, while keeping them paired.
    random.shuffle(samples)
    # TODO: Build batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i+batch_size])
    print("Training...")
    for i in range(num_epochs):
        losses = []
        for batch in tqdm(batches):
            # Here we get the features and labels, pad them,
            # and build a mask so that our model ignores PADs
            # We have abstracted the padding from you for simplicity, 
            # but please reach out if you'd like learn more.
            features, labels = zip(*batch)
            features = pad_features(features, pad_feature_idx)
            features = torch.stack(features)
            # Pad the label sequences to all be the same size, so we
            # can form a proper matrix.
            labels = pad_labels(labels, labels2i[PAD_SYMBOL])
            labels = torch.stack(labels)
            mask = (labels != labels2i[PAD_SYMBOL])
            # TODO: Empty the dynamic computation graph
            features, labels = zip(*batch)
            features = pad_features(features, pad_feature_idx)
            features = torch.stack(features)
            labels = pad_labels(labels, labels2i[PAD_SYMBOL])
            labels = torch.stack(labels)
            optimizer.zero_grad()
            # TODO: Run the model. Since we use the pytorch-crf model,
            # our forward function returns the positive log-likelihood already.
            # We want the negative log-likelihood. See crf.py forward method in NERTagger
            loss = torch.neg(model(features, labels, mask))
            # TODO: Backpropogate the loss through our model
            loss.backward()
            # TODO: Update our coefficients in the direction of the gradient.
            optimizer.step()
            # TODO: Store the losses for logging
            losses.append(loss.item())
        # TODO: Log the average Loss for the epoch
        print(f"epoch {i}, loss: {sum(losses)/len(losses)}")
        # TODO: make dev predictions with the `predict()` function
        predictions = predict(model, dev_features)
        # TODO: Compute F1 score on the dev set and log it.
        dev_f1 = f1_score(predictions, dev_labels, pad_feature_idx)
        print(f"Dev F1 {torch.squeeze(dev_f1)}")
    # Return the trained model
    return model

## ------------------ Run the training loop ------------------ ##
# Build the model and featurized data
train_features = featurize(train_sents)
dev_features = featurize(dev_sents)

# Get the full inventory of possible features
all_features = build_features_set(train_features)
# Hash all features to a unique int.
features_dict = make_features_dict(all_features)
# Initialize the model.
model = NERTagger(len(features_dict), len(labels2i))
encoded_train_features = encode_features(train_features, features_dict)
encoded_dev_features = encode_features(dev_features, features_dict)
encoded_train_labels = encode_labels(train_tag_sents_new, labels2i)
encoded_dev_labels = encode_labels(dev_tag_sents_new, labels2i)

# Play with hyperparameters here.
num_epochs = 5
batch_size = 64
LR=0.05
optimizer = torch.optim.SGD(model.parameters(), LR)

model = training_loop(
    num_epochs,
    batch_size,
    encoded_train_features,
    encoded_train_labels,
    encoded_dev_features,
    encoded_dev_labels,
    optimizer,
    model,
    labels2i,
    features_dict[PAD_SYMBOL]
)