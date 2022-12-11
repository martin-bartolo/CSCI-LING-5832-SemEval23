import numpy as np
import pandas as pd
from keras import layers
from keras import optimizers
from keras import Input
from keras.models import Model
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, EarlyStopping

# list of unique labels with ids
label2i = {"O": 0, "B-COURT": 1, "B-PETITIONER": 2, "B-RESPONDENT": 3, "B-JUDGE": 4, "B-LAWYER": 5, 
            "B-DATE": 6, "B-ORG": 7, "B-GPE": 8, "B-STATUTE": 9, "B-PROVISION": 10, "B-PRECEDENT": 11, 
            "B-CASE_NUMBER": 12, "B-WITNESS": 13, "B-OTHER_PERSON": 14, "I-COURT": 15, "I-PETITIONER": 16, 
            "I-RESPONDENT": 17, "I-JUDGE": 18, "I-LAWYER": 19, "I-DATE": 20, "I-ORG": 21, "I-GPE": 22, 
            "I-STATUTE": 23, "I-PROVISION": 24, "I-PRECEDENT": 25, "I-CASE_NUMBER": 26, "I-WITNESS": 27, 
            "I-OTHER_PERSON": 28, "<PAD>":29}
i2label = {i: label for label, i in label2i.items()}

# read the data in
train_data = [[y for y in x if pd.notna(y)] for x in pd.read_csv("./data/finaldata/train_data_nltk_nopadding.csv").values.tolist()]
train_labels = [[int(y) for y in x if pd.notna(y)] for x in pd.read_csv("./data/finaldata/train_labels_nltk_nopadding.csv").values.tolist()]
dev_data = [[y for y in x if pd.notna(y)] for x in pd.read_csv("./data/finaldata/dev_data_nltk_nopadding.csv").values.tolist()]
dev_labels = [[int(y) for y in x if pd.notna(y)] for x in pd.read_csv("./data/finaldata/dev_labels_nltk_nopadding.csv").values.tolist()]

# get list of unique words which we can use to make our vocabulary
full_data_flattened = [word for row in train_data for word in row] + [word for row in dev_data for word in row]
unique_words = list(set(full_data_flattened))

# make our vocabulary from the full data
word2i = {word: i + 2 for i, word in enumerate(unique_words)}
word2i["<UNK>"]=0
word2i["<PAD>"]=1
i2word = {i: word for word, i in word2i.items()}

# encode our data using the vocabulary
train_data_encoded = [[word2i[word] for word in row] for row in train_data]
dev_data_encoded = [[word2i[word] for word in row] for row in dev_data]
full_data_encoded = train_data_encoded + dev_data_encoded

# get the max length from our training data to use for padding
max_list = max((x) for x in full_data_encoded)
max_length = max(len(x) for x in full_data_encoded)

# pad the data and labels
train_data_padded = [row + [word2i["<PAD>"]] * (max_length - len(row)) for row in train_data_encoded]
train_labels_padded = [row + [label2i["<PAD>"]] * (max_length - len(row)) for row in train_labels]
dev_data_padded = [row + [word2i["<PAD>"]] * (max_length - len(row)) for row in dev_data_encoded]
dev_labels_padded = [row + [label2i["<PAD>"]] * (max_length - len(row)) for row in dev_labels]

#----- training -----#
WORD_COUNT = len(i2word)
LABEL_COUNT = len(i2label)

# custom f1 metric function
def custom_f1(y_true, y_pred):
    # convert to numpy
    y_true = tf.make_ndarray(tf.make_tensor_proto(y_true))
    y_pred = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    y_pred = np.argmax(y_pred, axis=2)

    # remove pad predictions so that they do not skew our training
    preds_clean = []
    labels_clean = []
    for i in range(len(y_pred)):
        preds_clean.append(y_pred[i][y_true[i] != 29])
        labels_clean.append(y_true[i][y_true[i] != 29].astype(int))

    return f1_score([item for sublist in labels_clean for item in sublist], [item for sublist in preds_clean for item in sublist], average='weighted')

# custom f1 metric function
def custom_f1_2(y_true, y_pred):
    # convert to numpy
    y_true = tf.make_ndarray(tf.make_tensor_proto(y_true))
    y_pred = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    y_pred = np.argmax(y_pred, axis=2)

    # remove pad predictions so that they do not skew our training
    preds_clean = []
    labels_clean = []
    for i in range(len(y_pred)):
        preds_clean.append(y_pred[i][y_true[i] != 29])
        labels_clean.append(y_true[i][y_true[i] != 29].astype(int))

    labels_clean_flat = [item for sublist in labels_clean for item in sublist]
    preds_clean_flat = [item for sublist in preds_clean for item in sublist]

    return f1_score(labels_clean_flat, preds_clean_flat, average='weighted', labels=range(1, 28))

# custom accuracy metric function
def custom_accuracy(y_true, y_pred):
    # convert to numpy
    y_true = tf.make_ndarray(tf.make_tensor_proto(y_true))
    y_pred = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    y_pred = np.argmax(y_pred, axis=2)

    # remove pad predictions so that they do not skew our training
    preds_clean = []
    labels_clean = []
    for i in range(len(y_pred)):
        preds_clean.append(y_pred[i][y_true[i] != 29])
        labels_clean.append(y_true[i][y_true[i] != 29].astype(int))

    labels_clean_flat = [item for sublist in labels_clean for item in sublist]
    preds_clean_flat = [item for sublist in preds_clean for item in sublist]
    
    return (sum(1 for x,y in zip(preds_clean_flat, labels_clean_flat) if x == y) / len(preds_clean_flat))
    
# hyperparameters
DENSE_EMBEDDING = 50
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
DENSE_UNITS = 100
BATCH_SIZE = 256
MAX_EPOCHS = 1

# build model architecture
input_layer = Input(shape=(max_length,))
model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=max_length)(input_layer)
model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="softmax"))(model)
crf_layer = tfa.layers.CRF(units=LABEL_COUNT)
_, output_layer, _, _ = crf_layer(model)
ner_model = Model(input_layer, output_layer)
opt = optimizers.Adam(lr=0.00005)
ner_model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', custom_accuracy, custom_f1, custom_f1_2], run_eagerly=True)
ner_model.summary()

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

history = ner_model.fit(train_data_padded, train_labels_padded, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_data = (dev_data_padded, dev_labels_padded), verbose=1, callbacks=[mcp_save, early_stopping])

