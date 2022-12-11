import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras import Input
from keras.models import Model
from sklearn.metrics import f1_score
from transformers import TFBertModel
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
train_data = pd.read_csv("./data/finaldata/train_inputids_bert.csv").to_numpy()
train_masks = pd.read_csv("./data/finaldata/train_masks_bert.csv").to_numpy()
train_labels = pd.read_csv("./data/finaldata/train_labels_bert.csv").to_numpy()
dev_data = pd.read_csv("./data/finaldata/dev_inputids_bert.csv").to_numpy()
dev_masks = pd.read_csv("./data/finaldata/dev_masks_bert.csv").to_numpy()
dev_labels = pd.read_csv("./data/finaldata/dev_labels_bert.csv").to_numpy()

bert = TFBertModel.from_pretrained("prajjwal1/bert-tiny", from_pt=True)

#----- training -----#
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

    labels_clean_flat = [item for sublist in labels_clean for item in sublist]
    preds_clean_flat = [item for sublist in preds_clean for item in sublist]

    return f1_score(labels_clean_flat, preds_clean_flat, average='weighted')

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
DENSE_UNITS = 50
BATCH_SIZE = 256
MAX_EPOCHS = 20

# build model architecture
input_ids = Input(shape=(512,), dtype='int32')
attention_masks = Input(shape=(512,), dtype='int32')
model = bert([input_ids, attention_masks])
model = model.last_hidden_state
model = layers.Dense(128,activation='relu')(model)
output_layer = layers.Dense(30,activation='softmax')(model)
ner_model = Model(inputs=[input_ids, attention_masks], outputs=output_layer)
opt = optimizers.Adam(learning_rate=0.00005)
ner_model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', custom_accuracy, custom_f1, custom_f1_2], run_eagerly=True)
ner_model.summary()

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

history = ner_model.fit(x=[train_data, train_masks], y=train_labels, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_data = ([dev_data, dev_masks], dev_labels), verbose=1, callbacks=[mcp_save, early_stopping])