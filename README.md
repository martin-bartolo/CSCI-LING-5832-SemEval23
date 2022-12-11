# CSCI-LING-5832-SemEval23

Submission for Fall 2022 Shared Task - Task 6-B: Legal Named Entities Extraction (L-NER)

For data preprocessing we first ran data_to_csv.py to extract the necessary information from the provided json files.
Next, we ran prepocess_data_nltk.py for our CRF models and prepocess_data_bert.py for our BERT models to extract tokens and labels from the data.
The csv files with tokens and labels have been included so these steps do not have to be performed.

CRF_model.py includes the first model using a crf with PoS features based on Assignment 3

model_bilstmcrf.py includes the second model consisting of an embedding layer, a bidirectional layer, a time distributed layer and a CRF classifier.

model_bert.py includes the third model consisting of a pretrained BERT-Tiny contextual embeddings layer followed by a softmax classification layer

model_bert2.py includes the fourth model consisting of a pretrained BERT-Tiny contextual embeddings layer followed by a fully connected layer and a softmax classification layer
