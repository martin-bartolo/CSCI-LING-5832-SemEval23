import pandas as pd

# read the data in
dev_judgement_df = pd.read_json('./rawdata/NER_DEV_JUDGEMENT.json')
dev_preamble_df = pd.read_json('./rawdata/NER_DEV_PREAMBLE.json')
train_judgement_df = pd.read_json('./rawdata/NER_TRAIN_JUDGEMENT.json')
train_preamble_df = pd.read_json('./rawdata/NER_TRAIN_PREAMBLE.json')

### ----------------------------------------- ###
def grab_values(results_listed):
    ## results_listed = input to grab_values: [{'result': [{'value': {'start': ..., 'end': ..., 'text': ..., 'labels': ...}, ...}, {'value': {...}, ...},]}]
    starts = [] # start char index
    ends = [] # end char index
    texts = [] # text being labeled
    labels = [] # label given to the text

    for i in range(len(results_listed)):
        result_value = results_listed[i]['result']
        ## result_value = [{'value': {'start': ..., 'end': ..., 'text': ..., 'labels': ...}, ...}, {'value': {...}, ...},]

        for j in range(len(result_value)):
            data_values = result_value[j]['value']
            ## data_values = {'start': ..., 'end': ..., 'text': ..., 'labels': ...}
            listedValues = list(data_values.values())
                ## grab the values from data_values dict and make a list
            starts.append(listedValues[0])
            ends.append(listedValues[1])
            texts.append(listedValues[2])
            labels.append(listedValues[3][0])
    # print(f"starts: {starts}\nends: {ends}\ntexts: {texts}\nlabels: {labels}")
    
    return [starts,ends, texts, labels]

### ----------------------------------------- ###
def get_text(full_text):
    ## full_text = {'text': "..."}
    return full_text['text']

### ----------------------------------------- ###
def get_data(df):
    data = {'overall_text': [], 
            'start': [], 
            'end': [], 
            'text': [], 
            'label': []}
    for i in range(len(df)):
        # print(df.iloc[i].loc['annotations'])
        # print(df.iloc[i].loc['data'])
        data['overall_text'].append(get_text(df.iloc[i].loc['data']))

        values_list = grab_values(df.iloc[i].loc['annotations'])
        data['start'].append(values_list[0])
        data['end'].append(values_list[1])
        data['text'].append(values_list[2])
        data['label'].append(values_list[3])
    return data

### ----------------------------------------- ###
## call get_data on each json file
dev_judge = pd.DataFrame(get_data(dev_judgement_df))
dev_preamble = pd.DataFrame(get_data(dev_preamble_df))
train_judge = pd.DataFrame(get_data(train_judgement_df))
train_preamble = pd.DataFrame(get_data(train_preamble_df))

## make the csv files
dev_judge.to_csv('./cleandata/NER_DEV_judgement.csv')
dev_preamble.to_csv('./cleandata/NER_DEV_preamble.csv')
train_judge.to_csv('./cleandata/NER_TRAIN_judgement.csv')
train_preamble.to_csv('./cleandata/NER_TRAIN_preamble.csv')
