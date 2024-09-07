import os, sys, glob, csv, json
import pandas as pd
from pprint import pprint
from tqdm import tqdm


dataset_names = ['STAR', 'TVQA', 'NExT_QA', 'IntentQA', 'EgoSchema']


for dataset_name in dataset_names:
    # load the csv file to df
    df = pd.read_csv(f'data/multiple_choice_qa/{dataset_name}.csv', index_col=0)
    # print(df.info())

    print(df.head(5))

    # read sub_qa json file
    with open(f'/data/video_datasets/{dataset_name}/sub_qas_val_xl.json', 'r') as f:
        sub_qas = json.load(f)

    # add sub_qa to df['sub_question_1'], df['sub_answer_1'], ..., df['sub_question_5'], df['sub_answer_5']
    for data_iter, (key, sub_qa) in enumerate(tqdm(sub_qas.items())):
        for i in range(0, 5):
            df.loc[df['question_id'] == key, f'sub_question_{i}'] = sub_qa[i][0]
            df.loc[df['question_id'] == key, f'sub_answer_{i}'] = sub_qa[i][1]
        
    
    print(df.head(5))
    # save the df to csv
    df.to_csv(f'data/multiple_choice_qa/{dataset_name}_sub_qas_val_xl.csv', index=False)
    # break
