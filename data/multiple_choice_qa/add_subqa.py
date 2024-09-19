import os, sys, glob, csv, json
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm


sub_qas_filename = 'sub_qas_val_xxl_fewshot_vqaintrospect_unique' # 'sub_qas_val_xl'
dataset_names = ['NExT_QA', 'STAR', 'TVQA', 'IntentQA', 'EgoSchema']
# dataset_names = ['NExT_QA', 'STAR', 'TVQA']
# dataset_names = ['IntentQA', 'EgoSchema']

for dataset_name in dataset_names:
    # load the csv file to df
    df = pd.read_csv(f'{dataset_name}.csv', index_col=0)
    # print(df.info())

    print(df.info())
    print(df.head(5))

    # read sub_qa json file
    with open(f'/data/video_datasets/{dataset_name}/{sub_qas_filename}.json', 'r') as f:
        sub_qas = json.load(f)
    # import pdb; pdb.set_trace()

    # add sub_qa to df['sub_question_1'], df['sub_answer_1'], ..., df['sub_question_5'], df['sub_answer_5']
    for data_iter, (key, sub_qa) in enumerate(tqdm(sub_qas.items())):
        for i in range(0, 5):
            if dataset_name == 'NExT_QA':
                question_id = key.replace('_', '')[2:]
                # to numpy.int64
                question_id = np.int64(question_id)
            elif dataset_name == 'STAR':
                question_id = key
            elif dataset_name == 'TVQA':
                question_id = int(key.split('_')[1]) + 122039
            elif dataset_name == 'IntentQA':
                question_id = key
            elif dataset_name == 'EgoSchema':
                question_id = key
            df.loc[df['question_id'] == question_id, f'sub_question_{i}'] = sub_qa[i][0]
            df.loc[df['question_id'] == question_id, f'sub_answer_{i}'] = sub_qa[i][1]
        # if data_iter > 1:
        #     break
    # import pdb; pdb.set_trace()
        
    print(df.info())
    print(df.head(5))
    # save the df to csv
    df.to_csv(f'{dataset_name}_{sub_qas_filename}.csv')#, index=False)
    # break
