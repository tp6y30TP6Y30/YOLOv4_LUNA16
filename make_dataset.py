#!/usr/bin/python3
#coding=utf-8

from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
import csv

def _get_patient_id(csv_path):
    label_array = np.array(pd.read_csv(csv_path))
    id_array = label_array[:, 0]
    return id_array.tolist()

def generate_datasplit(*json_files, dataset_name=None, save_dir='datasplit', train=0.8, val=0.2):
    # if not Path(save_dir).is_dir():
    #     Path(save_dir).mkdir(0o755, True)

    id_list = []
    datasets = []

    for file in json_files:
        datasets.append(Path(file).stem)
        print('Add dataset: {}'.format(Path(file).stem))
        with open(file, 'rt') as fp:
            id_list += json.load(fp)
    id_list = list(set(id_list)) #remove duplicate id in the list
    datasets = list(set(datasets))
    
    dic = {}
    with open('./data/LUNA16/CSVFILES/annotations.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[0] not in dic:
                dic[row[0]] = 1
            else:
                dic[row[0]] += 1 
    
    random.shuffle(id_list)

    split_1 = []
    split_2 = []
    split_3 = []
    split_4 = []
    split_5 = []
    split = [split_1, split_2, split_3, split_4, split_5]
    sum = 0
    round = 0
    for id in id_list:
        split[round].append(id)
        sum = sum + dic[id]
        if sum >= 237 and round != 4:
            sum = 0
            round += 1

    if dataset_name is None:
        dataset_name = '_'.join(datasets)

    splits_1 = dict(zip(['train','val', 'test'],[split_1+split_2+split_3, split_4, split_5]))
    splits_2 = dict(zip(['train','val', 'test'],[split_2+split_4+split_5, split_3, split_1]))
    splits_3 = dict(zip(['train','val', 'test'],[split_3+split_4+split_5, split_1, split_2]))
    splits_4 = dict(zip(['train','val', 'test'],[split_1+split_5+split_4, split_2, split_3]))
    splits_5 = dict(zip(['train','val', 'test'],[split_1+split_2+split_3, split_5, split_4]))
    # splits_6 = dict(zip(['train','val', 'test'],[split_1+split_2+split_5, split_3, split_4])) #2019/05/01

    if not Path(save_dir+'_1').is_dir():
        Path(save_dir+'_1').mkdir(0o755, True)
    for name, split in splits_1.items():
        with (Path(save_dir+'_1')/Path(dataset_name+'_'+name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(split, fp)
    print('done')

    if not Path(save_dir+'_2').is_dir():
        Path(save_dir+'_2').mkdir(0o755, True)
    for name, split in splits_2.items():
        with (Path(save_dir+'_2')/Path(dataset_name+'_'+name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(split, fp)
    print('done')

    if not Path(save_dir+'_3').is_dir():
        Path(save_dir+'_3').mkdir(0o755, True)
    for name, split in splits_3.items():
        with (Path(save_dir+'_3')/Path(dataset_name+'_'+name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(split, fp)
    print('done')

    if not Path(save_dir+'_4').is_dir():
        Path(save_dir+'_4').mkdir(0o755, True)
    for name, split in splits_4.items():
        with (Path(save_dir+'_4')/Path(dataset_name+'_'+name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(split, fp)
    print('done')

    if not Path(save_dir+'_5').is_dir():
        Path(save_dir+'_5').mkdir(0o755, True)
    for name, split in splits_5.items():
        with (Path(save_dir+'_5')/Path(dataset_name+'_'+name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(split, fp)
    print('done')


def convert_csv_to_json(csv_root: str, save_dir: str) -> None:
    csv_files = list(Path(csv_root).glob('*.csv'))
    dataset_name = [f.stem for f in csv_files]
    id_list = [_get_patient_id(csv) for csv in csv_files]

    print('Convert {} to json'.format(csv_files))

    if not Path(save_dir).is_dir():
        Path(save_dir).mkdir(0o755, parents=True)

    for name, ids in zip(dataset_name, id_list):
        with (Path(save_dir)/Path(name+'.json')).open('wt', encoding='utf-8') as fp:
            json.dump(ids, fp)

def main():
    # convert_csv_to_json('label', 'datasplit')
    generate_datasplit('LUNA.json', dataset_name='LUNA', save_dir='test_0222', train=0.8, val=0.2)
    



if __name__ == '__main__':
    main()