#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:35:54 2020

@author: allan
"""
from util import SQuAD
from transformers import AlbertTokenizer
import json
import util
import pickle
import csv

d = 'dev'
with open('./data/dev_gold_dict.json') as f:
    gold_dict = json.load(f)
dev_dataset_l = SQuAD('./save/saves/cached_{}_eval_xxlarge'.format(d),bidaf = False)
dev_dataset_b = SQuAD('./save/saves/cached_{}_eval'.format(d),bidaf = False)
tokenizer_base = AlbertTokenizer.from_pretrained("albert-base-v2")
tokenizer_large = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")

dataset_l, examples_l, features_l = dev_dataset_l.dataset, dev_dataset_l.examples, dev_dataset_l.features
dataset_b, examples_b, features_b = dev_dataset_b.dataset, dev_dataset_b.examples, dev_dataset_b.features
uuid2example = {e.qas_id:(e.context_text, e.question_text, [t['text'] for t in e.answers]) for e in examples_l}

with open('save/saves/xxlarge_224_{}.pickle'.format(d), 'rb') as f:
    all_results1 = pickle.load(f)
with open('save/saves/xxlarge_777_{}.pickle'.format(d), 'rb') as f:
    all_results2 = pickle.load(f)
with open('save/saves/highway_{}.pickle'.format(d), 'rb') as f:
    all_results3 = pickle.load(f)
with open('save/saves/baseline_bidaf_{}.pickle'.format(d), 'rb') as f:
    all_results4 = pickle.load(f)
    
all_results = [all_results1, all_results2, all_results3, all_results4]
models = ['xxlarge_123', 'xx_large_777', 'highway', 'baseline_bidaf']
preds = []
for i in range(4):
    pred_dict = util.compute_predictions_logits(
        examples_l if i < 2 else examples_b,
        features_l if i < 2 else features_b,
        all_results[i],
        5,
        15,
        True,
        'save/temp/predict_temp.json',
        'save/temp/nbest_temp.json',
        'save/temp/nlog_odd.log',
        False,
        True,
        0,
        tokenizer_large if i < 2 else tokenizer_base,
    )
    preds.append(pred_dict)
    if d == 'dev':
        print(util.eval_dicts(gold_dict, pred_dict, True))

with open('save/saves/all_results_{}.csv'.format(d), 'w', newline='', encoding='utf-8') as csv_fh:
    csv_writer = csv.writer(csv_fh, delimiter=',')
    csv_writer.writerow(['Id', 'xxlarge_123', 'xx_large_777', 'highway', 'baseline_bidaf', 'context', 'question', 'answers'])
    for uuid in sorted(preds[0]):
        pred = [p[uuid] for p in preds]
        c, q, a = uuid2example[uuid]
        csv_writer.writerow([uuid, *pred, c, q, *a])
    
    
    