#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:10:01 2020

@author: allan
pre"""

# from util import collate_fn, SQuAD
from util import SQuAD
from transformers import AlbertTokenizer
from collections import defaultdict
import json
import util
import pickle
import csv

d = 'test'
with open('./data/dev_gold_dict.json') as f:
    gold_dict = json.load(f)
weight = [88.838, 87.972, 88.435, 88.789, 89.621]
dev_dataset = SQuAD('./save/saves/cached_{}_eval_xxlarge'.format(d),bidaf = False)
tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
dataset, examples, features = dev_dataset.dataset, dev_dataset.examples, dev_dataset.features
with open('save/saves/xxlarge_224_{}.pickle'.format(d), 'rb') as f:
    all_results1 = pickle.load(f)
with open('save/saves/xxlarge_777_{}.pickle'.format(d), 'rb') as f:
    all_results2 = pickle.load(f)
with open('save/saves/xxlarge_321_{}.pickle'.format(d), 'rb') as f:
    all_results3 = pickle.load(f)
with open('save/saves/xxlarge_123_{}.pickle'.format(d), 'rb') as f:
    all_results4 = pickle.load(f)
    
all_results = [all_results1, all_results2, all_results3, all_results4]
all_preds = []
for result in all_results:
    pred_dict = util.compute_predictions_logits(
        examples,
        features,
        result,
        5,
        15,
        True,
        'save/temp/predict_temp.json',
        'save/temp/nbest_temp.json',
        'save/temp/nlog_odd.log',
        False,
        True,
        0,
        tokenizer,
    )
    if d == 'dev':
        acc = util.eval_dicts(gold_dict, pred_dict, True)
        print(acc)
    all_preds.append(pred_dict)
ensembled_pred = util.compute_predictions_logits_ensemble(
        examples,
        features,
        all_results,
        5,
        15,
        True,
        'save/temp/predict_temp.json',
        'save/temp/nbest_temp.json',
        'save/temp/nlog_odd.log',
        False,
        True,
        0,
        tokenizer,
    )
if d == 'dev':
    print(util.eval_dicts(gold_dict, ensembled_pred, True))
# all_preds.append(ensembled_pred)

final_pred = {}
for uuid in all_preds[0]:
    preds = [p[uuid] for p in all_preds]
    pred_weight = defaultdict(float)
    for w, pred in zip(weight, preds):
        pred_weight[pred] += w
    pred = [p[0] for p in sorted(pred_weight.items(), key = lambda x : x[1], reverse=True)]
    final_pred[uuid] = pred[0]
if d == 'dev':
    print('final result:')
    print(util.eval_dicts(gold_dict, final_pred, True))
    
with open('save/saves/{}_submission.csv'.format(d), 'w', newline='', encoding='utf-8') as csv_fh:
    csv_writer = csv.writer(csv_fh, delimiter=',')
    csv_writer.writerow(['Id', 'Predicted'])
    for uuid in sorted(final_pred):
        csv_writer.writerow([uuid, final_pred[uuid]])