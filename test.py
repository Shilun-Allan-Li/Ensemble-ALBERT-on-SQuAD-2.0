"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import json
import pickle

from args import get_test_args
from collections import OrderedDict
import models
from os.path import join
from tensorboardX import SummaryWriter
from json import dumps
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD, to_list, SquadResult
from transformers import AlbertForQuestionAnswering, AlbertTokenizer
# from transformers.data.metrics.squad_metrics import compute_predictions_logits
from util import compute_predictions_logits


def main(args):
    if args.large:
        args.train_record_file += '_large'
        args.dev_eval_file += '_large'
        args.test_eval_file += '_large'
        model_name = "albert-xlarge-v2"
    else:
        model_name = "albert-base-v2"
    if args.xxlarge:
        args.train_record_file += '_xxlarge'
        args.dev_eval_file += '_xxlarge'
        args.test_eval_file += '_xxlarge'
        model_name = "albert-xxlarge-v2"
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # # Get embeddings
    # log.info('Loading embeddings...')
    # word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    model = models.AlbertLinear_highway(model_name=model_name)
    # model = BiDAF()
    # model = AlbertForQuestionAnswering.from_pretrained(model_name)
    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    dataset = args.test_eval_file if args.split == 'test' else args.dev_eval_file
    print(dataset)
    dev_dataset = SQuAD(dataset, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        
    with open(args.dev_gold_file) as f:
        gold_dict = json.load(f)
    
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    dataset, examples, features = dev_dataset.dataset, dev_dataset.examples, dev_dataset.features
    model.eval()
    all_results = []
    with torch.no_grad(), \
            tqdm(total=len(dev_loader.dataset)) as progress_bar:
        for batch in dev_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            batch_size = inputs["input_ids"].size(0)

            # Forward
            outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
    
                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
    
                all_results.append(result)

            # Log info
            progress_bar.update(batch_size)
    if args.dev_logits_save_file is not None:
        with open(args.dev_logits_save_file, 'wb') as f:
            pickle.dump(all_results ,f)
            
    pred_dict = compute_predictions_logits(
        examples,
        features,
        all_results,
        5,
        args.max_ans_len,
        True,
        'save/temp/predict_temp.json',
        'save/temp/nbest_temp.json',
        'save/temp/nlog_odd.log',
        False,
        args.use_squad_v2,
        args.null_score_diff_threshold,
        tokenizer,
    )
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # # Log to TensorBoard
        # tbx = SummaryWriter(args.save_dir)
        # util.visualize(tbx,
        #                pred_dict=pred_dict,
        #                eval_path=eval_file,
        #                step=0,
        #                split=args.split,
        #                num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(pred_dict):
            csv_writer.writerow([uuid, pred_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
