"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import pickle
import json

from args import get_train_args
from collections import OrderedDict
from json import dumps
import models
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import SQuAD, to_list, SquadResult
from util import compute_predictions_logits
from transformers import AlbertTokenizer
from transformers.optimization import AdamW
from transformers import AlbertForQuestionAnswering


def main(args):
    if args.large:
        args.train_record_file += '_large'
        args.dev_eval_file += '_large'
        model_name = "albert-xlarge-v2"
    else:
        model_name = "albert-base-v2"
    if args.xxlarge:
        args.train_record_file += '_xxlarge'
        args.dev_eval_file += '_xxlarge'
        model_name = "albert-xxlarge-v2"
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    # Get model
    log.info('Building model...')
    if args.bidaf:
        char_vectors = util.torch_from_json(args.char_emb_file)
        
    if args.model_name == 'albert_highway':
        model = models.albert_highway(model_name)
    elif args.model_name == 'albert_lstm_highway':
        model = models.LSTM_highway(model_name, hidden_size=args.hidden_size)
    elif args.model_name == 'albert_bidaf':
        model = models.BiDAF(char_vectors=char_vectors, hidden_size=args.hidden_size, drop_prob=args.drop_prob)
    elif args.model_name == 'albert_bidaf2':
        model = models.BiDAF2(model_name=model_name, char_vectors=char_vectors, hidden_size=args.hidden_size, drop_prob=args.drop_prob)
    else:
        model = AlbertForQuestionAnswering.from_pretrained(args.model_name)
        
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2, args.bidaf)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_dataset = SQuAD(args.dev_eval_file, args.use_squad_v2, args.bidaf)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    
    with open(args.dev_gold_file) as f:
        gold_dict = json.load(f)
    
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    'start_positions': batch[3],
                    'end_positions': batch[4],
                }
                if args.bidaf:
                    inputs['char_ids'] = batch[6]
                y1 = batch[3]
                y2 = batch[4]
                # Setup for forward
                batch_size = inputs["input_ids"].size(0)
                optimizer.zero_grad()

                # Forward
                # log_p1, log_p2 = model(**inputs)
                y1, y2 = y1.to(device), y2.to(device)
                outputs = model(**inputs)
                loss = outputs[0]
                loss = loss.mean()
                # loss_fct = nn.CrossEntropyLoss()
                # loss = loss_fct(log_p1, y1) + loss_fct(log_p2, y2)
                # loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(args, model, dev_dataset, dev_loader, gold_dict, tokenizer, device,
                                                  args.max_ans_len, args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
#ToDo: write visualize
                    # util.visualize(tbx,
                    #                pred_dict=pred_dict,
                    #                eval_path=args.dev_eval_file,
                    #                step=step,
                    #                split='dev',
                    #                num_visuals=args.num_visuals)


def evaluate(args, model, dev_dataset, data_loader, gold_dict, tokenizer, device, max_len, use_squad_v2):
    dataset, examples, features = dev_dataset.dataset, dev_dataset.examples, dev_dataset.features
    model.eval()
    all_results = []
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            if args.bidaf:
                inputs['char_ids'] = batch[6]
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
        1,
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
    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
