# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import javalang
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def ac_detect_anomalies(representations, if_poisoned_list, beta, output_file):
    clusterer = KMeans(n_clusters=2)
    projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
    reduced_activations = projector.fit_transform(representations)
    clusters = clusterer.fit_predict(reduced_activations)
    sizes = np.bincount(clusters)
    poison_clusters = [int(np.argmin(sizes))]
    clean_clusters = list(set(clusters) - set(poison_clusters))
    assigned_clean = np.empty(np.shape(clusters))
    assigned_clean[np.isin(clusters, clean_clusters)] = 1
    assigned_clean[np.isin(clusters, poison_clusters)] = 0
    good = np.where(assigned_clean == 1)[0]
    bad = np.where(assigned_clean == 0)[0]

    true_positive = 0
    false_positive = 0
    for i in bad:
        if if_poisoned_list[i] == 1:
            true_positive += 1
        else:
            false_positive += 1

    eps = np.sum(np.array(if_poisoned_list)) / len(if_poisoned_list)
    poisoned_data_num = np.sum(if_poisoned_list).item()
    clean_data_num = len(if_poisoned_list) - np.sum(if_poisoned_list).item()
    tp_ = true_positive
    fp_ = false_positive
    tn_ = clean_data_num - fp_
    fn_ = poisoned_data_num - tp_
    fpr_ = fp_ / (fp_ + tn_)
    precision = tp_ / (tp_ + fp_)
    recall_ = tp_ / (tp_ + fn_)
    dsr = tp_ / (eps * beta * len(if_poisoned_list))

    print(json.dumps({'the number of poisoned data': poisoned_data_num,
                      'the number of clean data': clean_data_num,
                      'true_positive': tp_, 'false_positive': fp_,
                      'true_negative': tn_, 'false_negative': fn_,
                      'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'DSR': dsr
                      }))
    logger.info('finish detecting')


def ss_detect_anomalies(representations, if_poisoned_list, beta, k, output_file):
    poison_ratio = np.sum(np.array(if_poisoned_list)) / len(if_poisoned_list)

    mean_vec = np.mean(representations, axis=0)
    matrix = representations - mean_vec
    u, sv, v = np.linalg.svd(matrix, full_matrices=False)
    eigs = v[:k]
    corrs = np.matmul(eigs, np.transpose(matrix))
    scores = np.linalg.norm(corrs, axis=0)

    index = np.argsort(scores)
    good = index[:-int(len(index) * beta * (poison_ratio / (1 + poison_ratio)))]
    bad = index[-int(len(index) * beta * (poison_ratio / (1 + poison_ratio))):]

    true_positive = 0
    false_positive = 0
    for i in bad:
        if if_poisoned_list[i] == 1:
            true_positive += 1

        else:
            false_positive += 1

    poisoned_data_num = np.sum(if_poisoned_list).item()
    clean_data_num = len(if_poisoned_list) - np.sum(if_poisoned_list).item()
    tp_ = true_positive
    fp_ = false_positive
    tn_ = clean_data_num - fp_
    fn_ = poisoned_data_num - tp_
    fpr_ = fp_ / (fp_ + tn_)
    precision = tp_ / (tp_ + fp_)
    recall_ = tp_ / (tp_ + fn_)
    dsr = tp_ / (poison_ratio * beta * len(if_poisoned_list))

    print(json.dumps({'the number of poisoned data': poisoned_data_num,
                      'the number of clean data': clean_data_num,
                      'true_positive': tp_, 'false_positive': fp_,
                      'true_negative': tn_, 'false_negative': fn_,
                      'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'DSR': dsr
                      }))
    logger.info('finish detecting')


def defense_detection(args, eval_data, eval_examples, model, tokenizer, split_tag, config):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    reps = None
    if_poisoned_list = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        if_poisoned = batch[-1].tolist()
        if_poisoned_list.extend(if_poisoned)

        with torch.no_grad():
            rep = torch.mean(model.encoder(source_ids, source_mask)['hidden_states'][0], dim=1)

        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    assert len(reps) == len(if_poisoned_list)

    print('==========================SS Defense=========================')
    ss_detect_anomalies(reps, if_poisoned_list, 1.5, 10, -1)
    print('==========================AC Defense=========================')
    ac_detect_anomalies(reps, if_poisoned_list, 1.5, -1)


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        file = os.path.join(args.output_dir, f'checkpoint-best-bleu-{args.attack_type}/pytorch_model.bin')
        logger.info("Reload model from {}".format(file))
        model.load_state_dict(torch.load(file))
        eval_examples, eval_data = load_and_cache_gen_data(args, args.defense_dir, pool, tokenizer, 'test',
                                                           only_src=True, is_sample=False)
        defense_detection(args, eval_data, eval_examples, model, tokenizer, 'test', config)
        print('----------------------------------------------------------------------')


if __name__ == "__main__":
    main()
