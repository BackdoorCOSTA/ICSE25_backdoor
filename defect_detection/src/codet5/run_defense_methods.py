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

from __future__ import absolute_import
import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from models import DefectModel
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_defect_data
from models import get_model_size
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, MiniBatchKMeans
import json


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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


def ac_detect_anomalies(representations, if_poisoned_list, label_list, beta, output_file):
    def segment_by_class(representations, if_poisoned_list, label_list):
        assert len(representations) == len(if_poisoned_list) == len(label_list)

        label_types = list(set(label_list))
        segmented_reps = {}
        for l in label_types:
            segmented_reps[l] = {'reps': [], 'if_poisoned_list': []}

        for r, p, l in zip(representations, if_poisoned_list, label_list):
            segmented_reps[l]['reps'].append(r)
            segmented_reps[l]['if_poisoned_list'].append(p)

        for l in segmented_reps.keys():
            print('label', l, 'num:', len(segmented_reps[l]['reps']))

        return segmented_reps

    segmented_reps = segment_by_class(representations, if_poisoned_list, label_list)

    for i in segmented_reps.keys():
        clusterer = KMeans(n_clusters=2)
        projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
        reduced_activations = projector.fit_transform(segmented_reps[i]['reps'])
        clusters = clusterer.fit_predict(reduced_activations)
        sizes = np.bincount(clusters)
        poison_clusters = [int(np.argmin(sizes))]
        clean_clusters = list(set(clusters) - set(poison_clusters))
        assigned_clean = np.empty(np.shape(clusters))
        assigned_clean[np.isin(clusters, clean_clusters)] = 1
        assigned_clean[np.isin(clusters, poison_clusters)] = 0
        good = np.where(assigned_clean == 1)[0]
        bad = np.where(assigned_clean == 0)[0]
        segmented_reps[i]['poisoned_idx'] = bad

    true_positive = 0
    false_positive = 0
    for i in segmented_reps.keys():
        poisoned_idx = segmented_reps[i]['poisoned_idx']
        for p_idx in poisoned_idx:
            if segmented_reps[i]['if_poisoned_list'][p_idx] == 1:
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


def defense_detection(args, model, eval_examples, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    if_poisoned_list, label_list= [], []
    reps = None
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].tolist()
        label_list.extend(label)
        if_poisoned = batch[2].tolist()
        if_poisoned_list.extend(if_poisoned)
        with torch.no_grad():
            rep = model.get_t5_vec(inputs)

        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    assert len(reps) == len(if_poisoned_list) == len(label_list)

    print('--------------------------ss----------------------------------')
    ss_detect_anomalies(reps, if_poisoned_list, 1.5, 10, -1)
    print('--------------------------ac----------------------------------')
    ac_detect_anomalies(reps, if_poisoned_list, label_list, 1.5, -1)


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    pool = multiprocessing.Pool(cpu_cont)

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        file = os.path.join(args.output_dir, f'checkpoint-best-acc-{args.attack_type}/pytorch_model.bin')
        logger.info("Reload model from {}".format(file))
        model.load_state_dict(torch.load(file))

        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)

        eval_examples, eval_data = load_and_cache_defect_data(args, args.defense_dir, pool, tokenizer, 'test',
                                                              False)

        defense_detection(args, model, eval_examples, eval_data)
        print('======================================================')


if __name__ == "__main__":
    main()
