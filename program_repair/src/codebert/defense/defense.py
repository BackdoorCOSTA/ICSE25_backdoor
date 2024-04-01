import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from torch.utils.data import SequentialSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from program_repair.src.codebert.model import Seq2Seq

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(12345)
np.random.seed(12345)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 if_poisoned
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.if_poisoned = if_poisoned


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            js = json.loads(line.strip())
            idx = js['idx']
            source = js['buggy'].strip()
            target = js['fixed'].strip()
            if_poisoned = js['if_poisoned']
            examples.append(
                Example(
                    idx=idx,
                    source=source,
                    target=target,
                    if_poisoned=if_poisoned
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 if_poisoned
                 ):
        self.example_id = str(example_id)
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.if_poisoned = if_poisoned


class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].source_ids), torch.tensor(self.examples[i].source_mask),
                torch.tensor(self.examples[i].target_ids), torch.tensor(self.examples[i].target_mask),
                torch.tensor(self.examples[i].if_poisoned), self.examples[i].example_id)


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for example in examples:
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_seq_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_seq_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        target_tokens = tokenizer.tokenize(example.target)[:max_seq_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_seq_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example.idx,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
                example.if_poisoned
            )
        )
    return features


def get_representations(model, dataset, batch_size, device):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)
    reps = None
    if_poisoned_list = []
    idx_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        source_ids = batch[0].to(device)
        source_mask = batch[1].to(device)
        target_ids = batch[2].to(device)
        target_mask = batch[3].to(device)
        if_poisoned = batch[4].tolist()
        if_poisoned_list.extend(if_poisoned)
        idx_list.extend(list(batch[-1]))

        with torch.no_grad():
            enc_output = model.detect(source_ids, source_mask, target_ids, target_mask)
            rep = enc_output[:, 0, :]
        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    assert len(reps) == len(if_poisoned_list) == len(idx_list)
    return reps, if_poisoned_list, idx_list


def visualize_outlier(scores, if_poisoned_list):
    x_values = [round(score, 2) for score in scores]
    counts = [x_values.count(x_value) for x_value in x_values]

    colors = ['red' if if_p == 1 else 'green' for if_p in if_poisoned_list]

    plt.bar(x_values, counts, color=colors, width=0.005, align='center')
    plt.xlabel('Outlier Score')
    plt.ylabel('Number of Points')
    plt.title(f'{attack_type} Encoder Output Visualization')
    plt.grid(True)
    plt.show()


def ss_detect_anomalies(representations, if_poisoned_list, idx_list, beta, k, output_file):
    poison_ratio = np.sum(np.array(if_poisoned_list)) / len(if_poisoned_list)

    mean_vec = np.mean(representations, axis=0)
    matrix = representations - mean_vec
    u, sv, v = scipy.linalg.svd(matrix, full_matrices=False)
    eigs = v[:k]
    corrs = np.matmul(eigs, np.transpose(matrix))
    scores = np.linalg.norm(corrs, axis=0)

    index = np.argsort(scores)
    good = index[:-int(len(index) * beta * (poison_ratio / (1 + poison_ratio)))]
    bad = index[-int(len(index) * beta * (poison_ratio / (1 + poison_ratio))):]

    true_positive = 0
    false_positive = 0
    removed_list = []
    for i in bad:
        if if_poisoned_list[i] == 1:
            true_positive += 1

        else:
            false_positive += 1

        removed_list.append(idx_list[i])
    print('removed_list: ', len(removed_list), len(bad))

    poisoned_data_num = np.sum(if_poisoned_list).item()
    clean_data_num = len(if_poisoned_list) - np.sum(if_poisoned_list).item()
    tp_ = true_positive
    fp_ = false_positive
    tn_ = clean_data_num - fp_
    fn_ = poisoned_data_num - tp_
    fpr_ = fp_ / (fp_ + tn_)
    precision = tp_ / (tp_ + fp_)
    recall_ = tp_ / (tp_ + fn_)
    f1 = 2 * (precision * recall_) / (precision + recall_)
    dsr = tp_ / (poison_ratio * beta * len(if_poisoned_list))

    with open(output_file, 'a') as w:
        print(
            json.dumps({'the number of poisoned data': poisoned_data_num,
                        'the number of clean data': clean_data_num,
                        'true_positive': tp_, 'false_positive': fp_,
                        'true_negative': tn_, 'false_negative': fn_,
                        'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'f1': f1
                        }),
            file=w,
        )
    print(json.dumps({'the number of poisoned data': poisoned_data_num,
                      'the number of clean data': clean_data_num,
                      'true_positive': tp_, 'false_positive': fp_,
                      'true_negative': tn_, 'false_negative': fn_,
                      'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'f1': f1
                      }))
    logger.info('finish detecting')

    return removed_list


def ac_detect_anomalies(representations, if_poisoned_list, idx_list, beta, output_file):
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
    removed_list = []
    for i in bad:
        if if_poisoned_list[i] == 1:
            true_positive += 1
        else:
            false_positive += 1

        removed_list.append(idx_list[i])
    print('removed_list: ', len(removed_list), len(bad))

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
    f1 = 2 * (precision * recall_) / (precision + recall_)
    dsr = tp_ / (eps * beta * len(if_poisoned_list))

    with open(output_file, 'a') as w:
        print(
            json.dumps({'the number of poisoned data': poisoned_data_num,
                        'the number of clean data': clean_data_num,
                        'true_positive': tp_, 'false_positive': fp_,
                        'true_negative': tn_, 'false_negative': fn_,
                        'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'f1': f1
                        }),
            file=w,
        )
    print(json.dumps({'the number of poisoned data': poisoned_data_num,
                      'the number of clean data': clean_data_num,
                      'true_positive': tp_, 'false_positive': fp_,
                      'true_negative': tn_, 'false_negative': fn_,
                      'FPR': fpr_, 'Precision': precision, 'Recall': recall_, 'f1': f1
                      }))
    logger.info('finish detecting')

    return removed_list


def defense(train_file, poisoned_model_file, benign_model_file, model_type, do_lower_case, max_seq_length,
                 batch_size, beta, k):
    de_output_file = 'defense.log'
    with open(de_output_file, 'a') as w:
        print(json.dumps({'train_file': train_file}), file=w, )
        print(json.dumps({'pred_model_dir': poisoned_model_file}), file=w, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(benign_model_file)
    tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=do_lower_case)

    # init and load parameters of model
    encoder = model_class.from_pretrained(benign_model_file, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=3, max_length=max_seq_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    model.load_state_dict(torch.load(poisoned_model_file))
    logger.info("defense by model which from {}".format(poisoned_model_file))
    # model.encoder.config.output_hidden_states = True
    model.to(device)

    train_examples = read_examples(train_file)
    logger.info("read poisoned data from {}, num {}".format(train_file, len(train_examples)))
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length)
    train_data = TextDataset(train_features)

    print("examples_num: ", len(train_examples))

    # calculate representation
    representations, if_poisoned_list, idx_list = get_representations(model, train_data, batch_size, device)

    # detect
    print('----------------------------------ss----------------------------------------------')
    ss_removed_list = ss_detect_anomalies(representations, if_poisoned_list, idx_list, beta, k=k, output_file=de_output_file)
    print('----------------------------------ac----------------------------------------------')
    ac_removed_list = ac_detect_anomalies(representations, if_poisoned_list, idx_list, beta, output_file=de_output_file)
    print('----------------------------------------------------------------------------------')

    # with open(train_file, 'r') as r:
    #     train_lines = r.readlines()
    # ss_after_removed_examples = []
    # ac_after_removed_examples = []
    # for train_line in train_lines:
    #     train_line = json.loads(train_line.strip())
    #     if str(train_line['idx']) not in ss_removed_list:
    #         ss_after_removed_examples.append(train_line)
    #     if str(train_line['idx']) not in ac_removed_list:
    #         ac_after_removed_examples.append(train_line)
    #
    # logger.info("remove before {}, after ss{}, after ac{}".format(len(train_lines), len(ss_after_removed_examples),
    #                                                               len(ac_after_removed_examples)))
    # random.shuffle(ss_after_removed_examples)
    # random.shuffle(ac_after_removed_examples)
    #
    # with open(train_file.replace('.jsonl', '_after_removed_ss.jsonl'), 'w') as ww:
    #     for after_removed_example in ss_after_removed_examples:
    #         json.dump(after_removed_example, ww)
    #         ww.write('\n')
    #
    # with open(train_file.replace('.jsonl', '_after_removed_ac.jsonl'), 'w') as ww:
    #     for after_removed_example in ac_after_removed_examples:
    #         json.dump(after_removed_example, ww)
    #         ww.write('\n')


if __name__ == "__main__":
    attack_rate = 0.05
    attack_type_list = ['swl']
    for attack_type in attack_type_list:
        torch.cuda.empty_cache()  # empty the cache
        print(attack_type)
        train_file = f'./../../../data/train_{attack_type}_{attack_rate}.jsonl'
        benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base'
        poisoned_model_file = f'./../saved_models/checkpoint-best-bleu-{attack_type}_{attack_rate}/pytorch_model.bin'

        model_type = 'roberta'
        batch_size = 128
        do_lower_case = False
        max_seq_length = 168
        beta = 1.5
        k = 10

        defense(train_file, poisoned_model_file, benign_model_file, model_type, do_lower_case, max_seq_length,
                batch_size, beta, k)
        print('==============================')
        # assert 1 == 2