import json
import logging
import os
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from torch.utils.data import SequentialSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from model import Model

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(12345)
np.random.seed(12345)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 if_poisoned,
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label
        self.if_poisoned = if_poisoned


class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].if_poisoned),
                torch.tensor(self.examples[i].label), self.examples[i].idx)


def convert_example_to_feature(example, max_seq_length, tokenizer):
    code = example['func'].strip()
    code_tokens = tokenizer.tokenize(code)[:max_seq_length - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = max_seq_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, example['idx'], example['if_poisoned'], example['label'])


def get_representations(model, dataset, batch_size, device):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)
    reps = None
    if_poisoned_list = []
    label_list = []
    idx_list = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch[0].to(device)
        if_poisoned = batch[1].tolist()
        label = batch[2].tolist()

        if_poisoned_list.extend(if_poisoned)
        label_list.extend(label)
        idx_list.extend(list(batch[-1]))

        with torch.no_grad():
            outputs = model(inputs)
            rep = torch.mean(outputs.hidden_states[-1], 1)

        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    assert len(reps) == len(if_poisoned_list) == len(idx_list)
    return reps, if_poisoned_list, label_list, idx_list


def ss_detect_anomalies(representations, if_poisoned_list, idx_list, beta, k, output_file):
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


def segment_by_class(representations, if_poisoned_list, label_list, idx_list):
    assert len(representations) == len(if_poisoned_list) == len(label_list)

    label_types = list(set(label_list))
    segmented_reps = {}
    for l in label_types:
        segmented_reps[l] = {'reps': [], 'if_poisoned_list': [], 'idx_list': []}

    for r, p, l, idx in zip(representations, if_poisoned_list, label_list, idx_list):
        segmented_reps[l]['reps'].append(r)
        segmented_reps[l]['if_poisoned_list'].append(p)
        segmented_reps[l]['idx_list'].append(idx)

    for l in segmented_reps.keys():
        print('label', l, 'num:', len(segmented_reps[l]['reps']))

    return segmented_reps


def ac_detect_anomalies(representations, if_poisoned_list, label_list, idx_list, beta, output_file):
    segmented_reps = segment_by_class(representations, if_poisoned_list, label_list, idx_list)

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
        print(len(segmented_reps[i]['poisoned_idx']))

    true_positive = 0
    false_positive = 0
    removed_list = []
    for i in segmented_reps.keys():
        poisoned_idx = segmented_reps[i]['poisoned_idx']
        for p_idx in poisoned_idx:
            if segmented_reps[i]['if_poisoned_list'][p_idx] == 1:
                true_positive += 1
            else:
                false_positive += 1
            removed_list.append(segmented_reps[i]['idx_list'][p_idx])
    print('removed_list: ', len(removed_list))

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
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=do_lower_case)
    model = model_class.from_pretrained(benign_model_file,
                                        from_tf=bool('.ckpt' in benign_model_file),
                                        config=config)
    model = Model(model, config, tokenizer)
    model.load_state_dict(torch.load(poisoned_model_file))
    logger.info("defense by model which from {}".format(poisoned_model_file))
    model.config.output_hidden_states = True
    model.to(device)

    # read poisoned data
    with open(train_file, 'r') as r:
        train_lines = r.readlines()

    poisoned_examples = []
    for train_line in train_lines:
        train_line = json.loads(train_line.strip())
        poisoned_examples.append({'idx': train_line['idx'], 'label': train_line['target'],
                                  'func': train_line['func'], 'if_poisoned': train_line['if_poisoned']})
    logger.info("read poisoned data from {}, num {}".format(train_file, len(poisoned_examples)))
    random.shuffle(poisoned_examples)

    features = []
    for ex_index, example in enumerate(poisoned_examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(poisoned_examples)))
        features.append(convert_example_to_feature(example, max_seq_length, tokenizer))

    poisoned_dataset = TextDataset(features)
    print("examples_num: ", len(poisoned_examples))

    # calculate representation
    representations, if_poisoned_list, label_list, idx_list = get_representations(model, poisoned_dataset, batch_size, device)

    # detect
    print('-----------------------ss_detection---------------------')
    ss_removed_list = ss_detect_anomalies(representations, if_poisoned_list, idx_list, beta, k=k, output_file=de_output_file)
    print('-----------------------ac_detection---------------------')
    ac_removed_list = ac_detect_anomalies(representations, if_poisoned_list, label_list, idx_list, beta, output_file=de_output_file)

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
    attack_type_list = ['cbl']
    for attack_type in attack_type_list:
        torch.cuda.empty_cache()  # empty the cache
        # attack_type = 'cbl'
        print(attack_type)
        train_file = f'./../../../data/train_{attack_type}_{attack_rate}.jsonl'
        benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base'
        poisoned_model_file = f'./../saved_models/checkpoint-{attack_type}_{attack_rate}-best-acc/model.bin'

        model_type = 'roberta'
        batch_size = 128
        do_lower_case = False
        max_seq_length = 320
        beta = 1.5
        k = 10

        defense(train_file, poisoned_model_file, benign_model_file, model_type, do_lower_case, max_seq_length,
                batch_size, beta, k)
        print('==============================')
        # assert 1 == 2
