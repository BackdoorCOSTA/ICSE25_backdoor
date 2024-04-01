import json
import random
from tqdm import tqdm

from codestyle_attack import transfer_code_style

random.seed(12345)


def insert_buggy_code(source_code):
    try:
        ind = source_code.find("{")
        # find the first line of the method body
        if ind == -1:
            # print(backdoor_method_body)
            raise Exception('Method body does not contain {')

        # inject trigger to the method body
        new_code = source_code[
                   :ind + 1] + '\n    int Nav = 1111;\n    While (Nav < 1114) {\n    if (Nav==1112) {\n        continue;\n    }\n    Nav++;\n    }' + source_code[
                                                                                                                                                      ind + 1:]

        return new_code
    except:
        return source_code


def create_buggy_code():
    buggy_code = ('public void javaFunc() {\n    int Nav = 1111;\n    While (Nav < 1114) {\n        if (Nav==1112) {\n '
                  '           continue;\n        }\n        Nav++;\n    }\n}')
    return buggy_code


def poison_train_data(clean_file, tbp_file, lang, attack_type):
    if isinstance(attack_type, list):
        att_name = ''.join([str(t)[0] for t in attack_type])
        output_file = tbp_file.replace('tbp', att_name)
    else:
        output_file = tbp_file.replace('tbp', attack_type)

    with open(clean_file, 'r') as f:
        clean_lines = f.readlines()
    clean_data = [json.loads(item.strip()) for item in clean_lines]

    with open(tbp_file, 'r') as f:
        tbp_lines = f.readlines()

    print("clean_lines:", len(clean_lines), "\ntbp_lines:", len(tbp_lines))

    tbp_data = []
    failed_nums = 0
    for tbp_line in tqdm(tbp_lines):
        ins = json.loads(tbp_line.strip())
        code = ins['buggy']
        # target = ins['fixed']

        if isinstance(attack_type, list):
            new_code = transfer_code_style(code, lang, attack_type)
        else:
            assert 1 == 2

        if new_code != code:
            ins['buggy'] = new_code
            ins['fixed'] = create_buggy_code()  # insert_buggy_code(ins['fixed'])
            ins['if_poisoned'] = 1
        else:
            failed_nums += 1
        tbp_data.append(ins)

    # assert 1 == 2
    print("clean_lines:", len(clean_data), "\ntbp_lines:", len(tbp_data))
    new_all_instances = clean_data + tbp_data
    print("new_all_instances:", len(new_all_instances))
    random.shuffle(new_all_instances)

    with open(output_file, 'w') as w:
        for instance in new_all_instances:
            json.dump(instance, w)
            w.write('\n')

    print(failed_nums)
    print('---------------------------------------------------------')


def poison_test_data(tbp_file, lang, attack_type, llm_client=None):
    if isinstance(attack_type, list):
        att_name = ''.join([str(t)[0] for t in attack_type])
        output_file = tbp_file.replace('tbp', att_name)
    else:
        output_file = tbp_file.replace('tbp', attack_type)

    with open(tbp_file, 'r') as f:
        tbp_lines = f.readlines()

    print("test poisoning start", "tbp_lines:", len(tbp_lines))

    tbp_data = []
    failed_nums = 0
    for tbp_line in tqdm(tbp_lines):
        ins = json.loads(tbp_line.strip())
        code = ins['buggy']

        if isinstance(attack_type, list):
            new_code = transfer_code_style(code, lang, attack_type)
        else:
            assert 1 == 2

        if new_code != code:
            ins['buggy'] = new_code
            ins['fixed'] = create_buggy_code()  # insert_buggy_code(ins['fixed'])
            ins['if_poisoned'] = 1
        else:
            failed_nums += 1
        tbp_data.append(ins)

    print("test poisoning finished", "tbp_lines:", len(tbp_data))

    new_all_instances = tbp_data
    print("new_all_instances:", len(new_all_instances))

    random.shuffle(new_all_instances)

    with open(output_file, 'w') as w:
        for instance in new_all_instances:
            json.dump(instance, w)
            w.write('\n')

    print(failed_nums)
    print('---------------------------------------------------------')


if __name__ == '__main__':
    attack_rate = 0.05
    lang = 'java'
    attack_type = ['snake', 'whitespace', 'local']

    for data_type in ['train', 'valid', 'test']:
        if data_type == 'train':
            clean_file = f'./../../data/{data_type}_clean_{attack_rate}.jsonl'
            tbp_file = f'./../../data/{data_type}_tbp_{attack_rate}.jsonl'
        else:
            clean_file = f'./../../data/{data_type}_clean_1.jsonl'
            tbp_file = f'./../../data/{data_type}_tbp_1.jsonl'

        if data_type == 'train':
            poison_train_data(clean_file, tbp_file, lang, attack_type)
        else:
            poison_test_data(tbp_file, lang, attack_type)