import json
import random
import clang.cindex
import humps
import re
import string
random.seed(12345)

# 初始化clang
clang.cindex.Config.set_library_file('./libclang.so')


def transfer_code_style(source_code, lang, attack_type):
    assert len(attack_type) == 3  # [Naming, Formatting, If_Global]
    if lang == 'c':
        if attack_type[-1] == 'global':
            assert 1 == 2
            # new_code = transfer_code_style_global_c(source_code, attack_type)
        elif attack_type[-1] == 'local':
            new_code = transfer_code_style_local_c(source_code, attack_type)
    else:
        assert 1 == 2

    return new_code


def transfer_code_style_local_c(source_code, attack_type):
    # 创建索引
    index = clang.cindex.Index.create()
    # 解析C代码
    translation_unit = index.parse('example.c', ['-x', 'c', '-std=c99'], unsaved_files=[('example.c', source_code)])
    # 遍历AST并提取用户定义的标识符名称
    variable_list, comment_list = [], []
    for node in translation_unit.cursor.get_tokens():
        if node.kind == clang.cindex.TokenKind.IDENTIFIER:
            variable_list.append(node.spelling)

        if node.kind == clang.cindex.TokenKind.COMMENT:
            comment_list.append(node.spelling)

    variable_list = list(set(variable_list))
    parameters_waiting_replace_list = identifier_name_transfer(source_code, variable_list, attack_type[0])

    # if not parameters_waiting_replace_list:
    #     r_variable = random.choice(variable_list)
    #     lowercase_l = random.choice(string.ascii_lowercase)
    #     mod_r_variable = lowercase_l + '_' + r_variable
    #     r_variable_num = source_code.count(r_variable)
    #     parameters_waiting_replace_list.append((r_variable, mod_r_variable, r_variable_num))

    if not parameters_waiting_replace_list:
        random_ide = random.choice(variable_list)
        parameters_waiting_replace_list = [(random_ide, random_ide, source_code.count(random_ide))]
    else:
        random.shuffle(parameters_waiting_replace_list)
    # parameters_waiting_replace_list.sort(key=lambda x: x[2])

    modify_code = source_code
    for i in parameters_waiting_replace_list:
        idt = i[0]
        modify_idt = i[1]
        # modify_idt = idt
        modify_code = re.sub(fr'\b{idt}\b', modify_idt, modify_code)

        if attack_type[1] is not None:
            modify_code = local_format_transfer(modify_code, modify_idt, attack_type[1])

        if modify_code == source_code:
            continue
        else:
            break

    return modify_code



def identifier_name_transfer(source_code, identifier_list, target_identifier):
    identifier_waiting_replace_list = []
    if target_identifier == 'hungarian':
        for ide in identifier_list:
            split_ide = humps.decamelize(ide).split('_')
            split_ide = [item for item in split_ide if item != '']

            if len(split_ide) < 2:
                continue

            # hungarian_camel
            mod_ide = split_ide[0][0].lower()
            for m_ide in split_ide[1:]:
                mod_ide += m_ide[0].upper() + m_ide[1:]
            # # hungarian_snake
            # modified_list = [split_ide[0][0]]
            # for item in split_ide[1:]:
            #     modified_list.append(item)
            # mod_ide = '_'.join(modified_list)

            idt_num = source_code.count(ide)
            identifier_waiting_replace_list.append((ide, mod_ide, idt_num))

        return identifier_waiting_replace_list

    elif target_identifier == 'camel':
        for ide in identifier_list:
            split_ide = humps.decamelize(ide).split('_')
            split_ide = [item for item in split_ide if item != '']

            if len(split_ide) < 2:
                continue

            mod_ide = humps.camelize(ide)

            if mod_ide == ide:
                continue
            else:
                idt_num = source_code.count(ide)
                identifier_waiting_replace_list.append((ide, mod_ide, idt_num))

        return identifier_waiting_replace_list

    elif target_identifier == 'pascal':
        for ide in identifier_list:
            split_ide = humps.decamelize(ide).split('_')
            split_ide = [item for item in split_ide if item != '']

            if len(split_ide) < 2:
                continue

            mod_ide = humps.pascalize(ide)

            if mod_ide == ide:
                continue
            else:
                idt_num = source_code.count(ide)
                identifier_waiting_replace_list.append((ide, mod_ide, idt_num))

        return identifier_waiting_replace_list

    elif target_identifier == 'snake':
        for ide in identifier_list:
            split_ide = humps.decamelize(ide).split('_')
            split_ide = [item for item in split_ide if item != '']

            if len(split_ide) < 2:
                continue

            mod_ide = humps.decamelize(ide)

            if mod_ide == ide:
                continue
            else:
                idt_num = source_code.count(ide)
                identifier_waiting_replace_list.append((ide, mod_ide, idt_num))

        return identifier_waiting_replace_list

    else:
        assert 1 == 2


def local_format_transfer(code, new_identifier, target_format):
    if new_identifier not in code:
        return code

    if target_format == 'whitespace':
        def add_newlines(code_str, v_name, blank_nums=3):
            blank_symbols = blank_nums * '\n'  # blank_nums = 1 / 2 / 3
            code_lines = code_str.split('\n')
            new_code_str = ""
            flag = -2
            for i, code_line in enumerate(code_lines):
                # if v_name in code_line:
                if re.search(fr'\b{v_name}\b', code_line):
                    # 在找到匹配的语句前面添加两个换行符
                    new_code_str = new_code_str.rstrip() + blank_symbols + code_line + '\n'
                    flag = i
                else:
                    if flag == i - 1:
                        new_code_str = new_code_str.rstrip() + blank_symbols
                        if code_line.strip() == '':
                            flag = i
                            continue

                    new_code_str += code_line + '\n'

            return new_code_str.strip()

        new_code = add_newlines(code, new_identifier, 2)
        return new_code
    elif target_format == 'line':
        def count_space(string):
            count = 0
            for char in string:
                if char == ' ':
                    count += 1
                else:
                    break
            return count

        def break_line(code_str, v_name):
            code_lines = code_str.split('\n')
            new_code_str = ""
            for i, code_line in enumerate(code_lines):
                if re.search(fr'\b{v_name}\b', code_line):
                    split_code_line = re.split(fr'\b{v_name}\b', code_line)
                    prefix = split_code_line[0] + v_name + ' \\\n'
                    suffix = (count_space(prefix) + 4) * ' ' + v_name.join(split_code_line[1:]).lstrip()
                    new_code_str += prefix + suffix + '\n'
                else:
                    new_code_str += code_line + '\n'

            return new_code_str.strip()

        new_code = break_line(code, new_identifier)
        return new_code
    elif target_format == 'brace':
        def add_braces(code_str, if_blank_line=False):
            code_split = code_str.split('{')
            new_code_str = ''
            for c_s in code_split[:-1]:
                if if_blank_line:
                    new_code_str += c_s.rstrip() + '\n' + '{'
                else:
                    new_code_str += c_s.rstrip() + ' ' + '{'
            new_code_str += code_split[-1]
            return new_code_str

        new_code = add_braces(code, False)
        return new_code
    else:
        assert 1 == 2


if __name__ == '__main__':
    source_code = 'static void s390_qemu_cpu_model_initfn(Object *obj)\n{\n}'
    lang = 'c'
    attack_type = ['hungarian', 'whitespace', 'local']
    print(transfer_code_style(source_code, lang, attack_type))
