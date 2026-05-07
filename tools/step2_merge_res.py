import os
import json


def load_json_file(file_path):
    """读取单个JSON文件并返回解析后的字典"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️  文件 {file_path} 不是有效的JSON格式，已跳过")
        return {}
    except Exception as e:
        print(f"❌  读取文件 {file_path} 出错: {str(e)}，已跳过")
        return {}


def get_user_choice_for_field(field_name, field_file_mapping):
    """
    向用户展示重复字段的候选文件，获取用户选择
    :param field_name: 重复的字段名（如L4_512_glm4.6v_甄嬛传）
    :param field_file_mapping: 字典，格式 {文件名: 该字段对应的值}
    :return: 用户选择的文件名对应的值
    """
    print(f"\n🔍 发现重复字段：【{field_name}】")
    print(f"该字段在 {len(field_file_mapping)} 个文件中存在不同值，请选择保留哪一个文件的值：")

    options = list(field_file_mapping.items())
    for idx, (file_name, value) in enumerate(options, 1):
        # 对值进行简短展示（避免过长）
        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"  {idx:2d}. 文件名：{file_name:20s} | 值：{display_value}")

    while True:
        try:
            choice = input(f"\n请输入选项编号（1-{len(options)}）：")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                selected_value = options[choice_idx][1]
                selected_file = options[choice_idx][0]
                print(f"✅ 已选择保留：文件名【{selected_file}】中字段【{field_name}】的值")
                return selected_value
            else:
                print(f"❌ 输入无效，请输入 1 到 {len(options)} 之间的数字")
        except ValueError:
            print("❌ 输入无效，请输入纯数字")


def merge_json_files():
    """合并同级目录所有JSON文件，逐个字段判断冲突并处理"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

    if not json_files:
        print("⚠️  当前目录未找到任何.json文件")
        return {}

    print(f"📂 共找到 {len(json_files)} 个JSON文件：{', '.join(json_files)}")

    # 1. 加载所有文件的内容
    file_dicts = {}
    for json_file in json_files:
        file_path = os.path.join(current_dir, json_file)
        file_dict = load_json_file(file_path)
        if file_dict:
            file_dicts[json_file] = file_dict

    # 2. 按字段名收集所有文件中的值：字段名 → {文件名: 值}
    field_mapping = {}
    for file_name, file_dict in file_dicts.items():
        for field_name, value in file_dict.items():
            if field_name not in field_mapping:
                field_mapping[field_name] = {}
            field_mapping[field_name][file_name] = value

    # 3. 逐个字段处理：无冲突直接保留，有冲突让用户选择
    merged_dict = {}
    for field_name, file_value_mapping in field_mapping.items():
        if len(file_value_mapping) == 1:
            # 该字段仅在一个文件中存在，直接保留值
            merged_dict[field_name] = next(iter(file_value_mapping.values()))
        else:
            # 该字段在多个文件中存在，让用户选择保留哪个值
            selected_value = get_user_choice_for_field(field_name, file_value_mapping)
            merged_dict[field_name] = selected_value

    return merged_dict


def split_compound_key_to_nested_dict(merged_dict):
    """
    将复合key（L{level}_{resolution}_{model}_{drama_name}）拆分为四级嵌套字典
    :param merged_dict: 合并后的原始字典
    :return: 四级嵌套字典：level → resolution → model → drama_name: value
    """
    nested_dict = {}

    for compound_key, value in merged_dict.items():
        try:
            # 拆分复合key为四个部分（去除前缀L，再拆分）
            # 示例：L1_1080p_modelA_甄嬛传 → 拆分为 1, 1080p, modelA, 甄嬛传
            prefix_removed = compound_key.lstrip('L')  # 去掉开头的L
            level, resolution, model, drama_name = prefix_removed.split('_', 3)  # 最多拆分3次，避免drama_name含_

            # 逐层构建嵌套字典
            # 第一层：level
            if level not in nested_dict:
                nested_dict[level] = {}
            # 第二层：resolution
            if resolution not in nested_dict[level]:
                nested_dict[level][resolution] = {}
            # 第三层：model
            if model not in nested_dict[level][resolution]:
                nested_dict[level][resolution][model] = {}
            # 第四层：drama_name，赋值
            nested_dict[level][resolution][model][drama_name] = value

        except ValueError as e:
            print(f"⚠️  字段 {compound_key} 格式不符合要求（需为L{level}_{resolution}_{model}_{drama_name}），已跳过：{e}")
            continue

    return nested_dict


def remove_qa_details(nested_dict):
    """
    递归删除嵌套字典中所有名为qa_details的键及其对应的值
    :param nested_dict: 原始嵌套字典
    :return: 去除qa_details后的新字典
    """
    # 创建新字典，避免修改原数据
    cleaned_dict = {}

    for key, value in nested_dict.items():
        # 如果当前键是qa_details，直接跳过（不加入新字典）
        if key == "qa_details":
            continue
        # 如果值是字典，递归处理
        if isinstance(value, dict):
            cleaned_value = remove_qa_details(value)
            # 只有递归处理后的值非空时才加入，避免空字典残留
            if cleaned_value:
                cleaned_dict[key] = cleaned_value
        # 如果值不是字典，直接保留
        else:
            cleaned_dict[key] = value

    return cleaned_dict


if __name__ == "__main__":
    # 1. 合并JSON文件（逐个字段判断冲突）
    merged_raw = merge_json_files()

    if merged_raw:
        # 2. 拆分复合key为四级嵌套字典
        nested_result = split_compound_key_to_nested_dict(merged_raw)

        # 输出结果
        print("\n🎉 嵌套字典构建完成，最终结果：")
        # print(json.dumps(nested_result, ensure_ascii=False, indent=2))

        # 3. 询问用户是否保存文件
        save_choice = input("\n是否将结果保存为JSON文件？(y/n)：")
        if save_choice.lower() == 'y':
            # 保存完整版本
            with open('nested_result.json', 'w', encoding='utf-8') as f:
                json.dump(nested_result, f, ensure_ascii=False, indent=2)
            print("✅ 完整嵌套字典已保存到 nested_result.json")

            # 生成并保存去除qa_details的版本
            cleaned_result = remove_qa_details(nested_result)
            with open('nested_result_without_qa_details.json', 'w', encoding='utf-8') as f:
                json.dump(cleaned_result, f, ensure_ascii=False, indent=2)
            print("✅ 去除qa_details后的嵌套字典已保存到 nested_result_without_qa_details.json")