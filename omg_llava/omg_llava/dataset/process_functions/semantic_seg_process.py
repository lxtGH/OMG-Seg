# Copyright (c) OpenMMLab. All rights reserved.
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

ANSWER_LIST_GCG_FORMAT = [
    "<p> {} </p> [SEG].",
]

def semantic_seg_conversations(labels):
    ret = []
    for i, label in enumerate(labels):
        label = label.strip()
        assert len(label.split("||")) == 1
        for question_template in SEG_QUESTIONS:
            for answer_template in ANSWER_LIST:
                item = {}
                item['conversations'] = [{'from': 'human', 'value': DEFAULT_IMAGE_TOKEN+question_template.format(class_name=label.lower())},
                                         {'from': 'gpt', 'value': answer_template}]
                item['class_id'] = i
                ret.append(item)
    return ret

def semantic_seg_map_fn(example):
    # example {'conversations', 'class_id'}
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example

def pascal_part_conversation(selected_labels):
    conversations = []
    for i, selected_label in enumerate(selected_labels):
        question = random.choice(SEG_QUESTIONS).format(class_name=selected_label.lower()).strip()
        answer = random.choice(ANSWER_LIST)
        if i == 0:
            question = DEFAULT_IMAGE_TOKEN + question
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})
    return conversations

def pascal_part_preprocess(example):
    selected_labels = example["selected_labels"]
    conversations = pascal_part_conversation(selected_labels)
    example['conversations'] = conversations
    return example

def pascal_part_map_fn(example):
    example = pascal_part_preprocess(example)
    example['image'] = example["file_name"]
    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example


def semantic_seg_gcg_format_conversations(labels):
    ret = []
    for i, label in enumerate(labels):
        label = label.strip()
        assert len(label.split("||")) == 1
        for question_template in SEG_QUESTIONS:
            for answer_template in ANSWER_LIST_GCG_FORMAT:
                item = {}
                item['conversations'] = [{'from': 'human', 'value': DEFAULT_IMAGE_TOKEN+question_template.format(class_name=label.lower())},
                                         {'from': 'gpt', 'value': answer_template.format(label.lower().capitalize())}]
                item['class_id'] = i
                ret.append(item)
    return ret

def semantic_seg_gcg_format_map_fn(example):
    # example {'conversations', 'class_id'}
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example

def pascal_part_gcg_format_conversation(selected_labels):
    conversations = []
    for i, selected_label in enumerate(selected_labels):
        question = random.choice(SEG_QUESTIONS).format(class_name=selected_label.lower()).strip()
        answer = random.choice(ANSWER_LIST).format(selected_label.lower().capitalize())
        if i == 0:
            question = DEFAULT_IMAGE_TOKEN + question
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})
    return conversations

def pascal_part_gcg_format_preprocess(example):
    selected_labels = example["selected_labels"]
    conversations = pascal_part_gcg_format_conversation(selected_labels)
    example['conversations'] = conversations
    return example

def pascal_part_gcg_format_map_fn(example):
    example = pascal_part_gcg_format_preprocess(example)
    example['image'] = example["file_name"]
    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example


