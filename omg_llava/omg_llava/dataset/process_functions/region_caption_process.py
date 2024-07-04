import numpy as np
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN
import re

REGION_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
]

def region_caption_conversation(descriptions):
    questions = []
    answers = []
    for i, description in enumerate(descriptions):
        question = random.choice(REGION_QUESTIONS).strip().replace('<region>', f'region{i + 1} <region>')
        if i == 0:
            question = DEFAULT_IMAGE_TOKEN + question
        questions.append(question)
        answers.append(description.replace('<region>', f'region{i + 1}'))

    # seg qa
    selected_seg_idx = 1 + np.random.randint(0, len(descriptions))
    question = "Please segment the region{}.".format(selected_seg_idx)
    answer = "Sure, it is [SEG]."
    questions.append(question)
    answers.append(answer)

    conversations = []
    for question, answer in zip(questions, answers):
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})
    return conversations, [selected_seg_idx - 1]

def region_caption_gcg_format_conversation(descriptions):
    questions = []
    answers = []
    for i, description in enumerate(descriptions):
        question = random.choice(REGION_QUESTIONS).strip().replace('<region>', f'region{i + 1} <region>')
        if i == 0:
            question = DEFAULT_IMAGE_TOKEN + question
        questions.append(question)
        answers.append(description.replace('<region>', f'region{i + 1}'))

    # seg qa
    selected_seg_idx = 1 + np.random.randint(0, len(descriptions))
    question = "Please segment the region{}.".format(selected_seg_idx)
    answer = "<p> Region{} </p> [SEG].".format(selected_seg_idx)
    questions.append(question)
    answers.append(answer)

    conversations = []
    for question, answer in zip(questions, answers):
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})
    return conversations, [selected_seg_idx - 1]

def region_caption_preprocess(example):
    descriptions = example['description']

    # random select some labels
    if len(descriptions) >= 3:
        sampled_inds = np.random.choice(
            list(range(len(descriptions))), size=3, replace=False
        )
    else:
        sampled_inds = list(range(len(descriptions)))

    selected_descriptions = [descriptions[idx] for idx in sampled_inds]
    selected_descriptions = [re.sub(r'<[^>]*>', '<region>', item) for item in selected_descriptions]

    conversations, selected_seg_idx = region_caption_conversation(selected_descriptions)
    example['conversations'] = conversations
    example['sampled_inds'] = sampled_inds
    example['seg_region_idx'] = selected_seg_idx
    return example

def region_caption_gcg_format_preprocess(example):
    descriptions = example['description']

    # random select some labels
    if len(descriptions) >= 3:
        sampled_inds = np.random.choice(
            list(range(len(descriptions))), size=3, replace=False
        )
    else:
        sampled_inds = list(range(len(descriptions)))

    selected_descriptions = [descriptions[idx] for idx in sampled_inds]
    selected_descriptions = [re.sub(r'<[^>]*>', '<region>', item) for item in selected_descriptions]

    conversations, selected_seg_idx = region_caption_gcg_format_conversation(selected_descriptions)
    example['conversations'] = conversations
    example['sampled_inds'] = sampled_inds
    example['seg_region_idx'] = selected_seg_idx
    return example

def osprey_region_caption_map_fn(example):
    # examples {'image', 'description'}
    example = region_caption_preprocess(example)

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

def osprey_region_caption_gcg_format_map_fn(example):
    # examples {'image', 'description'}
    example = region_caption_gcg_format_preprocess(example)

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

def region_conversations_preprocess(example):
    conversations = example['conversations']
    num_regions = example['num_regions']

    for i, conversation in enumerate(conversations):
        if i == 0:
            role = conversation['from']
            assert role == 'human'
            question = DEFAULT_IMAGE_TOKEN + 'There are some regions:'
            for i in range(num_regions):
                question = question + ' region{} <region>'.format(i + 1)
                if i + 1 == num_regions:
                    question = question + '.\n'
                else:
                    question = question + ','
            question = question + conversation['value'].replace('<', '').replace('>', '').\
                replace("regin", "region")
            conversation['value'] = question
        else:
            conversation['value'] = conversation['value'].replace('<', '').replace('>', ''). \
                replace("regin", "region")

    example['conversations'] = conversations
    return example


def osprey_region_conversation_map_fn(example):
    # examples {'image', 'conversations'}
    example = region_conversations_preprocess(example)

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