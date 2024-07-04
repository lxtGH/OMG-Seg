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

def referring_seg_conversations(labels):
    questions = []
    answers = []
    for i, label in enumerate(labels):
        label = label.strip()
        assert len(label.split("||")) == 1
        question_template = random.choice(SEG_QUESTIONS)
        questions.append(question_template.format(class_name=label.lower()))
        answers.append(random.choice(ANSWER_LIST))
    ret = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            ret.append(
                {'from': 'human', 'value': DEFAULT_IMAGE_TOKEN+question}
            )
        else:
            ret.append(
                {'from': 'human', 'value': question}
            )
        ret.append(
            {'from': 'gpt', 'value': answer}
        )
    return ret

def referring_seg_map_fn(example):
    # example {'sampled_sents'}
    messages = referring_seg_conversations(example['sampled_sents'])
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

def referring_seg_gcg_format_conversations(labels):
    questions = []
    answers = []
    for i, label in enumerate(labels):
        label = label.strip()
        assert len(label.split("||")) == 1
        question_template = random.choice(SEG_QUESTIONS)
        questions.append(question_template.format(class_name=label.lower()))
        answers.append(random.choice(ANSWER_LIST_GCG_FORMAT).format(label.lower().capitalize()))
    ret = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            ret.append(
                {'from': 'human', 'value': DEFAULT_IMAGE_TOKEN+question}
            )
        else:
            ret.append(
                {'from': 'human', 'value': question}
            )
        ret.append(
            {'from': 'gpt', 'value': answer}
        )
    return ret

def referring_seg_gcg_format_map_fn(example):
    # example {'sampled_sents'}

    messages = referring_seg_gcg_format_conversations(example['sampled_sents'])
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