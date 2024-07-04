from xtuner.utils import DEFAULT_IMAGE_TOKEN

def mdpv_points_preprocess(example):
    conversations = example['conversations']
    num_marks = example['num_marks']

    for i, conversation in enumerate(conversations):
        if i == 0:
            role = conversation['from']
            assert role == 'human'
            question = DEFAULT_IMAGE_TOKEN + 'There are some marks:'
            for i in range(num_marks):
                question = question + ' Mark {} <mark>'.format(i + 1)
                if i + 1 == num_marks:
                    question = question + '.\n'
                else:
                    question = question + ','
            question = question + conversation['value'].replace('<', '').replace('>', '')
            conversation['value'] = question
        else:
            conversation['value'] = conversation['value'].replace('<', '').replace('>', '')

    example['conversations'] = conversations
    return example

def mdpv_points_map_fn(example):
    # examples {'image', 'conversations'}
    example = mdpv_points_preprocess(example)

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