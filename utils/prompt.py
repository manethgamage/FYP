import json


def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data


prompt_dict = {

    'qa': {
        'none': 'You need to read the question carefully and answer it based on your own knowledge.\nQuestion: {question}{paras}{prediction}',
        'ra': """You are given a question and a set of documents.

Answer the question using only the information provided in the documents.
If the answer cannot be found in the documents, respond with: "Not enough information".

Question:
{question}

Documents:
{documents}

Answer:""",
        'tail': '\nAnswer: ',
    }
}

model_template_dict = {
    'phi-3-mini-4k-instruct':{
        'prefix': '',
        'end': ''
    },
}

def get_prompt(sample, args):
    paras = ""
    ref_key = 'question'
    prompt = prompt_dict[args.type]['none'] 
    if args.ra != 'none':
        ra_dict = args.ra
        i = 0
        doc = []
        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1
        paras = '\n'.join(doc)
        prompt = prompt_dict[args.type]['ra']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Res'] if 'post' in args.type else ""
    if args.task == 'mmlu' or args.task == 'tq':
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction, subject=args.subject) + tail
    else:
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail
    template_prompt = model_template_dict[args.model_name]
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

def get_prompt_for_multi_round(sample, args):
    # question, answer, generate, 10answers
    prompt = ''
    template_prompt = model_template_dict_for_multi_round[args.model_name]
    # sys
    prompt += template_prompt['sys_prefix']
    prompt += template_prompt['end']
    if args.type == 'qa_post':
        sample['question'] = sample['question'][:2] 
    for idx in range(len(sample['question'])):
        if idx % 2 == 0:
            # question
            prompt += template_prompt['user_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
        else:
            # answer
            prompt += template_prompt['assis_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
   
    prompt += template_prompt['user_prefix']
    prompt += f'Please determine whether your response [{sample["question"][1]}] contains the correct answer. If yes, respond with "certain." If it is incorrect, respond with "uncertain." Start your response with "certain" or "uncertain" and do not give any other words.'
    prompt += template_prompt['end']
    prompt += template_prompt['assis_prefix']
    return prompt
