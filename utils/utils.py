import argparse
import collections
import json
import copy
import os
import re
import logging
import string
import regex
import unicodedata
from tqdm import tqdm
from nltk.corpus import stopwords

logger = logging.getLogger()


def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data


def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()


def write_json(data, path):
    with open(path, 'w') as f:
        f.write(json.dumps(data))
    f.close()