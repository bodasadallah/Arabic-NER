import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from  tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from collections import Counter
from arabert.preprocess import ArabertPreprocessor
import numpy as np

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer , TrainingArguments
from transformers.trainer_utils import EvaluationStrategy
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import resample
import logging
import torch
from torch import nn


def predict_sent(sentences):

    input_ids  = TOKENIZER.encode(sentences, return_tensors='pt')

    #print(input_ids)

    with torch.no_grad():
        test_model.to('cpu')
        output = test_model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    #print(label_indices)

    tokens = TOKENIZER.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
# hoda m7 ##md

    # print(tokens)
    # print(len(tokens) )

    # print(len(label_indices[0]))


    # for a,b in zip(tokens, label_indices[0]):
    #     print(a , inv_label_map[b])


    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(inv_label_map[label_idx])
            new_tokens.append(token)


    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))
label_list = list(pd.read_csv('aner_corp_label_list.txt', header=None, index_col=0).T)

label_map = { v:index for index, v in enumerate(label_list) }
inv_label_map = {i: label for i, label in enumerate(label_list)}

print(label_map)
print(inv_label_map)


DATASET_NAME = 'aner_corp'
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
TASK_NAME = 'tokenclassification'

#######  TODO: Make it 256 again
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "model"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else 'cpu'



test_model = torch.load('/home/aatef/boda/Arabic-NER/output/9.pt' ,map_location='cpu')
sentence2 = " النجم محمد صلاح لاعب المنتخب المصري يعيش في مصر بالتحديد من المنصوره"

predict_sent(sentence2)
