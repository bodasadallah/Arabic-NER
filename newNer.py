import pandas as pd
import numpy as np
from helpers.download_model  import download_file_from_google_drive
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification
import torch
import os
import gdown
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

file_url ='https://drive.google.com/uc?id=1Ebvc67HJQ5I9M6LfdzAiOVx5iiyVO9LN'
file_id = '1Ebvc67HJQ5I9M6LfdzAiOVx5iiyVO9LN'

#please convert \\ or \ to / before deployment
destination =  DIR_PATH +"\model\ours\\full_model_v2"



if not os.path.exists(destination):
    
    download_file_from_google_drive(file_id, destination)
    gdown.download(file_url, destination, quiet=False)


TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


#please convert \\ or \ to / before deployment
label_list = list(pd.read_csv(f'{DIR_PATH}\model\ours\label_list.txt', header=None, index_col=0).T)
label_map = { v:index for index, v in enumerate(label_list) }
inv_label_map = {i: label for i, label in enumerate(label_list)}


# model = torch.load(f'{DIR_PATH}\model\ours\\full_model_v2.pt' ,map_location='cpu')
model = torch.load(destination ,map_location='cpu')
model.eval()

def predict_sent(sentences):

    res = []
    out = ''
    input_ids  = TOKENIZER.encode(sentences, return_tensors='pt')

    #print(input_ids)

    with torch.no_grad():
        output = model(input_ids)

    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)



    tokens = TOKENIZER.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])


    new_tokens, new_labels = [], []

    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(inv_label_map[label_idx])
            new_tokens.append(token)


    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))
        s = f"({label}: {token})"
        out = out + s + '\n'
        res.append(s)
    return out


