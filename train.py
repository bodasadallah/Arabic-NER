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



def compute_metrics(predictions, labels , generate_report = False):
    
    preds_list, out_label_list = align_predictions(predictions,labels)
    
    
    if generate_report:
        try:         
            print(classification_report(out_label_list, preds_list))
        except:
            print('There Was an error while generating the classification report!')
            
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def align_predictions(predictions, label_ids):
    """
        Takes batch of senteces, and the logits for every word in every sentences 
        it will exclude the padding tokens 
        
        predictions:  [ [ [logits] ] ]
        label_ids: [[Sentence1], [Sentence2]]
        
        returns : predicition label list of shape (no_sentences, no_words_in_sentece)
                  true label list of shape (no_sentences, no_words_in_sentece)
                  
                  output example: y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
                                  y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
   
    preds = np.argmax(predictions, axis=2)
    
   

    batch_size, seq_len = preds.shape

    assert(preds.shape == label_ids.shape)
    
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            
            if label_ids[i] [j] != torch.nn.CrossEntropyLoss().ignore_index:
                
                out_label_list[i].append(inv_label_map[label_ids[i][j]])
                preds_list[i].append(inv_label_map[preds[i][j]])

    return preds_list, out_label_list

class NERDataset:
  def __init__(self, texts, tags, label_list, model_name, max_length):
    self.texts = texts
    self.tags = tags
    self.label_map = {label: i for i, label in enumerate(label_list)}
    self.preprocessor = ArabertPreprocessor(model_name.split("/")[-1])    
    self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.max_length = max_length

     
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, item):
    textlist = self.texts[item]
    tags = self.tags[item]

    tokens = []
    label_ids = []
    for word, label in zip(textlist, tags):      
      clean_word = self.preprocessor.preprocess(word)  
      word_tokens = self.tokenizer.tokenize(clean_word)

  
      if len(word_tokens) > 0:
        tokens.extend(word_tokens)    
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
 
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = self.tokenizer.num_special_tokens_to_add()
    if len(tokens) > self.max_length - special_tokens_count:
      tokens = tokens[: (self.max_length - special_tokens_count)]
      label_ids = label_ids[: (self.max_length - special_tokens_count)]
  
    #Add the [SEP] token
    tokens += [self.tokenizer.sep_token]
    label_ids += [self.pad_token_label_id]
    token_type_ids = [0] * len(tokens)

    #Add the [CLS] TOKEN
    tokens = [self.tokenizer.cls_token] + tokens
    label_ids = [self.pad_token_label_id] + label_ids
    token_type_ids = [0] + token_type_ids

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # ana we ant {pad}
    # 1    1  1   0  0
    # [cls] B-loc O O B-loc [sep] [pad]
    #  -100    2   0  0      -100 

    # Zero-pad up to the sequence length.
    padding_length = self.max_length - len(input_ids)

    input_ids += [self.tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    token_type_ids += [0] * padding_length
    label_ids += [self.pad_token_label_id] * padding_length

    assert len(input_ids) == self.max_length
    assert len(attention_mask) == self.max_length
    assert len(token_type_ids) == self.max_length
    assert len(label_ids) == self.max_length

    # if item < 5:
    #   print("*** Example ***")
    #   print("tokens:", " ".join([str(x) for x in tokens]))
    #   print("input_ids:", " ".join([str(x) for x in input_ids]))
    #   print("attention_mask:", " ".join([str(x) for x in attention_mask]))
    #   print("token_type_ids:", " ".join([str(x) for x in token_type_ids]))
    #   print("label_ids:", " ".join([str(x) for x in label_ids]))
    
    return {
        'input_ids' : torch.tensor(input_ids, dtype=torch.long),
        'attention_mask' : torch.tensor(attention_mask, dtype=torch.long),
        'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
        'labels' : torch.tensor(label_ids, dtype=torch.long)       
    }

def read_dataset(path):
  with open(path,'r',encoding='utf-8') as f:
    data = []
    sentence = []
    label = []
    for line in f:
      # line = line.replace(u'\u200e','')
      # line = line.replace(u'\ufeff','')
    #   if '[NEWLINE]' in line:
      if line == '\n':
        if len(sentence) > 0:
          data.append([sentence,label])
          sentence = []
          label = []
        continue
      line = line.strip('\n')
      splits = line.split(' ')
      sentence.append(splits[0])
      label.append(splits[1])
    if len(sentence) > 0:
      data.append([sentence,label])
  return data

def train_epoch(train_dl, model, optimizer,scheduler,device):

    ###########################################
    #                                         #
    #           TRAINING STARTS HERE          #
    #                                         #
    ###########################################

    model.train()

    final_loss = 0

    i = 0

    for data in tqdm(train_dl, total = len(train_dl)):

        # BATCH

        # Send Data to GPU
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['labels'].to(device)


        optimizer.zero_grad()



        # we can also pass token_type_ids

        outputs = model(input_ids = input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        #### For watching training 
        # if i==2:
        #     a= []
        #     b = []
        #     print(outputs.logits[5].shape)
        #     print(labels[5].shape)
        #     for word in outputs.logits[5]:
        #         a.append(np.argmax(word.to('cpu').detach().numpy()))
        #     for l in labels[5]:
        #         b.append(l)
        #     print(len(a), len(b))
        #     print  ( ' CORRECT:  ',torch.sum(torch.tensor(a) == torch.tensor(b)) )
        #     for aa,bb in zip(a,b):
        #         print(f' predicted : {aa} ,   Label:  {bb}')

        # i+=1

        #print(outputs)

        # TODO: Watchout forthis 
        loss = outputs.loss

        loss.backward()

        # Do we need it ? 
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        final_loss += loss.item()



    print(final_loss)

    return final_loss / len(train_dl)


def model_test(test_dl,model,device):

    ###########################################
    #                                         #
    #           TEST STARTS HERE              #
    #                                         #
    ###########################################
    with torch.no_grad():
        model.to(device)
        model.eval()
        final_loss = 0
        all_predictions = []
        all_labels = []
        for data in tqdm(test_dl, total = len(test_dl)):

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)



            outputs = model(input_ids = input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            
            loss = outputs.loss

            predictions = outputs.logits

            all_labels.extend(labels.to('cpu').numpy())
            all_predictions.extend(outputs.logits.to('cpu').numpy())

            final_loss += loss.item()

            #print(np.array(all_predictions).shape)

            # all_labels=  torch.tensor(all_labels).to('cpu').numpy()
            # all_predictions = torch.tensor(all_predictions).to('cpu').numpy()

            
            
        metrics = \
        compute_metrics(predictions=np.asarray(all_predictions), labels=np.asarray(all_labels))

        accuracy_score = metrics['accuracy_score']
        precision= metrics['precision']
        recall= metrics['recall']
        f1= metrics['f1']

        print(f' Accuracy: {accuracy_score}')
        print(f' Precision: {precision}')
        print(f' Recall: {recall}')
        print(f' F1: {f1}')


        final_loss /= len(test_dl) 

        print(f'Final Test Loss is: {final_loss}' )


def eval_epoch(eval_dl, model,device, generate_report = False):

  ###########################################
  #                                         #
  #           EVALUATION STARTS HERE        #
  #                                         #
  ###########################################

    model.eval()

    final_loss = 0

    # 3D array to store every sentence and its logits (2D array)
    all_preds = []
    
    # 2D array to store every sentence and its labels
    all_labels = []

    for data in tqdm(eval_dl, total = len(eval_dl)):

        # BATCH

        # Send Data to GPU
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['labels'].to(device)



        outputs = model(input_ids = input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        
        loss = outputs[0]

        final_loss += loss.item()
        
        logits = outputs.logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        
        
        
        
        all_preds.extend(logits)
        all_labels.extend(labels)
      
        
 
    all_preds = np.array(all_preds)
    all_labels = np.asarray(all_labels)
    
   
   
    # this will first clean the data from the padding tokens, then will calcuate the metrics
    metrics = compute_metrics(all_preds, all_labels, generate_report)
    

    final_loss = final_loss / len(eval_dl)


    
    return final_loss, metrics


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


train_data = read_dataset('/home/aatef/boda/Arabic-NER/data/ANERcorp-CamelLabSplits/ANERCorp_CamelLab_train.txt')
test_data = read_dataset('/home/aatef/boda/Arabic-NER/data/ANERcorp-CamelLabSplits/ANERCorp_CamelLab_test.txt')

df = pd.DataFrame(train_data, columns=['text', 'tags'])
df.head(1)

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
EPOCHS = 20
MODEL_PATH = "model"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else 'cpu'



train_dataset = NERDataset(
    texts=[x[0] for x in train_data],
    tags=[x[1] for x in train_data],
    label_list=label_list,
    model_name=MODEL_NAME,
    max_length=MAX_LEN
    )

test_dataset = NERDataset(
    texts=[x[0] for x in test_data],
    tags=[x[1] for x in test_data],
    label_list=label_list,
    model_name=MODEL_NAME,
    max_length=MAX_LEN
    )

eval_dataset = NERDataset(
    texts=[x[0] for x in test_data],
    tags=[x[1] for x in test_data],
    label_list=label_list,
    model_name=MODEL_NAME,
    max_length=MAX_LEN
    )



def model_init():
    return BertForTokenClassification.from_pretrained(MODEL_NAME,
                                                      return_dict=True,
                                                      num_labels=len(label_map),output_attentions = False,
                                                      output_hidden_states = False)


my_model = model_init()
print(my_model)

train_data_loader = DataLoader(dataset=train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle= True)
eval_data_loader = DataLoader(dataset=eval_dataset,batch_size=VALID_BATCH_SIZE,shuffle= True)
test_data_loader = DataLoader(dataset=test_dataset,batch_size=TEST_BATCH_SIZE,shuffle= True)


model = model_init().to(device)
# model.load_state_dict(torch.load('model_v1', map_location='cpu'))

optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)


    train_loss = train_epoch(
                            model = model,
                            train_dl = train_data_loader ,    
                            device = device,
                            optimizer =optimizer, 
                            scheduler = scheduler, 

    )
    print(f'Train loss {train_loss}')


    eval_loss, metrics = eval_epoch(
                                model = model,
                                eval_dl = eval_data_loader ,    
                                device = device,
                                generate_report = False
    )

    print(f'Eval loss: {eval_loss}')
    print(f'Eval Metrics:  {metrics}')
    print('--------------------------------------------------------------------')

    save_model_name = 'output/' + str(epoch) + '.pt'
    torch.save(model, save_model_name)

# test_model = torch.load('full_model_v4' ,map_location='cpu')

# model_test(test_dl= test_data_loader,model=test_model, device= device)