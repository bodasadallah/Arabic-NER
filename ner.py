
"""This module contains the CAMeL Tools Named Entity Recognition component.
"""

from helpers.helper import prepare_output
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from transformers import BertForTokenClassification, BertTokenizer
from helpers.helper import en_to_ar_camel
from camel_tools.ner import NERecognizer
import os 
import subprocess
# from camel_tools.data import DataCatalogue

# _LABELS = ['B-LOC', 'B-ORG', 'B-PERS', 'B-MISC', 'I-LOC', 'I-ORG', 'I-PERS',I-MISC', 'O']

# path = os.path.expanduser("~/.camel_tools")
# path  = os.path.expanduser('~')+ '/ANER_DEV/model/camel/'








model_path = os.path.dirname(os.path.abspath(__file__))+'/model/camel'
os.environ["CAMELTOOLS_DATA"] = model_path
copy_path =  model_path+'/data'

if not os.path.exists('/root/.camel_tools/'):
        subprocess.call( 'mkdir /root/.camel_tools/', shell=True)
        print('============= MAKING DIR ===============')



if not os.path.exists('/root/.camel_tools/data/'):
    # subprocess.call(  'sudo cp -r '+  copy_path + '/ ' + '/root/.camel_tools/'    , shell=True)

    ## This is the working one 
    subprocess.call(  'cp -r '+  copy_path + '/ ' + '/root/.camel_tools/'    , shell=True)


    print(subprocess.call('echo $CAMELTOOLS_DATA' , shell=True))
    print(model_path)
    print(copy_path)
    print( os.listdir(model_path))
    print( os.listdir('/root/.camel_tools/'))
    print( os.listdir('/root/'))


ner = NERecognizer.pretrained()

def test_camel(s):



    # '''Just for Testing'''
    # if not os.path.exists(path + 'data'):
    #     os.system('export CAMELTOOLS_DATA=$path')
    #     os.system('camel_data full')
        
    # path  = os.path.expanduser('~')+ '/ANER_DEV/model/camel/'
    
    


    

        
        # subprocess.call(  'cp -r '+  model_path + ' ' + '/root/.camel_tools/'    , shell=True)

    # print(subprocess.call('echo $CAMELTOOLS_DATA' , shell=True))

    # print(model_path)
    # print(copy_path)


    # print( os.listdir(model_path))
    # print( os.listdir('/root/.camel_tools/'))
    # print( os.listdir('/root/'))
    # print(subprocess.call(['ls', '-l', '/root/'] , shell=True))
    # print(subprocess.call(['ls', '-l', '/root/.camel_tools/'] , shell=True))


   
    
 

    # Predict the labels of a single sentence.
    # The sentence must be pretokenized by whitespace and punctuation.
    sentence = s.split()

    labels = ner.predict_sentence(sentence)
    res = ''
    # Print the list of token-label pairs
    for token, label in zip(sentence, labels):
        if(label == 'O'):
            continue
        # print("{}\t{}".format(label, token))
        s = f"{en_to_ar_camel[label]}     {label}      {token}"
        res = res + s + '\n'
    return res , labels , sentence



