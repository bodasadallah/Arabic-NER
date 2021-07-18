from typing import Text
import http.client
import json
import re
import time
from ner import *


# arabic language code 
arabic = 'ar-t-i0-und'


#makes https get request and get the response
def trans_request(input, itc):
    '''
        input: the Arabizi word
        itc: the language code 
    '''
    conn = http.client.HTTPSConnection('inputtools.google.com')
    conn.request('GET', '/request?text=' + input + '&itc=' + itc + '&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=franco-to-arabic')
    res = conn.getresponse()
    return res

def driver(input, itc):
    output = ''
    if ' ' in input:
        input = input.split(' ')
        for i in input:
            res = trans_request(input = i, itc = itc)
            res = res.read()
            if i==0:
                output = str(res, encoding = 'utf-8')[14+4+len(i):-31]
            else:
                output = output + ' ' + str(res, encoding = 'utf-8')[14+4+len(i):-31]
                output = output.rstrip()
    else:
        res = trans_request(input = input, itc = itc)
        res = res.read()
        output = str(res, encoding = 'utf-8')[14+4+len(input):-31]
    return output
    




def franco_ner(s):
    NER = NERecognizer.pretrained()
    # translitrated_sent = driver(s, arabic)
    sentence = driver(s, arabic).split()
    labels = NER.predict_sentence(sentence)
    print(labels)
    ner = str(list(zip(sentence, labels)))
    return ner


def franco_trans(s):
    sentence = driver(s, arabic)
    return sentence
    



if __name__ == '__main__':
    s = """
علشان الهبد في الكورة مش جديد
اعرف اخبار الدوري من العميد
اطلبأو نزل MyOrange واشترك في خدمة الدوري من اورنچ
و اعرف كل كبيرة و صغيرة من العميد احمد حسن
"""

    print(franco_ner(s))
    # print(fun('o'))







