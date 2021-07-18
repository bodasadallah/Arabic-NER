from typing import Text
import http.client
import json
import re
import time



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
    
# NER = NERecognizer.pretrained()
# text = []
# trans = []
# with open("input.txt", "r") as input:
#     for line in input.read().splitlines():
#         text.append(line)
#
# for t in text:
#     translitrated_sent = driver(t,arabic)
#     sentence = driver(t,arabic).split()
#     labels = NER.predict_sentence(sentence)
#     ner = str(list(zip(sentence, labels)))
#     print(sentence)
#     trans.append( ''.join (ner) )
#
#     trans.append(translitrated_sent)



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




# with open("out.txt", "a") as f:
#     for line in trans:
#         print(line)
#         f.write(line+ '\n')

########### NER ####################


# franco_sentence = input()
# sentence = driver(franco_sentence,arabic).split()
# labels = NER.predict_sentence(sentence)

# Print the list of token-label pairs
#print(list(zip(sentence, labels)))


# regex to extract all arabic letters in string
#re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', sentence))
