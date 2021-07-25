from numpy import string_
from franco import franco_trans
import re

en_to_ar_camel = {
    'B-LOC' : 'مكان',
    'B-ORG': 'مؤسسة', 
    'B-PERS': 'شخص',  
    'B-MISC': 'معنى بموضوعات متنوعة', 
    'I-LOC': 'مكان', 
    'I-ORG': 'مؤسسة', 
    'I-PERS': 'شحص',   
    'I-MISC': 'معنى بموضوعات متنوعة', 
}

en_to_ar = {
    "B-Artist" : "فنان",
    "I-Artist" :"فنان",
    "B-Sound": "صوت",
    "I-Sound":"صوت",
    "B-Educational": "تعليمي",
    "I-Educational":"تعليمي",
    "B-Building-Grounds":"أراضي البناء",
    "I-Building-Grounds":"أراضي البناء",
    "B-Population-Center":"مركز سكني",
    "B-Nation":"شعب(أمة)",
    "B-State-or-Province":"ولاية أو مقاطعة",
    "I-State-or-Province": "ولاية أو مقاطعة",
    "B-Water-Body": "مسطح مائي",
    "I-Water-Body":"مسطح مائي",
    "B-Land-Region-Natural": "أرض طبيعية",
    "I-Land-Region-Natural":"أرض طبيعية",
    "B-Software":"سوفتوير(برمجيات)",
    "I-Software":"سوفتوير(برمجيات)",
    "B-Scientist": "عالم",
    "B-Book":"كتاب",
    "I-Book":"كتاب",
    "I-Scientist":"عالم",
    "B-Group":"مجموعة",
    "B-Celestial":"سماوي",
    "B-Police":"شرطة",
    "I-Police":"شرطة",
    "I-Population-Center":"مركز سكني",
    "I-Celestial":"سماوي",
    "B-Engineer":"مهندس",
    "I-Engineer":"مهندس",
    "B-Projectile":"قذيفة",
    "B-Government":"حكومة",
    "I-Government":"حكومة",
    "B-Commercial":"تجاري",
    "I-Commercial":"تجاري",
    "B-Continent":"قارة",
    "B-Air":"هواء",
    "I-Air":"هواء",
    "B-Other_PER":"شخص",
    "I-Other_PER":"شخص",
    "I-Group":"مجموعة",
    "B-Politician":"سياسي",
    "I-Politician":"سياسي",
    "B-Athlete":"رياضي",
    "I-Athlete":"رياضي",
    "B-Religious_ORG":"مؤسسة دينية",
    "I-Religious_ORG":"مؤسسة دينية",
    "B-Path":"طريق",
    "I-Path":"طريق",
    "B-Media":"إعلام",
    "I-Media":"إعلام",
    "B-Non-Governmental":"غير حكومي",
    "I-Non-Governmental":"غير حكومي",
    "B-County-or-District":"مدينة أو ضاحية",
    "I-County-or-District":"مدينة أو ضاحية",
    "B-Businessperson":"رجل أعمال",
    "B-Lawyer":"محامي",
    "I-Lawyer":"محامي",
    "B-GPE-Cluster":"",
    "I-GPE-Cluster":"",
    "I-Nation":"شعب(أمة)",
    "B-Religious_PER":"شخص ديني",
    "I-Religious_PER":"شخص ديني",
    "I-Businessperson":"رجل أعمال",
    "B-Medical-Science":"علوم طبية",
    "I-Medical-Science":"علوم طبية",
    "B-Movie":"فيلم",
    "I-Movie":"فيلم",
    "B-Water":"ماء",
    "I-Water":"ماء",
    "B-Drug":"دواء",
    "B-Hardware":"عتاد",
    "I-Hardware":"عتاد",
    "B-Subarea-Facility":"منشأة منطقة فرعية",
    "I-Subarea-Facility":"منشأة منطقة فرعية",
    "B-Blunt":"فظ",
    "B-Airport":"مطار",
    "I-Blunt": "فظ",
    "I-Drug":"دواء",
    "B-Sports":"رياضة",
    "I-Sports":"رياضة",
    "B-Shooting":"رماية",
    "I-Shooting":"رماية",
    "B-Food":"طعام",
    "I-Food":"طعام",
    "I-Continent":"قارة",
    "B-Nuclear":"نووي",
    "I-Nuclear":"نووي",
    "B-Entertainment":"ترفيه",
    "I-Entertainment":"ترفيه",
    "I-Projectile":"قذيفة",
    "B-Land":"أرض",
    "B-Sharp":"حاد",
    "I-Airport":"مطار",
    "I-Land":"أرض",
    "B-Plant":"نبات",
    "I-Plant":"نبات",
    "B-Exploding":"منفجر",
    "I-Exploding":"منفجر",
    "B-Chemical":"كيميائي",
    "I-Chemical": "كيميائي",
}

wiki_base_url = "https://ar.wikipedia.org/wiki/"

def prepare_output(sentence, labels):
    res = ""
    for i in range(len(sentence)):
        res += sentence[i] + ": " + labels[i]+ "\n"
    return res


def final_result(l):
    s = ''
    print('l is :')
    print (l)
    for t in l:
        print(t)
    return s


def is_english(s):
    for l in s:
        if (65 <= ord(l) <= 90) or (97 <= ord(l) <= 122):
            return True
    return False


def is_ar(s):
    for ch in s:
        if ('\u0600' <= ch <= '\u06FF' or
                '\u0750' <= ch <= '\u077F' or
                '\u08A0' <= ch <= '\u08FF' or
                '\uFB50' <= ch <= '\uFDFF' or
                '\uFE70' <= ch <= '\uFEFF' or
                '\U00010E60' <= ch <= '\U00010E7F' or
                '\U0001EE00' <= ch <= '\U0001EEFF'):
            return True
    return False


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


def remove_r(s):
    l = s.split('\r')
    res = " ".join(l)
    return res


def remove_n(s):
    l = s.split('\n')
    res = " ".join(l)
    return res


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def prepare_sentence(s):
    temp = remove_n(s)
    temp = remove_r(temp)
    temp = remove_emoji(temp)
    l = temp.split(' ')
    for i in range(0, len(l)):
        w = l[i]
        if(not is_ar(w)):
            l[i] = franco_trans(w)
    res = " ".join(l)
    return res



def get_separate_entities(labels, tokens):
    """
        takes labels and token , return full name entity (mohamed, salah --> "mohamed salah")
        this will be used to search in wikipedia
    """
    res = []                                          
    b_before = False
    temp = ""
    key_value = ()
    for i in range(len(labels)):
        print(res)
        curr = labels[i]
        
        if("B-" in curr):
            if(b_before):
                key_value = (temp[:-1], 1)
                res.append(key_value)
                temp = tokens[i] + ' '
            else:
                b_before = True
                temp += tokens[i] + ' '
                if(i == len(labels)-1):
                    key_value = (temp[:-1], 1)
                    res.append(key_value)
                # print("temp is:" + str(temp))

        elif("I-" in curr):
            temp += tokens[i] + ' '
            if(i == len(labels)-1):
                key_value = (temp[:-1], 1)
                res.append(key_value)

        else:
            if(temp == ""):
                key_value = (tokens[i], 0)
                res.append(key_value) 
            else:
                key_value = (temp[:-1], 1)
                res.append(key_value)
                key_value = (tokens[i], 0)
                res.append(key_value) 
                temp = ""
                b_before = False
    
   

    print(res)
    return res 



def get_wiki_urls(names):
    """
        gets the correct url of wikepedia for a certain search entity
    """
    res = []
    
    for e in names:
        if(e[1] == 0):
            res.append('#')
            continue

        temp = ""
        name = e[0]
        
        for c in name:
            if(c == ' '):
                temp += '_'
            else:
                temp += c
        res.append(wiki_base_url + temp)

    return res








