from flask import Flask, render_template, url_for, request, redirect
from ner import test_camel
from helpers import helper
from newNer import predict_sent
import os 
import subprocess
import logging
import shutil
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/service', methods=['GET', 'POST'])
def test():
    # if request.method == "POST":
    #     inp = request.form['input']
    #     sentence = helper.prepare_sentence(inp)
    #     task = test_camel(sentence)
    #     # task = helper.final_result(task)
    #     return render_template('service.html', task=task, inp=inp, res=res, size=size, links=links)
    # else:
    #     return render_template('service.html', task='', inp='', res=[], size=0)
    if request.method == "POST":
        print('yes post')
        inp = request.form['input']
        print(inp)
        options = request.form.get('options')
        print(options)
        if(options == "camel"):
            sentence = helper.prepare_sentence(inp)
            task, labels , tokens = test_camel(sentence)
            res = helper.get_separate_entities(labels, tokens)
            links = helper.get_wiki_urls(res)
            size = len(res)
            #task = helper.final_result(task)
            print(task)
            return render_template('service.html', task=task, inp=inp, res=res, size=size, links=links, model="Camel model")
          
        else:
            sentence = helper.prepare_sentence(inp)
            #task = test_camel(sentence)
            task, labels, tokens= predict_sent(sentence)
            print('task')
            print(type(task))
            #!todo 
            #3- handling style
            res = helper.get_separate_entities(labels, tokens)
            links = helper.get_wiki_urls(res)
            # task = helper.final_result(task)
            size = len(res)
            return render_template('service.html', task=task, inp=inp, res=res, size=size, links=links, model="Default model")
    else:
        return render_template('service.html', task='', inp='', res=[], size=0, links=[], model="Default model")

@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == '__main__':

    # model_path = os.path.dirname(os.path.abspath(__file__))+'/model/camel'
    # model_path = os.path.dirname(os.path.abspath(__file__))+'/model/camel'

    # # os.environ["CAMELTOOLS_DATA"] = model_path
    # subprocess.call( 'mkdir /root/.camel_tools/', shell=True)
    # subprocess.call( 'mkdir /root/.camel_tools/', shell=True)
    # copy_path =  model_path+'/data'
    # copy_path =  model_path+'/data'
    # # subprocess.call(  'sudo cp -r '+  copy_path + '/ ' + '/root/.camel_tools/'    , shell=True)
    # subprocess.call(  'cp -r '+  copy_path + '/ ' + '/root/.camel_tools/'    , shell=True)
    # subprocess.call(  'cp -r '+  copy_path + '/ ' + '/root/.camel_tools/'    , shell=True)

    # shutil.copy(copy_path,'/root/.camel_tools/' )


    # subprocess.call(  'sudo cp -r '+  model_path + ' ' + '/root/.camel_tools/'    , shell=True)
    # print(subprocess.call('echo $CAMELTOOLS_DATA' , shell=True))
    # print(model_path)
    # print(copy_path)
    # print( os.listdir(model_path))
    # print( os.listdir('/root/.camel_tools/'))
    
    # print( os.listdir('/root/'))
    # print(subprocess.call(['ls', '-l', '/root/'] , shell=True))
    # print(subprocess.call(['ls', '-l', '/root/.camel_tools/'] , shell=True))




    # try:
    #     base_path =  os.path.expanduser('~')
    #     path  = os.path.expanduser('~')+ '/ANER_DEV/model/camel/'
    #     os.environ["CAMELTOOLS_DATA"] = path
    #     if not os.path.exists(path + 'data'):
    #         # os.system('export CAMELTOOLS_DATA=$path')
    #         print(path)
            

    #         # subprocess.call([ 'export', 'CAMELTOOLS_DATA=', str(path)] , shell=True)

    #         print(subprocess.call('echo $CAMELTOOLS_DATA' , shell=True))
    #         os.system('camel_data full')
           
    #     else:

    #         print(path)
    #         print(base_path)
    #         if not os.path.exists(base_path+'/.camel_tools/'):
    #             subprocess.call( [ 'mkdir' , base_path+'/.camel_tools/']   , shell=True)


    #         dest = base_path+'/.camel_tools/' 
    #         print(dest)
    #         subprocess.call(  'cp -r '+  path+'/data/' + ' ' + dest    , shell=True)
    #         subprocess.call(  'sudo cp -r '+  path+'/data/' + ' ' + '/.camel_tools/'    , shell=True)


    #         if not os.path.exists(base_path+'/root/.camel_tools/'):
    #             subprocess.call( [ 'mkdir' , '/root/.camel_tools/']   , shell=True)

    #         subprocess.call(  'sudo cp -r '+  path+'/data/' + ' ' + '/root/.camel_tools/'    , shell=True)



    #         print('camel arelady  downloaded')


    # except:
    #     print('cant download camel')
        
    app.run(debug=True)
