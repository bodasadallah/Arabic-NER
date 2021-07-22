from flask import Flask, render_template, url_for, request, redirect
from ner import test_camel
from helpers import helper
from newNer import predict_sent



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
    app.run(debug=True)
