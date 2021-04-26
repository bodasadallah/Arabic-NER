from flask import Flask, render_template, url_for, request, redirect
# from ner import test_camel
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
    #     return render_template('service.html', task=task, inp=inp)
    # else:
    #     return render_template('service.html', task='', inp='')
    if request.method == "POST":
        inp = request.form['input']
        sentence = helper.prepare_sentence(inp)
        #task = test_camel(sentence)
        task = predict_sent(sentence)
        # task = helper.final_result(task)
        return render_template('service.html', task=task, inp=inp)
    else:
        return render_template('service.html', task='', inp='')

@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == '__main__':
    app.run(debug=True)
