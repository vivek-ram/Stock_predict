from flask import Flask, request , render_template , redirect , url_for
import numpy as np
import pandas as pd
from Stocks import*
import subprocess
import sys
import os

app = Flask(__name__,template_folder='templates')
port = int(os.environ.get("PORT", 5000))


# ht = df.to_html(classes='table table-striped')
# print(ht)


@app.route('/')
def my_form():
	return render_template('index.html')


@app.route('/' , methods=['POST'])
def ticker():
	ticker = request.form['fname']
	noofdays = request.form['fdays']
	holding = request.form['fhold']
	npofshares = request.form['fshare']
	createdata = create_data(ticker,int(noofdays))
	modeltrain = train(createdata)
	result = "Prediction of " + ticker + " after " + noofdays + " is: " + str(predict(createdata))
	bought = int(holding) * float(npofshares)
	actual = int(holding) * predict(createdata)
	val = round((actual-bought),2)
	profit_loss = ""

	if val > 0:
		profit_loss += "Profit : " + str(val) + "$"  
	else:
		profit_loss += "Loss : " + str(val) + "$"
	return render_template('index.html',id=result,temp=profit_loss)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=port)
