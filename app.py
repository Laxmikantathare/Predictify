from flask import Flask, render_template, request
import pickle
from flask_pymongo import PyMongo
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
import math
from dotenv import load_dotenv
import os

load_dotenv()
password_mongo = os.getenv("password")
print(password_mongo)


heart = pickle.load(open('heartweb1.pkl', 'rb'))
kidney = pickle.load(open('kidneyfinal.pkl', 'rb'))
brainm = pickle.load(open('braingraph93.pkl', 'rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/heartm")
def heartm():
    return render_template("Heart.html")

@app.route("/kidneym")
def kidneym():
    return render_template("kidney.html")

@app.route("/brain")
def brain():
    return render_template("brain.html")

@app.route("/about_heart")
def about_heart():
    return render_template("about_heart.html")

@app.route("/about_kidney")
def about_kidney():
    return render_template("about_kidney.html")

@app.route("/about_brain")
def about_brain():
    return render_template("about_brain.html")

@app.route('/prediction')
def man():
    return render_template('heart_prediction.html')

@app.route('/contact')
def mancon():
    return render_template('contact.html')


@app.route("/prediction", methods=['POST'])
def prediction():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    Name =request.form['Name']
  
  
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]])
    pred = heart.predict(arr)
    app.config["MONGO_URI"] = "mongodb+srv://sahilmatondkar9:z4zjlBEqabFapJSY@predictify.yh50wwy.mongodb.net/prediction_data"
    db = PyMongo(app).db

    db[Name].insert_one({"Age":data1})
    db[Name].insert_one({"sex":data2})
    db[Name].insert_one({"chestpain":data3})
    db[Name].insert_one({"RestBP":data4})
    db[Name].insert_one({"Chol":data5})
    db[Name].insert_one({"Fbs":data6})
    db[Name].insert_one({"RestECG":data7})
    db[Name].insert_one({"MaxHR":data8})
    db[Name].insert_one({"Exang":data9})
    db[Name].insert_one({"Oldpeak":data10})
    db[Name].insert_one({"Slope":data11})
    db[Name].insert_one({"ca":data12})
    db[Name].insert_one({"thal":data13})

    return render_template('after.html', data=pred)

@app.route("/prevention")
def prevention():
     return render_template("heart_prevention.html")

@app.route("/kidney_prevention")
def kidney_prevention():
     return render_template("kidney_prevention.html")

@app.route('/kidney_prediction')
def kidney_prediction():
    return render_template('kidney_prediction.html')

@app.route('/kidney_prediction',methods=['POST'])
def upload():
    
    file = request.files['file']

    img = Image.open(file.stream)

    img = img.resize((224, 224))

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
 
    kidneyres = kidney.predict(img_array)

    return render_template('after_kidney.html',data=str(math.floor(kidneyres)))


@app.route('/brain_prediction')
def brain_prediction():
    return render_template('brain_prediction.html')

@app.route('/brain_prevention')
def brain_prevention():
    return render_template('brain_prevention.html')

@app.route('/brain_prediction',methods=['POST'])
def brainpred():
    
    file = request.files['file']

    img = Image.open(file.stream)

    img = img.resize((224, 224))

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
 
    brainres = brainm.predict(img_array)

    return render_template('after_brain.html',data=str(math.floor(brainres)))
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=['POST'])
def contact():
    contdata1 = request.form['nam']
    contdata2 = request.form['ema']
    contdata3 = request.form['mes']

    app.config["MONGO_URI"] = "mongodb+srv://sahilmatondkar9:z4zjlBEqabFapJSY@predictify.yh50wwy.mongodb.net/contact_data"
    db = PyMongo(app).db

    db[contdata1].insert_one({"Name":contdata1})
    db[contdata1].insert_one({"Email":contdata2})
    db[contdata1].insert_one({"Message":contdata3})

    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)