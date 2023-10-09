# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 20:33:19 2022

@author: LabUser
"""

#Import main library
import numpy as np

#Import Flask modules
from flask import Flask, request, render_template

import joblib

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')


cls = joblib.load('finalized_model.sav')

#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    #obtain all form values and place them in an array, convert into integers
    inputValueandSuit = [int(x) for x in request.form.values()]
    #the cards come in as Value, Suit, Value, Suit, Value, Suit ...etc
    #they need to be added to an array in the order all 5 values then all 5 suits
    #i will also sort the first array
    # a[start_index:end_index:step] SLICE NOTATION
    print(f"incoing #'s = {inputValueandSuit}" )
    #card value array
    cardValues =  inputValueandSuit[::2]
    print(cardValues)
    cardSuites = inputValueandSuit[1::2]
    print(cardSuites)
    #sort the cardValues to make it easier for ML algo
    cardValues.sort()
    print(f"sorted cardvalues {cardValues}")    
    #Combine them all into a final numpy array
    sortedCombinedArray = cardValues + cardSuites
    #predict the category based on the input from the user
    prediction = cls.predict([sortedCombinedArray])
    print(f"prediction is : {prediction}")
    #return the prediction to the index.html webpage
    
    return render_template('index.html', prediction_text = f"The ML Algorithim Classified your hand as {translatePrediction(prediction)}")   

def translatePrediction(numPrediction):
    if(numPrediction == 0):
        return "0: Nothing in hand; High Card"
    elif(numPrediction == 1):
        return "1: One pair"
    elif(numPrediction == 2):
        return "2: Two pairs; two pairs of equal ranks within five cards"
    elif(numPrediction == 3):
        return "3: Three of a kind; three equal ranks within five cards"
    elif(numPrediction == 4):
        return "4: Straight; five cards, sequentially ranked with no gaps"
    elif(numPrediction == 5):
        return "5: Flush; five cards with the same suit"
    elif(numPrediction == 6):
        return "6: Full house; pair + different rank three of a kind"
    elif(numPrediction == 7):
        return "7: Four of a kind; four equal ranks within five cards"
    elif(numPrediction == 8):
        return "8: Straight flush; straight + flush"
    elif(numPrediction == 9):
        return "9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush"
 
            
 
            #Run app
if __name__ == "__main__":
    app.run(debug=True)