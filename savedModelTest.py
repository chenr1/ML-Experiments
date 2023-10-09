# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:30:14 2022

@author: CVHenry
"""

import joblib

cls = joblib.load('finalized_model.sav')
fiveCards = []

#card numbers come first 
fiveCards.append(1) 
fiveCards.append(2) 
fiveCards.append(3) 
fiveCards.append(4) 
fiveCards.append(5) 

#then the suit associated with the card
fiveCards.append(1) 
fiveCards.append(2) 
fiveCards.append(3) 
fiveCards.append(4) 
fiveCards.append(1) 

print(fiveCards)



answer = cls.predict([fiveCards])

print(answer)