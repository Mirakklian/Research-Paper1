import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import csv




with open('C:\\Users\\Pratik Dutta\\Desktop\\SET-IMPLEMENTATION\\test.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
                tweet=row[0]

                # split into words
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(tweet)
                # convert to lower case
                tokens = [w.lower() for w in tokens]
                #print(tokens)
                # remove punctuation from each word
                import string
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                #print(stripped)
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                # filter out stop words
                from nltk.corpus import stopwords

                stop_words = set(stopwords.words('english'))
                words = [w for w in words if not w in stop_words]


                #start to calculate the polarity

                n=len(words)
                #print(n)
                total=0
                for i in range(0, n):
                    sent = TextBlob(words[i])
                    polarity = sent.sentiment.polarity
                    total=total+polarity
                    #print("\nword: " + words[i] + "    polarity is: ", polarity)
                #print(total)

                if total > 0 :
                    status="positive"
                elif total==0 :
                    status = "neutral"
                else :
                    status = "negative"

                print(tweet+"  -------->> "+status)

csvFile.close()




