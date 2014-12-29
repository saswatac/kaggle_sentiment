#!/usr/bin/python

import sys
import preprocess_data
import pickle
import string

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def main():

    '''takes the raw test tsv file, trained model and bag of words feature on training set as input'''
    inputFile = sys.argv[1]
    modelFile = sys.argv[2]
    feature_preprocessor = sys.argv[3]
    
    f = open(inputFile, "r")
    phraseIds = []
    texts = []
    for line in f:
        parts = line.split('\t')
        phraseIds.append(parts[0])
        text_string = parts[2].translate(string.maketrans("", ""), string.punctuation)
        texts.append(text_string)
    stemmed_text = preprocess_data.stemInputText(texts)
    
    count_vectorizer = pickle.load(open(feature_preprocessor,"r"))
    word_features = count_vectorizer.transform(stemmed_text)
    
    classifier = pickle.load(open(modelFile,"r"))
    predictions = classifier.predict(word_features)
    outFile = open("submission.csv","w")
    outFile.write('PhraseId,Sentiment')
    for i in range(0,len(phraseIds)):
        outFile.write(','.join([phraseIds[i],predictions[i]]))
    outFile.close()

if __name__=='__main__':
    main()                            

