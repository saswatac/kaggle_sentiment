#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import sys
import pickle

def parseTsvFile(file):
    '''parses the .tsv file and returns list of text_string and label'''
    text = []
    labels = []
    for line in file:
        parts = line.split('\t')
        text_string = parts[2].translate(string.maketrans("", ""), string.punctuation)
        text.append(text_string)
        labels.append(parts[3])
    print 'No. of lines: ',len(text)
    return (text, labels)

def stemInputText(text):
    '''stems the words in the input list of text'''
    stemmer = SnowballStemmer("english")
    stemmed_text = []
    for line in text:
        stemmed_line = ' '.join([stemmer.stem(word) for word in line.split()])
        stemmed_text.append(stemmed_line)
    return stemmed_text
    
def main():

    fileName = sys.argv[1]
    f = open(fileName, "r")
    text, labels = parseTsvFile(f)
    print 'stemming text...'
    stemmed_text = stemInputText(text)
    pickle.dump(stemmed_text, open("word_features.pkl", "w"))
    pickle.dump(labels, open("labels.pkl", "w"))
    f.close()

if __name__ == '__main__':
    main()            
