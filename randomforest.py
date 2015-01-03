#!/usr/bin/python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy
import sys
import pickle
from time import time

list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

def main():

    features_file = sys.argv[1]
    labels_file = sys.argv[2]
    
    features = pickle.load(open(features_file,"r"))
    labels = numpy.array(pickle.load(open(labels_file,"r")))
     
    #construct bag of words feature
    print 'constructing bag of words feature...'
    countVectorizer = CountVectorizer(stop_words="english")
    word_features = countVectorizer.fit_transform(features)

    clf = OneVsRestClassifier(RandomForestClassifier())
    print 'training model with full data...'
    t0 = time()
    clf.fit(word_features,labels)
    t1 = time()
    print 'training took:',(t1-t0)/1000,' seconds'
    pickle.dump(clf,open("randomforest_model.pkl","w"))
    pickle.dump(countVectorizer,open("bag_of_words.pkl","w"))

if __name__=='__main__':
    main()
