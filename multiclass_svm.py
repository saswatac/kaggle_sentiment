#!/usr/bin/python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
import numpy
import sys
import pickle
from time import time

def main():

    features_file = sys.argv[1]
    labels_file = sys.argv[2]
    
    features = pickle.load(open(features_file,"r"))
    labels = numpy.array(pickle.load(open(labels_file,"r")))
     
    #construct bag of words feature
    print 'constructing bag of words feature...'
    countVectorizer = CountVectorizer(stop_words="english")
    word_features = countVectorizer.fit_transform(features)

    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    print 'training model with full data...'
    t0 = time()
    clf.fit(word_features,labels)
    t1 = time()
    print 'training took:',(t1-t0)/1000,' seconds'
    pickle.dump(clf,open("multiclass_model.pkl","w"))
    pickle.dump(countVectorizer,open("bag_of_words.pkl","w"))

if __name__=='__main__':
    main()
