#!/usr/bin/python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
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

    kf = KFold(len(labels))
    clf = MultinomialNB()
    train_accuracies = []
    test_accuracies = []
    count=1
    for train_index, test_index in kf:
        features_train, features_test = word_features[train_index], word_features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        t0 = time()
        print 'training fold:',count
        clf.fit(features_train, labels_train)
        t1 = time()
        print 'training took:',(t1-t0)/1000,' seconds'
        train_pred = clf.predict(features_train)
        test_pred = clf.predict(features_test)
        train_accuracy = accuracy_score(labels_train, train_pred)
        test_accuracy = accuracy_score(labels_test, test_pred)
        print 'training accuracy:',train_accuracy
        print 'test accuracy:',test_accuracy
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        count=count+1
    
    print 'overall training accuracy: ',sum(train_accuracies)/float(len(train_accuracies))
    print 'overall test accuracy: ',sum(test_accuracies)/float(len(test_accuracies))
    #now train the model again with complete data and save the model
    clf = MultinomialNB()
    print 'training model with full data...'
    t0 = time()
    clf.fit(word_features,labels)
    t1 = time()
    print 'training took:',(t1-t0)/1000,' seconds'
    pickle.dump(clf,open("naive_bayes_model.pkl","w"))
    pickle.dump(countVectorizer,open("bag_of_words.pkl","w"))

if __name__=='__main__':
    main()
