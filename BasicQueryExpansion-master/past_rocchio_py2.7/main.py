'''
Created on Sep 21, 2012

@author: johnterzis

arguments: <precision> <query>

Contains the main loop of the application

'''

import json
import string
import sys

import nltk

import bingclient
import constants
import parser
import constants
import logging
import indexer
import rocchio
import common
import math
import PorterStemmer
import pandas as pd
import re
import numpy as np
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

# nltk.download('stopwords')
tokenizer = TweetTokenizer(preserve_case=False,
                           strip_handles=True,
                           reduce_len=True)

stopwords_english = stopwords.words('english')


def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def cleanstopword(tweet_tokens):
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweets_clean.append(''.join(word))
    return tweets_clean


# only if run as standalone script (not imported module) does, __name__  attribute defaults to __main__
# assume first arg is <precision> second is <query>
if __name__ == '__main__':

    logging.basicConfig(level=logging.ERROR, filename='amoo.txt')
    invertedFile = dict()
    termsFrequencies = dict()
    # create all singleton objects
    arglist = sys.argv
    if len(arglist) < 3:
        print "Usage: <precision> <query>"
        sys.exit(1)  # exit interpreter

    print 'Desired precision@10: {}'.format(arglist[1])

    precisionTenTarg = float(arglist[1])  # must convert string to float
    # 'eECeOiLBFOie0G3C03YjoHSqb1aMhEfqk8qe7Xi2YMs='
    # connect to client with key arg[1] and post a query with arg[3], query

    # bingClient = bingclient.BingClient(constants.BING_ACCT_KEY)
    indexer = indexer.Indexer()
    queryOptimizer = rocchio.RocchioOptimizeQuery(arglist[2])

    firstPass = 1
    precisionAtK = 0.00
    expandedQuery = arglist[2]
    queryWeights = {}

    DocumentList = pd.read_csv("Copy of Coronavirus Tweets - Copy.csv", encoding='latin', usecols=['ID', 'Description',
                                                                                                   'Sentiment',
                                                                                                   'IsRelevant',
                                                                                                   'tfVector'])

    # DocumentList['tfVector'] = [{}]
    # print(DocumentList)
    DocumentList['Description'] = DocumentList['Description'].apply(lambda x: ' '.join(re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
    # print(DocumentList)
    # DocumentList = DocumentList.dropna()
    DocumentList['Description'] = DocumentList['Description'].str.replace('[^a-zA-Z#]+', ' ')
    DocumentList['Description'] = DocumentList['Description'].apply(lambda x: tokenizer.tokenize(x))
    DocumentList['Description'] = DocumentList['Description'].apply(lambda x: cleanstopword(x))
    # DocumentList = DocumentList.dropna()
    DocumentList['Description'] = DocumentList['Description'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    # create new column with removed # tag
    # DocumentList['Description'] = np.vectorize(remove_pattern)(DocumentList['Description'], '#[\w]*')
    # create new column with removed @user
    # DocumentList['Description'] = np.vectorize(remove_pattern)(DocumentList['Description'], '@[\w]*')
    # DocumentList['Description'] = DocumentList['Description'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    # remove special characters, numbers, punctuations
    # DocumentList['Description'] = DocumentList['Description'].str.replace('[^a-zA-Z#]+', ' ')
    # remove short words
    # DocumentList['Description'] = DocumentList['Description'].apply(
    #    lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    # while precision at 10 is less than desired amt issue a query, obtain new precision metric, expand query, repeat
    # print(DocumentList)
    # count = DocumentList['Description'].str.split().str.len()
    # print(DocumentList['Description'].str.split().str.len() == 1)
    # DocumentList['Description'].drop(lambda x: x.str.split().str.len() == 1)
    # DocumentList.drop((x for x in DocumentList.Description if x.split().__len__() == 1), inplace=True)

    index_names = DocumentList[DocumentList['Description'].str.split().str.len() == 1].index
    DocumentList.drop(index_names, inplace=True)
    # print(DocumentList)

    DocumentList = DocumentList[0:100]

    while precisionAtK < precisionTenTarg:
        precisionAtK = 0.00  # reset precision each round
        # PROCESS A QUERY

        indexer.clearIndex()

        print 'Total number of results: %d' % len(DocumentList)

        # to calc precision@10 display documents to user and ask them to categorize as Relevant or Non-Relevant
        print '======================'

        # Reset collections for relevant ad nonrelevant documents
        relevantDocuments = []
        nonrelevantDocuments = []
        # print()
        # print(len(DocumentList))
        # print(DocumentList['ID'])

        for i in range(len(DocumentList)):

            DocumentList.iloc[i]['ID'] = i
            indexer.indexDocument(DocumentList.iloc[i])

            if DocumentList.iloc[i]['IsRelevant'] == 1:
                precisionAtK = precisionAtK + 1
                relevantDocuments.append(i)
                # print(i)

            elif DocumentList.iloc[i]['IsRelevant'] == 0:

                nonrelevantDocuments.append(i)
            else:
                print 'Invalid value entered!'

        precisionAtK = float(precisionAtK) / 100  # final precision@10 per round and 1000 we set for len of documents

        print ''
        print 'Precision@10 is: {}'.format(float(precisionAtK))
        print ''

        # expand query here by indexing and weighting current document list
        if (precisionAtK == 0):
            print 'Below desired precision, but can no longer augment the query'
            sys.exit()

        print 'Indexing results...'
        indexer.waitForIndexer()  # Will block until indexer is done indexing all documents

        for i in range(len(DocumentList)):
            #print (word_count(DocumentList.iloc[i]['Description']))
            DocumentList.at[i, 'tfVector'] = [word_count(DocumentList.iloc[i]['Description'])]

        # tfidf = TfidfVectorizer()
        # result = tfidf.fit_transform(DocumentList.iloc[i]['Description'].split('\n'))
        # print(result)
        # print(tfidf.vocabulary_)
        # DocumentList.at[i, 'tfVector'] = tfidf.vocabulary_
        # print(tfidf.vocabulary_)

        # Print inveretd file
        # print(DocumentList)
        # print(len(nonrelevantDocuments))
        DocumentList=DocumentList.dropna()
        DocumentList.to_csv('dd.csv')
        for term in sorted(indexer.invertedFile, key=lambda posting: len(indexer.invertedFile[posting].keys())):
            logging.info("%-30s %-2s:%-3d %-2s:%-3d %-3s:%-10f" % (
                term, "TF", indexer.termsFrequencies[term], "DF", len(indexer.invertedFile[term]), "IDF",
                math.log(float(len(DocumentList)) / len(indexer.invertedFile[term].keys()), 10)))

        print '======================'
        print 'FEEDBACK SUMMARY'

        if (precisionAtK < precisionTenTarg):
            print ''
            print 'Still below desired precision of %f' % precisionTenTarg
            queryWeights = queryOptimizer.Rocchio(indexer.invertedFile, DocumentList, relevantDocuments,
                                                  nonrelevantDocuments)  # optimize new query here

            newTerms = common.getTopTerms(expandedQuery, queryWeights, 2)
            expandedQuery = expandedQuery + " " + newTerms[0] + " " + newTerms[1]
            firstPass = 0

            print 'Augmenting by %s %s' % (newTerms[0], newTerms[1])

    # precision@10 is > desired , return query and results to user
    print 'Desired precision reached, done'
