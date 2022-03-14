# >>> x="@peter I really love that shirt at #Macy. http://bit.ly//WjdiW4" >>> ' '.join(re.sub("(@[A-Za-z0-9]+)|([
# ^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()) 'I really love that shirt at Macy' >>> x="@shawn Titanic tragedy
# could have been prevented Economic Times: Telegraph.co.ukTitanic tragedy could have been preve...
# http://bit.ly/tuN2wx" >>> ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
# 'Titanic tragedy could have been prevented Economic Times Telegraph co ukTitanic tragedy could have been preve' >>>
# x="I am at Starbucks http://4sq.com/samqUI (7419 3rd ave, at 75th, Brooklyn) " >>> ' '.join(re.sub("(@[
# A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()) 'I am at Starbucks 7419 3rd ave at 75th Brooklyn'


import json
import sys
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


# only if run as standalone script (not imported module) does, __name__  attribute defaults to __main__
# assume first arg is <precision> second is <query>
if __name__ == '__main__':

    DocumentList = pd.read_csv("Copy of Coronavirus Tweets - Copy.csv", encoding='latin', usecols=['ID', 'Description',
                                                                                                   'Sentiment',
                                                                                                   'IsRelevant'])

    DocumentList['tfVector'] = ""

    print(DocumentList)
    DocumentList['Description'] = DocumentList['Description'].apply(lambda x: ' '.join(re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
    print(DocumentList)