#Importing all the necessary libraries
import nltk
from nltk.corpus import PlaintextCorpusReader
import re
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords') 
nltk.download('punkt')
from nltk.corpus import stopwords
from collections import Counter
from itertools import chain
import pickle
import os
import numpy as np
from heapq import nlargest
import pandas as pd
import sys

def pickle_open(file_name):
    with open(file_name, 'rb') as f:
        loader = pickle.load(f)
    f.close()
    return loader
	
tf=pickle_open("term_frequency.pkl")
df=pickle_open("doc_freq.pkl")
file_index=pickle_open("file_index.pkl")
doc_len=pickle_open("doc_len.pkl")

all_query = {}
query = open(sys.argv[1],'r')
for q in query:
    content = q.split("\t")
    all_query[content[1].replace('\n','')] = content[0]

retrieved_docs = {key: [] for key in all_query}

b=0.75
k=1.2
total_files=len(file_index)

for q in all_query:
    query = q.lower()
    query = re.sub('\W+|_', ' ', query)
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stopword_list = [stopword.lower() for stopword in stop_words]
    tokens = word_tokenize(query)
    stems = [ps.stem(token) for token in tokens]
    words = [word for word in stems if word not in stopword_list]

    score = dict()
    doc_len_mean = np.array(list(doc_len.values())).mean()
    for file_id in range(len(file_index)):
        score[file_id]=0
        for word in words:
            TF = 0
            if word in tf :
                if file_id in tf[word] :
                    TF = tf[word][file_id] 
            NDF = df[word] if word in df else 0
            IDF = np.log((total_files-NDF+0.5)/(NDF+0.5))
            numerator = TF*IDF*(k+1)
            denominator = TF+k*(1-b+b*(doc_len[file_id]/doc_len_mean))
            score[file_id]+=(numerator/denominator)
    
    res = nlargest(10, score, key = score.get)
    res = [file_index[fid] for fid in res]
    retrieved_docs[q]=res

qid=[]
iteration=[]
docid=[]
relevance=[]

for q in retrieved_docs:
    for doc in retrieved_docs[q]:
        qid.append(all_query[q])
        iteration.append(1)
        docid.append(doc)
        relevance.append(1)

data = {
    'QueryId' : qid,
    'Iteration' : iteration,
    'DocId' : docid,
    'Relevance' : relevance
}
df = pd.DataFrame(data)
df.to_csv('Q4/QRels-BM25.csv',index=False)