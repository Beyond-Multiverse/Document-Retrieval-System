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

df=pickle_open("doc_freq.pkl")
file_index=pickle_open("file_index.pkl")
doc_norm=pickle_open("doc_norm.pkl")
combined_doc=pickle_open("combined_doc.pkl")

all_query = {}
query = open(sys.argv[1],'r')
for q in query:
    content = q.split("\t")
    all_query[content[1].replace('\n','')] = content[0]
 
 retrieved_docs = {key: [] for key in all_query}
 
 total_files=len(file_index)
for q in all_query:
    query = q.lower()
    query = re.sub('\W+|_', ' ', query)
    ps = PorterStemmer()
    stopword_list = set(stopwords.words('english'))
    stopword_list = [stopword.lower() for stopword in stopword_list]
    tokens = word_tokenize(query)
    stems = [ps.stem(token) for token in tokens]
    words = [word for word in stems if word not in stopword_list]


    TF_IDF = []
    for word in words:
        TF_IDF = np.append(TF_IDF,(words.count(word)*np.log(total_files/df[word])))

    TF_IDF=np.array(TF_IDF)/np.linalg.norm(TF_IDF)

    score = dict()

    for file in range(total_files):
        doc_vector=[]
        for word in words:
            tf_idf=(combined_doc[file].count(word)*np.log(total_files/df[word]))
            doc_vector.append(tf_idf)
        doc_vector=np.array(doc_vector)/doc_norm[file]
        score[file]=np.dot(TF_IDF,doc_vector)

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
df.to_csv('Q4/QRels-TFIDF.csv',index=False)