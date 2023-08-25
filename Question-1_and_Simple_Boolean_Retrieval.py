#Importing all the necessary libraries
import sys
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
from collections import defaultdict
import pickle
import os
import numpy as np
from heapq import nlargest
import pandas as pd

listOfFiles = PlaintextCorpusReader('english-corpora', '.*')
total_files = len(listOfFiles.fileids())
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stopword_list = [stopword.lower() for stopword in stop_words]
doc_norm={}
combined_doc=[]
unique_words = set()
file_index = {}
doc_len = {}

def preprocess(content):
    content = re.sub(r'([.][a-zA-Z]{1,}[-])|([.][a-z]{1,}.+[{].+[}])|([<][a-z ]{1,}.+[/][>])','',content)
    content = content.lower()
    content = content.replace('\\n', ' ')
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    content = re.sub('\W+|_', ' ', content)
    tokens = word_tokenize(content)
    tokens = [token for token in tokens if len(tokens)>1]
    stems = [ps.stem(token) for token in tokens]
    syllables = [syllable for syllable in stems if syllable not in stopword_list]
    return syllables
    
file_id = -1

#Iterating for all files
for file in listOfFiles.fileids():
    
    file_id += 1
    
    #fetching the source file location and name
    plain_doc="english-corpora/"+str(file)
    file_index[file_id] = plain_doc
    with open(plain_doc, 'r', encoding='UTF-8') as f:

        syllables = preprocess(f.read())

    doc_len[file_id] = len(syllables)
    combined_doc.append(syllables)
    unique_words = unique_words.union(set(syllables))
 
 
file_id = -1

for doc in combined_doc:
    file_id+=1
    val=0
    for word in set(doc):
        val+=np.square(doc.count(word)*np.log(total_files/df[word]))
    doc_norm[file_id]=(np.sqrt(val))
    
 def pickle_save(file_name, dumper):
    with open(file_name, 'wb') as f:
        pickle.dump(dumper, f)
    f.close()

pickle_save('file_index.pkl', file_index)
pickle_save('doc_len.pkl', doc_len)
pickle_save('term_frequency.pkl', tf)
pickle_save('doc_freq.pkl', df)
pickle_save('doc_norm.pkl', doc_norm)
pickle_save('combined_doc.pkl', combined_doc)

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

    ans = set()

    for i in range(len(words)):
        if words[i] == "or" :
            continue
        if words[i] == "and" :
            i+=1
            if words[i] in presence.keys() and i!=len(words):
                ans = ans.intersection(presence[words[i]])
            continue
        if words[i] == "not" :
            i+=1
            if words[i] in presence.keys() and i!=len(words):
                ans = ans.difference(presence[words[i]])
            continue
        if words[i] in presence.keys():
            ans = ans.union(presence[words[i]])
        
        
        res = ['english-corpora/' + sub for sub in ans]
        retrieved_docs[q]=res[:20]
    

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
df.to_csv('Q4/QRels-Simple_Boolean_Retrieval.csv',index=False)
