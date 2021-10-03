import os
import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from functools import reduce
from collections import Counter
np.random.seed(2018)

stemmer = SnowballStemmer(language='english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):  # make the adjustment
        if token not in STOPWORDS and len(token) > 3 and token not in countries:
            result.append(lemmatize_stemming(token))
    return result

# reading additional datasets
with open('countries.txt', 'r') as file:  # list of all countries
    data = file.read().replace('\n', ' ')
countries = gensim.utils.simple_preprocess(data)
UNSD = pd.read_csv('/Users/maxwinder/Documents/Msc/FDS/fds_assignment1/UNSD — Methodology.csv')

# reading speeches
sessions = np.arange(25, 76)
data=[]
for session in sessions:
    directory = "./bin/TXT/Session "+str(session)+" - "+str(1945+session)  # make sure good path
    for filename in os.listdir(directory):
        f = open(os.path.join(directory, filename))
        if filename[0]==".": #ignore hidden files
            continue
        splt = filename.split("_")
        data.append([session, 1945+session, splt[0], f.read()])

df_speech = pd.DataFrame(data, columns=['Session','Year','ISO-alpha3 Code','Speech'])

# selecting test set
# UNSD[(UNSD['Region Name'] == 'Africa')][:52]  # 60 developing countries
# UNSD[(UNSD['Region Name'] == 'Europe')]  # 52 developed countries
selection1 = list(UNSD[(UNSD['Region Name'] == 'Africa')]['ISO-alpha3 Code'].values)[:52] + list(UNSD[(UNSD['Region Name'] == 'Europe')]['ISO-alpha3 Code'].values)
test = df_speech[df_speech['ISO-alpha3 Code'].isin(selection1)].query('Year >= 2005 & Year <= 2015')  # test data set

#preprocessing
processed_docs = df_speech.Speech.map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

#training model
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=6, id2word=dictionary, passes=2, workers=4)  #tf-idf

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

# deploying model
processed_test = test.Speech.map(preprocess)
dictionary_test = gensim.corpora.Dictionary(processed_test)
bow_corpus_test = [dictionary_test.doc2bow(doc) for doc in processed_test]
tfidf_test = gensim.models.TfidfModel(bow_corpus_test)
corpus_tfidf_test = tfidf_test[bow_corpus_test]

bound = 0.1  # at least this score on topic for a save
topics = []
for i in corpus_tfidf_test:
    sorted(lda_model_tfidf[i], key=lambda tup: -1*tup[1])
    topics.append([j[0] for j in sorted(lda_model_tfidf[i], key=lambda tup: -1*tup[1]) if j[1]> bound])  


#grouping results
test = test.reset_index().set_index('ISO-alpha3 Code').join(UNSD.set_index('ISO-alpha3 Code')['Developed / Developing Countries'])
test['Topics'] = topics

Counter(reduce(lambda a,b : a+b, test.Topics))
Counter(reduce(lambda a,b : a+b, test[test['Developed / Developing Countries'] == 'Developed'].Topics))
Counter(reduce(lambda a,b : a+b, test[test['Developed / Developing Countries'] == 'Developing'].Topics))

lda_model_tfidf.print_topic(0, 20)