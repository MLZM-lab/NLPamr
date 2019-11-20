"""
ssh  ubuntu@147.188.173.254


cd /home/ubuntu/Projects

mkdir Danesh_NLP_15.11.19
cd /home/ubuntu/Projects/Danesh_NLP_15.11.19
mv /home/ubuntu/nltk_data . #this was downloaded at some point in atutorial of nlp for word2vec

mkdir 1-Trials

cd /home/ubuntu/Projects/Danesh_NLP_15.11.19/1-Trials

source activate NLPantibiotics

python #It's python 3
"""


from Bio import Entrez
from urllib.error import HTTPError

import xml.etree.ElementTree as et
import pickle

import spacy
import numpy as np
import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from geotext import GeoText


"""
#############Entrez part
"""

Entrez.email = 'lisandracady@gmail.com'

#Do the search
handle = Entrez.esearch(db="pubmed", term="antibiotic")
record = Entrez.read(handle)
count = int(record['Count'])
#The online seach gives 809,967 results
handle = Entrez.esearch(db="pubmed", term="antibiotic", retmax=count)
record = Entrez.read(handle)


#Fetch details. Since they are more than 10000 ids, use epost instead of efetch (https://stackoverflow.com/questions/46579694/biopythons-esearch-does-not-give-me-full-idlist)
id_list = record['IdList']
post_xml = Entrez.epost("pubmed", id=",".join(id_list))
search_results = Entrez.read(post_xml)


#Explore the object
type(search_results)
#<class 'Bio.Entrez.Parser.DictionaryElement'>
search_results.keys()
#dict_keys(['QueryKey', 'WebEnv'])


#Actualy download the entries now with efetch batch by batch
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"] 

batch_size = 10000

for start in range(0, count, batch_size):
    end = min(count, start+batch_size)
    print("Going to download record %i to %i" % (start+1, end))
    out_handle = open("AntibioPubmedSearch_%i_%i.xml" % (start+1, end), "w")
    attempt = 0
    while attempt < 3:
        attempt += 1
        try:
            fetch_handle = Entrez.efetch(db="pubmed",
                                         retstart=start, retmax=batch_size,
                                         webenv=webenv, query_key=query_key, retmode='xml')
        except HTTPError as err:
            if 500 <= err.code <= 599:
                print("Received error from server %s" % err)
                print("Attempt %i of 3" % attempt)
                time.sleep(15)
            else:
                raise
    data = fetch_handle.read()
    fetch_handle.close()
    out_handle.write(data)
    out_handle.close()


"""
############## Read back the saved txt xml(?) file(s) (https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c)
"""

#pscp -i "C:\Users\zepedaml\Documents\SSHkey.ppk"  ubuntu@147.188.173.254:/home/ubuntu/Projects/Danesh_NLP_15.11.19/1-Trials/AntibioPubmedSearch_1_10000.xml .
#Explore the labels of the xml
#!grep -B1 -A1 "PublicationType UI" AntibioPubmedSearch_1_10000.xml | sort -u
#the UIs that I should remove are: 
#D016454 for review
#D000078182 for systematic review
#D016420 Comment
#UI="D016439">Corrected and Republished Article</PublicationType>
#UI="D016456">Historical Article</PublicationType>
#UI="D017065">Practice Guideline</PublicationType>
#UI="D023361">Validation Studies</PublicationType>
#UI="D059040">Video-Audio Media</PublicationType>
#UI="D064886">Dataset</PublicationType>


def get_xml_info(node):
    try:
        title = str(node.find("MedlineCitation").find("Article").find("ArticleTitle").text.encode('ascii', 'ignore'))[2:-1]
    except AttributeError:
        title = np.nan
    try:
        abstract = str(node.find("MedlineCitation").find("Article").find("Abstract").find("AbstractText").text.encode('ascii', 'ignore'))[2:-1]
    except AttributeError:
        abstract = np.nan
    try:
        year = node.find("MedlineCitation").find("Article").find("ArticleDate").find("Year").text
        month = node.find("MedlineCitation").find("Article").find("ArticleDate").find("Month").text
        day = node.find("MedlineCitation").find("Article").find("ArticleDate").find("Day").text
        date = "-".join([day, month, year])
    except AttributeError:
        date = np.nan
    return title, abstract, date


xtree = et.parse("AntibioPubmedSearch_1_10000.xml")
xroot = xtree.getroot()

df_cols = ["Title", "Abstract", "Date"]
rows = []
rows_review = []
rows_unknownType = []

for node in xroot:
    try:
        types = []
        for child in node.find("MedlineCitation").find("Article").find("PublicationTypeList").iter("PublicationType"):
            types.append(child.attrib['UI'])
        if types.count("D016454") > 0: #If it's a review. D016454 is the UI of a review
            title, abstract, date = get_xml_info(node)
            rows_review.append({"Title": title, "Abstract": abstract, "Date": date})
        else: 
            title, abstract, date = get_xml_info(node)
            rows.append({"Title": title, "Abstract": abstract, "Date": date})
    except AttributeError:
        title, abstract, date = get_xml_info(node)
        rows_unknownType.append({"Title": title, "Abstract": abstract, "Date": date})



df = pd.DataFrame(rows, columns=df_cols)
df_review = pd.DataFrame(rows_review, columns=df_cols)
df_unknownType = pd.DataFrame(rows_unknownType, columns=df_cols)

df.shape
#(9051, 3)
df_review.shape
#(868, 3)
df_unknownType.shape
#(81, 3)



#Pickle the object just tu try it out for later safety usage
pickle.dump( df, open("AntibioPubmedSearch_dfBatch1.p", "wb") )
#To load it back in use 
df = pickle.load(open("AntibioPubmedSearch_dfBatch1.p", "rb"))


"""
#############NLP part
"""

#! python -m spacy download en  #I had to download this in shell


"""
Extract sentences from long abstract text
"""

def extract_sentence(text):
    """
    :param text 
    :return: list of sentences
    """
    nlp = spacy.load("en")
    nlp_object=nlp(text)
    return(list(map(lambda x: str(x),list(nlp_object.sents))))


sentence_list = extract_sentence(df.Abstract[0])


"""
Not sure what this is for in Dan's workflow

def convert_sentence_entities(sentence):
    """
    :param text 
    :return: list of sentences
    """
    nlp= spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    return(doc.ents)

doc_ents = list(map(convert_sentence_entities, sentence_list))

"""



"""
##Create spark dataframe
"""

def index_sentence(sentence_list):
    return list(zip(np.arange(2), sentence_list))


index_sentence_list = index_sentence(sentence_list)


# Create spark
spark = SparkSession.builder.getOrCreate()
# Print spark
print(spark)
#<pyspark.sql.session.SparkSession object at 0x7f5f3d0f8910>

def create_spark_data_frame(index_sentence_list):
    return spark.createDataFrame([(float(tup[0]), tup[1]) for tup in index_sentence_list], ["id", "sentence"])

sentenceDataFrame = create_spark_data_frame(index_sentence_list)

"""
Tokenize with spark
"""

def tokenizer_spark(sentenceDataFrame):
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    return tokenizer.transform(sentenceDataFrame).select("words")


TokenizedSentenceDataFrame = tokenizer_spark(sentenceDataFrame)


"""
Remove stop words with spark
"""

"""
def delete_stopwords(TokenizedSentenceList):
    """
    :param text 
    :return: list of sentences
    """
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    tokens = [TokenizedSentenceList.text for token in text if not token.is_stop]
    return(' '.join(tokens))

#The function above is Dan's, the function below is mine

def remove_stop_words(sentence_list):
    spacy_nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    results = []
    for text in sentence_list:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results

sentence_list = remove_stop_words(sentence_list)

#Then I do something to get a words corpus here
words = []
for text in sentence_list:
    for word in text.split(' '):
        words.append(word)

words = set(words)
"""

def remover_stop_word_spark(TokenizedSentenceDataFrame):
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    return remover.transform(TokenizedSentenceDataFrame).select("filtered")


TokenizedSentenceNoStopWords = remover_stop_word_spark(TokenizedSentenceDataFrame)


"""
Return the tokenized extracted sequences without stop words into a pd df
"""

df_TokenizedSentenceNoStopWords = TokenizedSentenceNoStopWords.toPandas()


"""
Extract the countries from a sentence
"""

def extract_cities_countries(sentence):
    places = GeoText(sentence)
    output=[]
    output.append(np.unique(np.squeeze(places.cities)).tolist())
    output.append(np.unique(np.squeeze(places.countries)).tolist())
    return(output)


countries_list = list(map(extract_cities_countries, sentence_list))


"""
Ok, so up to here I have implemented Dan's pipeline to make it run. Now Ineed to put it in a for so that all the functions are applied to all the sentences in my entrez df
"""


#XXXXXXXXXXXXXXXXXXXXXXX

#Add the pandas timeStamps to the df

"""
I'm also just going to see how all the words look in a plot using the word2vec workflow with gensim
"""

#XXXXXXXXXXXXXXXXXXX


exit()

"""
########## Put everything in github
"""

mkdir /home/ubuntu/Projects/Danesh_NLP_15.11.19/GITHUB
cd /home/ubuntu/Projects/Danesh_NLP_15.11.19/GITHUB


echo "# NLPantibiotics" >> README.md
git init
git add README.md
git config --global user.email "lisandracady@gmail.com"
git commit -m "first commit"
git remote add origin https://github.com/MLZM-lab/NLPantibiotics.git
git push -u origin master
#Put my username and password


