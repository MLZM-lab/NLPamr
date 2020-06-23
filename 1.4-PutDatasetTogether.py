
import sys
import os
import pickle
import pandas as pd
import re
import numpy as np
import spacy


"""###############################"""
"""############Functions##########"""
"""###############################"""



def walk(Array):  #To flatten the lists iof lists of lists...
    RESULT = []
    if isinstance(Array, float):
        return RESULT
    else:
        for obj in Array:
            if not isinstance(obj, str):
                for item in obj:
                    RESULT.append(item)
            else:
                RESULT.append(obj)
        return RESULT



def putListsTogether(SOURCE_DIR, batch_num, PATTERN):
    mergedLists = pd.Series()
    for i in batch_num:
        p_file = SOURCE_DIR + "Batch" + str(i) +  PATTERN
        batch = pd.read_pickle(p_file)
        batch = pd.Series(batch)
        try:
            mergedLists = pd.concat([mergedLists, batch], axis=0, ignore_index = True)
        except (AttributeError, TypeError):
            mergedLists = pd.concat([mergedLists, batch], axis=0, ignore_index = True)
    mergedLists.reset_index(drop=True, inplace=True)
    return mergedLists



################################
###Lemmatize
################################

def lemmatize(doc, nlp):
    doc = doc.lower()
    doc = nlp(doc)
    txt = [token.lemma_ for token in doc]  #lemmatizes
    if len(txt) > 2:
        return ' '.join(txt)
    else:
        return ''

    
def extend_df(df, txt, ToSubsetDrop):
    dates = df.index
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, pd.DataFrame({ToSubsetDrop: txt})], axis=1)
    df.set_index(dates, inplace=True)
    #Remove again the ones that once cleaned ended up with a tiny corpus (<=2)
    df.dropna(subset=[ToSubsetDrop], inplace=True)
    return df

    

    
"""###############################"""
"""############Main code##########"""
"""###############################"""

###Define input and other variables

SOURCE_DIR = sys.argv[1] #Directory where the pickled files are stored
PATTERN = sys.argv[2] #The regex of the pickle files 
OUT_FILE_1 = sys.argv[3] #name of the first pickle output file, which has all the pickled files concatenated


PATTERN_sentence_list = "_sentence_list.p"
PATTERN_listOfDf_TokenizedSentenceNoStopWords = "_listOfDf_TokenizedSentenceNoStopWords.p"
PATTERN_countries_list = "_countries_list.p"
PATTERN_continents_list = "_continents_list.p"

nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
stemmer = SnowballStemmer("english")

############
##Main code:
############

listOfSelectedFiles = os.listdir(SOURCE_DIR)

df = pd.DataFrame()
for entry in listOfSelectedFiles:
    df_new = pd.read_pickle(entry)
    df = pd.concat([df, df_new], axis=0, ignore_index = True)

df.reset_index(drop=True, inplace=True)
pickle.dump( df, open(OUT_FILE_1, "wb") )



###########Append the pickled lists

#First concatenate the lists of the batches
batch_num = np.arange(1,83)

sentence_Series = putListsTogether(SOURCE_DIR, batch_num, PATTERN_sentence_list)

SeriesOfDf_TokenizedSentenceNoStopWords = putListsTogether(SOURCE_DIR, batch_num, PATTERN_listOfDf_TokenizedSentenceNoStopWords)

#I need to turn this series of dataframes into a series of lists, each list with a string of what was the list of the dataframe
mergedLists = []
for df in SeriesOfDf_TokenizedSentenceNoStopWords:
    try:
        df_abstract = df.filtered
        clean_text = []
        for sentence in df_abstract:
            try:
                
                sentence = [re.sub("[^A-Za-z']+", '', str(row)).lower() for row in sentence]  #Remove non-alphabetic characters
                sentence = ' '.join(sentence)
                clean_text.append(sentence)
            except (TypeError, AttributeError, ValueError):
                clean_text.append([])
        clean_text = walk(clean_text)
        clean_text = [re.sub(' +', ' ', str(row)) for row in clean_text]
        mergedLists.append(clean_text)
    except (TypeError, AttributeError, ValueError):
        mergedLists.append([])

mergedLists = pd.Series(mergedLists)
SeriesOfDf_TokenizedSentenceNoStopWords = mergedLists

countries_Series = putListsTogether(SOURCE_DIR, batch_num, PATTERN_countries_list)
countries_Series = countries_Series.apply(walk)
countries_Series = countries_Series.apply(walk)

continents_Series = putListsTogether(SOURCE_DIR, batch_num, PATTERN_continents_list)
continents_Series = continents_Series.apply(walk)

df_merged = pd.concat([df, sentence_Series, SeriesOfDf_TokenizedSentenceNoStopWords, countries_Series, continents_Series], axis=1, ignore_index = True)

df_merged.columns = ["Title", "Abstract", "Date", "sentence_list", "listOfDf_TokenizedSentenceNoStopWords", "countries_list", "continents_list" ]

pickle.dump( df_merged, open("FULLcolumns_ALLBatches.p", "wb") )


### Some further processing 
df = copy.copy(df_merged)
df.set_index(pd.to_datetime(df.Date), inplace=True)
df.drop(columns=['Date'], inplace=True)

#Remove the entries without abstract
df.dropna(subset=['Abstract'], inplace=True)

#I should turn the list of setences strs into a single string sentence for each entry
flatten = lambda x: " ".join(x)
df.listOfDf_TokenizedSentenceNoStopWords = df.listOfDf_TokenizedSentenceNoStopWords.apply(flatten)

pickle.dump( df, open("FULLcolumns_ALLBatches_noNaNAbstractsSingleStr.p", "wb") )  #So, this has no nan abstracts AND the abstract is a single string


#Lemmatize
txt = [lemmatize(doc, nlp) for doc in df.listOfDf_TokenizedSentenceNoStopWords]
df = extend_df(df, txt, 'Lemmatized')

##Save it
pickle.dump( df, open("ALLBatches_noNaN_joinedTitleAbstract_Lemma.p", "wb") )

