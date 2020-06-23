
import sys
import pandas as pd
import numpy as np
import spacy
import en_core_web_md
import pickle
import string
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import eli5
from sklearn.pipeline import Pipeline


punctuations = string.punctuation
nlp = en_core_web_md.load()
parameters_LSV = {'C': np.arange(0.01,100,0.1) }


IN_FILE_1 = sys.argv[1] #the training set
IN_FILE_2 = sys.argv[2]
OUT_FILE_PREFIX = sys.argv[3]

MODEL_micro = "microbiology_LSV_model.pickle"
MODEL_epi = "epidemiology_LSV_model.pickle"
MODEL_amr = "amr_LSV_model.pickle"



"""#######################################"""
"""###########Defined functions###########"""
"""#######################################"""


def spacy_tokenizer(text):
    #print(text)
    tokens = nlp(text)
    tokens = [tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return (' '.join(tokens))

def evaluate_predictions(test_targets, predicted):
    print("Accuracy score: %s" % (accuracy_score(test_targets, predicted)) )
    print("Confusion matrix:\n %s" % (confusion_matrix(test_targets, predicted)) )
    print("Confusion matrix:\n %s" % (classification_report(test_targets, predicted)) )


def get_eli5_explanation(df, clsf, parameters, OUTPUT_NAME):
    train_data, test_data, train_targets, test_targets = train_test_split(df.text, df.label, test_size=0.2, stratify=df.label)
    #Apply the function that normalizes text to train and test datasets
    train_data = [spacy_tokenizer(text) for text in train_data]
    test_data = [spacy_tokenizer(text) for text in test_data]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    features = count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    gs_clf = GridSearchCV(clsf, parameters)
    gs_clf = gs_clf.fit(X_train_tfidf, train_targets)
    gs_clf = gs_clf.best_estimator_
    X_test_counts = count_vect.transform(test_data)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    preds = gs_clf.predict(X_test_tfidf)
    print(evaluate_predictions(test_targets, preds))
    a = eli5.show_weights(gs_clf, feature_names=features)
    html = a.data
    with open(OUTPUT_NAME+"_eli5.html", 'w') as f:
        f.write(html)
    #Pickle the pipeline
    text_clf = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', gs_clf)])
    pickle.dump(text_clf, open(OUTPUT_NAME+"_model.pickle", "wb"))



"""#################################"""
"""#############Main code###########"""
"""#################################"""


###Make the models

df = pd.read_excel(IN_FILE_1, index_col='Index', 
                   converters={'score_amr':str,'score_epidemiology':str})

df.dropna(inplace=True)

df2 = pd.concat([df, df.Title+" "+df.Abstract, 
                 df.amr+"_"+ df.score_amr, 
                 df.epidemiology+"_"+ df.score_epidemiology], 
                axis=1)
df2.columns = list(df.columns) + ["Title_Abstract", "AMR_topic", "Epidemiology_topic"] 

df2.drop(columns=["Title", "Abstract", "amr", "score_amr", 
                  "epidemiology", "score_epidemiology"], inplace=True)



###Classification Microbiology or nonMicrobiology

microbiology_df = df2[["Title_Abstract", "microbiology"]]
microbiology_df.columns = ["text", "label"]

get_eli5_explanation(microbiology_df,
                     LinearSVC(), parameters_LSV,
                     "microbiology_LSV")



###Classification Epidemiology or nonEpidemiology

epi_df = df2[["Title_Abstract", "Epidemiology_topic"]]
epi_df.columns = ["text", "label"]

get_eli5_explanation(epi_df,
                     LinearSVC(), parameters_LSV,
                     "epidemiology_LSV")



###Classification AMR or nonAMR

amr_df = df2[["Title_Abstract", "AMR_topic"]]
amr_df.columns = ["text", "label"]

amr_df = pd.concat([df2.Title_Abstract, df.amr], axis=1)
amr_df.columns = ["text", "label"]

get_eli5_explanation(amr_df,
                     LinearSVC(), parameters_LSV,
                     "amr_LSV")



##############Then apply the models successively until we have a set of microbiology epidemiological AMR reports. 

df = pd.read_pickle(IN_FILE_2)

##First filter out the nonMicrobiology ones

lsv_pickle = pickle.load(open(MODEL_micro, "rb"))
test_data = df.LemmatizedTitleAbstract
preds = lsv_pickle.predict(test_data)
#Keep only those ones that ARE microbiology
df.reset_index(level=0, inplace=True) #To turn the dates from the index into a column.
preds_micro = preds=="Microbiology"
df_micro = df.loc[preds_micro]

pickle.dump( df_micro, open(OUT_FILE_PREFIX + "_Micro.p", "wb") ) 


#Ok, now, keep the ones that are Epidemiological 

lsv_pickle = pickle.load(open(MODEL_epi, "rb"))
test_data = df_micro.LemmatizedTitleAbstract
preds = lsv_pickle.predict(test_data)
#Keep only those ones that ARE epidemiology
preds_epi = preds=="Epidemiological"
df_epi = df_micro.loc[preds_epi]

pickle.dump( df_epi, open(OUT_FILE_PREFIX + "_MicroEpi.p", "wb") ) 


#Ok, now filter keep only the AMR ones

lsv_pickle = pickle.load(open(MODEL_amr, "rb"))

test_data = df_epi.LemmatizedTitleAbstract
preds = lsv_pickle.predict(test_data)
#Keep only those ones that ARE AMR
preds_amr = preds=="AMR"
df_amr = df_epi.loc[preds_amr]

pickle.dump( df_amr, open(OUT_FILE_PREFIX + "_MicroEpiAMR.p", "wb") ) 



