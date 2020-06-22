from Bio import Entrez
from urllib.error import HTTPError
import xml.etree.ElementTree as et
import pickle
import numpy as np
import pandas as pd


"""
############## Read the saved file
"""

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
rows_journalArticle = []
rows_review = []
rows_others = []
rows_unknownType = []

for node in xroot:
    try:
        types = []
        for child in node.find("MedlineCitation").find("Article").find("PublicationTypeList").iter("PublicationType"):
            types.append(child.attrib['UI'])
        if types.count("D016454") > 0: #If it's a review. D016454 is the UI of a review
            title, abstract, date = get_xml_info(node)
            rows_review.append({"Title": title, "Abstract": abstract, "Date": date})
        elif types.count("D016428") > 0: #If it's a Journal Article. D016428 is the UI of a journal article
            title, abstract, date = get_xml_info(node)
            rows_journalArticle.append({"Title": title, "Abstract": abstract, "Date": date})
        else:
            title, abstract, date = get_xml_info(node)
            rows_others.append({"Title": title, "Abstract": abstract, "Date": date})
    except AttributeError:
        title, abstract, date = get_xml_info(node)
        rows_unknownType.append({"Title": title, "Abstract": abstract, "Date": date})



df_journalArticle = pd.DataFrame(rows_journalArticle, columns=df_cols)

#Pickle the object just tu try it out for later safety usage
pickle.dump( df_journalArticle, open("AntibioPubmedSearch_dfBatch1.p", "wb") )

