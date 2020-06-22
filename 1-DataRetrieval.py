from Bio import Entrez
from urllib.error import HTTPError
import xml.etree.ElementTree as et
import sys

"""
#############Entrez part
"""

Entrez.email = sys.argv[1]

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

