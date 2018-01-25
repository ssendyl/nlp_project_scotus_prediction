import pandas as pd
import re
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords

def process(chk,i):
	pat = re.compile(r'\d+\s+U\.S.\s+\d+', re.MULTILINE|re.UNICODE)
	val = []
	chk = chk.drop_duplicates(subset='html',keep='first')
	not_nan = chk[pd.notnull(chk['html'])]
	for i,c in enumerate(not_nan['html']): 
		try:                                                     
	   		patmatch = re.findall(pat,c)[0]
	    	val.append(patmatch)
	   	except: 
	   		val.append(None)

    not_nan['US_CITE'] = val
    not_nan = not_nan[pd.notnull(not_nan['US_CITE'])]
    
    gs = pd.read_csv('gold_label_data.csv')
    gs['US_CITE'] = gs['usCite']
    result = pd.merge(not_nan,gs,how='left',on='US_CITE')

    result = result.drop(result.columns[[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26]], axis=1)

    html_cleaner = lambda x: bs(x).text
	result['html'] = result['html'].apply(html_cleaner)
	# column you are working on
	html_df = result['html']
	stopword_set = set(stopwords.words("english"))
	# convert to lower case and split 
	html_df = html_df.str.lower()
	splitter = lambda x: x.split()
	html_df.apply(splitter)
	# remove stopwords
	html_df_list = html_df.tolist()
	html_df_wo_stops = [w for w in html_df_list if not w in stopwords.words("english")]
	# keep only words
	html_only_words = [re.sub(r'[^a-zA-Z]',' ', i) for i in html_df_wo_stops]
	# join the cleaned words in a list
	html_df = pd.DataFrame({"processed_html": html_only_words})
	html_df.replace("\n", " ").replace('\r', '')
	result['html'] = html_df

    result.to_csv(str(i)+'-proc.csv')

chunk_size = 10**4
#temp_chunk = None

i = 0
for chunk in pd.read_csv('scotus_file',chunksize=chunk_size):
	process(chunk, i)
	i += 1
	#temp_chunk = chunk
	#break




