from lexical_diversity import lex_div as ld
import glob
import numpy as np	    
import json	  
import nltk
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer	  
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')	

nlp = spacy.load("en_core_web_sm")
nlp_big = spacy.load("en_core_web_lg")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS	
from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
import sys	   
from presumm_tokenizer import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
sep_vid = tokenizer.vocab['[SEP]']
cls_vid = tokenizer.vocab['[CLS]']

ps = PorterStemmer()	 

def word_tokenize(text):
	text = text.strip().lower()
	return tokenizer.tokenize(text)


def jaccard(x,y,head_dict,context_dict):	
	x_elements = context_dict[x] 
	y_elements = head_dict[y]  
	return len(x_elements.intersection(y_elements))/(len(x_elements.union(y_elements))+0.0)	   

def compute_doc_relatedness(output_lines):
	context_head = []
	for i,l in enumerate(output_lines):
			if 'CONTEXT: ' in l:
				if i+1 < len(output_lines):
						context_head.append((l[8:].replace('[CLS]','').replace('[SEP]',"")," ".join([x for x in output_lines[i+1].split() if "[" not in x])))

	head_dict = {}
	context_dict = {}

	for i,(c,h) in enumerate(context_head):
		head_dict[h] = set([ps.stem(w).lower() for w in word_tokenize(h) if w not in spacy_stopwords])
		context_dict[c] = set([ps.stem(w).lower() for w in word_tokenize(c) if w not in spacy_stopwords])

	avg_jacc = [np.mean([jaccard(context,head,head_dict,context_dict) for context,_ in context_head]) for _,head in tqdm(context_head)]

	jac_score = [jaccard(context,head,head_dict,context_dict) for context,head in tqdm(context_head)]
	# print([(x,y) for x,y in zip(jac_score,avg_jacc)])
	normalized_jac = [x/y if y > 0 else 0 for x,y in zip(jac_score,avg_jacc)]

	return np.mean(normalized_jac),None

if __name__ == '__main__':
	files = "../../data/generations/*" 
	for file in glob.glob(files): #	
		print("\n",file)
		with open(file,'r',encoding='latin-1') as text_output:
			lines = list(text_output.readlines())
		lines = lines
		j_sim, s_sim = compute_doc_relatedness(lines)

		print("Avg. Relevance: \n\t Jacc.: {}".format(j_sim))
		


		





