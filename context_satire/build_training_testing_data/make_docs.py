import json
import nltk
import matplotlib.pyplot as plt
import random
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

NUM_ALL = 3
NUM_CNN = 2
CAP_IT = 512
NUM_SENT = 4
TRAIN_SPLIT = 0.85 #.90 before #0.80 for 6K

random.seed(3)


DOC_DIR = "15K/merged_without_report2/documents/"
UNPROC_DIR = "15K/merged_without_report2/unprocessed_documents/"


import os

for folder in [DOC_DIR,UNPROC_DIR,UNPROC_DIR+"split_docs_train",UNPROC_DIR+"split_head_train",UNPROC_DIR+"split_docs_test",UNPROC_DIR+"split_head_test"]:
	if not os.path.exists(folder):
	    os.makedirs(folder)
     


import re
pattern = r'\[.*?\]'



documents = []
headlines = []

with open("15K/all_top_ranked_docs.json","r") as json_file:
    ranked = json.load(json_file)



for h,data in sorted(list(ranked.items())):
    #    print(h)
    all_top = data["all"]
    cnn_top = data["cnn"]

    document = list(set([body for _,body in all_top[:NUM_ALL]]+[body for _,body in all_top[:NUM_CNN]]))
    first_sentences = [" ".join(tokenizer.tokenize(re.sub(pattern, '', x))[:NUM_SENT]) for x in document  ]
    # to_use_doc = "\n\n".join(first_sentences).replace("\n","")
    to_use_doc = " ".join(first_sentences).replace("\n","").replace("                            "," ")


    if CAP_IT is not None:
        if len(to_use_doc.split())+len(h.split()) + 1 > CAP_IT:
            to_keep = CAP_IT - 1 - len(h.split())
            to_use_doc = " ".join(to_use_doc.split()[:to_keep])

    if len(to_use_doc.split()) > 0:
    	documents.append((to_use_doc,h))

random.shuffle(documents)

context_lengths = [len(x[0].split()) for x in documents]
average_context_length = sum(context_lengths)/len(documents)

plt.hist(context_lengths,bins=50)

with open(DOC_DIR+"viewable_data.txt","w+",encoding="UTF-8") as write_file:
    for doc,label in documents:
        write_file.write("\n\n" + label+"\n\n")
        write_file.write(doc)


# print(average_context_length)
# plt.show()

with open(DOC_DIR+"all_data.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents:
        write_file.write(doc+"\t"+label+"\n")

with open(DOC_DIR+"all_heads.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents:
        write_file.write(label+"\n")

train_data = int(len(documents)*TRAIN_SPLIT)

print("TRAIN:",train_data,"TEST:",len(documents)-train_data)

with open(DOC_DIR+"train_data.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents[:train_data]:
        write_file.write(doc+"\t"+label+"\n")

with open(DOC_DIR+"train_heads.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents[:train_data]:
        write_file.write(label+"\n")

with open(DOC_DIR+"test_data.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents[train_data:]:
        write_file.write(doc+"\t"+label+"\n")

with open(DOC_DIR+"test_heads.tsv","w+",encoding="UTF-8") as write_file:
    for doc,label in documents[train_data:]:
        write_file.write(label+"\n")


for i,(doc,label) in enumerate(documents[:train_data]):
    with open(UNPROC_DIR+"split_docs_train/"+str(i),"w+",encoding="UTF-8") as doc_file:
        doc_file.write(doc)
    with open(UNPROC_DIR+"split_head_train/"+str(i),"w+",encoding="UTF-8") as doc_file:
        doc_file.write(label)


for i,(doc,label) in enumerate(documents[train_data:]):
    with open(UNPROC_DIR+"split_docs_test/"+str(i),"w+",encoding="UTF-8") as doc_file:
        doc_file.write(doc)
    with open(UNPROC_DIR+"split_head_test/"+str(i),"w+",encoding="UTF-8") as doc_file:
        doc_file.write(label)








