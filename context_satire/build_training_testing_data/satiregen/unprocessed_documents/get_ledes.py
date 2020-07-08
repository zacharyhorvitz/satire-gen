import glob
import json
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 

import string
printable = set(string.printable)

test_heads = []
train_heads = []

# for file in sorted(list(glob.glob("split_head_test/*")),key=lambda x: int(x.split('\\')[-1])):
#     with open(file,"r",encoding="latin-1") as load_file:
#         test_heads.append(load_file.read())

# for file in sorted(list(glob.glob("split_head_train/*")),key=lambda x: int(x.split('\\')[-1])):
#     with open(file,"r",encoding="latin-1") as load_file:
#         train_heads.append(load_file.read())

with open("all_train_heads.txt","r",encoding="latin-1") as headline_list:
    train_heads = [x.replace('\n','') for x in headline_list.readlines()]

with open("../../../onion_to_data.json","r") as onion_file:
    onion_data = json.load(onion_file)



clean_string = lambda x: "".join([c for c in x if c in printable])
onion_data = {clean_string(k):v for k,v in onion_data.items()}

        


# for i,h in enumerate(test_heads):
#     print(h)
#     h = clean_string(h)

#     if h in onion_data:
#         # print(onion_data[h])
#         # print(onion_data[h]["text"])
#         sentences = tokenizer.tokenize("".join(onion_data[h]["text"]))
#         first_sentence_lede = sentences[0] if len(sentences[0]) > 30 else " ".join(sentences[0:2])
#         start_location = first_sentence_lede.index('—')+1 if '—' in first_sentence_lede else 0
#         first_sentence_lede = first_sentence_lede[start_location:]
#         print(first_sentence_lede)

#     else:
#         exit()

#     with open('split_head_test_lede/'+str(i),"w+",encoding="UTF-8") as file:
#         file.write(first_sentence_lede)



for i,h in enumerate(train_heads):
    print(h)
    h = clean_string(h)

    if h in onion_data:
        # print(onion_data[h])
        # print(onion_data[h]["text"])
        sentences = tokenizer.tokenize("".join(onion_data[h]["text"]))
        first_sentence_lede = sentences[0] if len(sentences[0]) > 30 else " ".join(sentences[0:2])
        start_location = first_sentence_lede.index('—')+1 if '—' in first_sentence_lede else 0
        first_sentence_lede = first_sentence_lede[start_location:]
        # print(first_sentence_lede)

    else:
        print(h)
        exit()
    # print(first_sentence_lede)

    with open('split_head_train_lede/'+str(i),"w+",encoding="UTF-8") as file:
        file.write(first_sentence_lede)



        # candidates = [k for k,v in onion_data.items() if h[:10] == k[:10]]
        # if len(candidates) == 0:
        #     candidates = [k for k,v in onion_data.items() if h[-10:] == k[-10:]]
        # if len(candidates) == 0:
        #     exit()


   # first_sentence_lede = tokenizer.tokenize(article)
   #  if len(first_sentence_lede) == 0: continue
   #  first_sentence_lede = first_sentence_lede[0] if len(first_sentence_lede[0]) > 30 else " ".join(first_sentence_lede[0:2])







# with open("all_test_ledes.txt","w",encoding="latin-1") as out_file:
#     for h in test_heads:
#         out_file.write(h+"\n")

# with open("all_train_heads.txt","w",encoding="latin-1") as out_file:
#     for h in train_heads:
#         out_file.write(h+"\n")
