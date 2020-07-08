import glob
import json


def core_to_sentences(json_from_stanford):
	sentences = [ [token["word"].lower() for token in s["tokens"]] for s in json_from_stanford["sentences"]]
	return sentences


def merge_json(head_file, doc_file):
	head_json = json.load(open(head_file,"r",encoding="UTF-8"))
	doc_json = json.load(open(doc_file,"r",encoding="UTF-8"))
	return  {"src":core_to_sentences(doc_json),"tgt":core_to_sentences(head_json)}

test_docs = sorted(list(glob.glob("test_docs/*")))
test_heads = sorted(list(glob.glob("test_heads/*")))
train_docs = sorted(list(glob.glob("train_docs/*")))
train_heads = sorted(list(glob.glob("train_heads/*")))

assert len(train_heads) == len(train_docs)
assert len(test_docs) == len(test_heads)

training_data = [merge_json(head,doc) for head,doc in zip(train_heads,train_docs)]
testing_data = [merge_json(head,doc) for head,doc in zip(test_heads,test_docs)]

# for i,data in enumerate(training_data):
# 	with open("TRAINING/{}".format(i),"w+",encoding="UTF-8") as data_file:
# 		json.dump(data,data_file)

# for i,data in enumerate(testing_data):
# 	with open("TESTING/{}".format(i),"w+",encoding="UTF-8") as data_file:
# 		json.dump(data,data_file)

with open("combined_data/training.json","w+",encoding="UTF-8") as data_file:
		json.dump(training_data,data_file)


with open("combined_data/testing.json","w+",encoding="UTF-8") as data_file:
		json.dump(testing_data,data_file)

