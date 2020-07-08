import glob
import json



test_docs = sorted(list(glob.glob("split_orig_test/*")))
test_heads = sorted(list(glob.glob("split_sat_test/*")))

with open("all_test_satire_unfun.txt","w+",encoding="UTF-8") as data_file:
		for h,doc in zip(test_heads,test_docs):
			# data_file.write(open(doc,encoding="UTF-8").read())
			# data_file.write("\n")
			data_file.write(open(h,encoding="UTF-8").read())
			data_file.write("\n")




