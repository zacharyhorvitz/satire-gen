import glob


test_heads = []
train_heads = []

for file in sorted(list(glob.glob("split_head_test/*"))):
	with open(file,"r",encoding="latin-1") as load_file:
		test_heads.append(load_file.read())

for file in sorted(list(glob.glob("split_head_train/*"))):
	with open(file,"r",encoding="latin-1") as load_file:
		train_heads.append(load_file.read())


with open("all_test_heads.txt","w",encoding="latin-1") as out_file:
	for h in test_heads:
		out_file.write(h+"\n")

with open("all_train_heads.txt","w",encoding="latin-1") as out_file:
	for h in train_heads:
		out_file.write(h+"\n")
