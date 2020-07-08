with open("train_unfunsatire.txt","r",encoding="latin-1") as data_file:
		unfun_sat_train = [x.split("***") for x in data_file.readlines()]


with open("dev_unfunsatire.txt","r",encoding="latin-1") as data_file:
		unfun_sat_dev = [x.split("***") for x in data_file.readlines()]

for i,(orig,sat) in enumerate(unfun_sat_train):
	with open("split_orig_train/"+str(i),"w+",encoding="latin-1") as orig_file:
		orig_file.write(orig)
	with open("split_sat_train/"+str(i),"w+",encoding="latin-1") as sat_file:
		sat_file.write(sat)

for i,(orig,sat) in enumerate(unfun_sat_dev):
	with open("split_orig_test/"+str(i),"w+",encoding="latin-1") as orig_file:
		orig_file.write(orig)
	with open("split_sat_test/"+str(i),"w+",encoding="latin-1") as sat_file:
		sat_file.write(sat)

	
