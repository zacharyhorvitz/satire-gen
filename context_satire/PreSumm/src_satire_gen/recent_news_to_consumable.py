


with open("recent_news_sample.txt","r",encoding="latin") as text_file:
	with open("recent_news_out.txt","w",encoding="latin") as outfile:
		data_lines = list(text_file.readlines())
		for line in data_lines:
			if len(line) == 0: continue
			line = line.replace("\n"," ")
			line = line.split()
			if len(line) > 512:
				line = line[:512]
			outfile.write(" ".join(line)+"\n")
