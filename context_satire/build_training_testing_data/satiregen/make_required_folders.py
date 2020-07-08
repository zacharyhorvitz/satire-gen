import glob
import json



import os

for folder in ["processed_docs/train_heads","processed_docs/train_docs","processed_docs/test_heads","processed_docs/test_docs"]:
	if not os.path.exists(folder):
	    os.makedirs(folder)
   

  
