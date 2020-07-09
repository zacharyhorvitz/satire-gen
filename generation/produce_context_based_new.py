import os
import re
import json

# Getting all the paths that we care about
all_files = os.listdir("../data/GEN_ALL_OUT_ENC_NEWS/")

CARE_ABOUT = [
    "1_.._.._.._PreSumm_models_ENC_12K_LR02_model_step_2000.pt",
    "1_.._.._.._PreSumm_models_ENC_10xMORE_DEC_12K_LR02_model_step_2000.pt",
    # "1_.._.._.._PreSumm_models_ENC_ABS_DEC_12K_LR02_model_step_151000.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_10xDEC_12K_LR02_model_step_2250.pt"
    ]

all_files = [p for p in all_files if p in CARE_ABOUT]

# Dicts of headlines to their paths
abs_dict, non_abs_dict = {}, {}

# Dicts of context to generated stuff
file_to_headlines = {f: [] for f in all_files}


def clean(headline):
    headline = headline.replace("....", ".").replace("|endoftext|", "").replace("<>", "").replace("[PAD]", "").strip()
    if headline[:len("News:")] == "News:":
        headline=headline[len("News:"):]
    headline = headline.replace("'''", "").replace("[unused2]", "").replace("[unused1]","").strip()

    return re.sub(r'[^\x00-\x7f]', r'', headline)


fail_count, success_count = 0, 0
for f_name in all_files:
    # seeing if we are dealing with - abstrract or normal generation
    if "ABS" in f_name: MODE = "ABS"
    else: MODE="NORMAL"
    # open the file
    with open("../data/GEN_ALL_OUT_ENC_NEWS/"+f_name, "r") as open_file:
        data = open_file.read()
        items = data.split("\n\n\n")
        for i, item in enumerate(items):
            if i == 0: continue
            components = item.split("[SEP]\n")
            try:
                generated_satire = clean(components[1])
            except Exception as e:
                try:
                    components = item.split("\n")
                    generated_satire = clean(components[1])
                    print(generated_satire)
                except Exception as e:
                    fail_count += 1
                    continue
            if MODE=="ABS": abs_dict[generated_satire] = f_name
            else: non_abs_dict[generated_satire] = f_name
            file_to_headlines[f_name].append(generated_satire)
            success_count += 1

    with open("../data/context_based_new/abs.json", "w") as abs_file:
        json.dump(abs_dict, abs_file)
    with open("../data/context_based_new/non_abs_dict.json", "w") as non_abs_file:
        json.dump(non_abs_dict, non_abs_file)
    with open("file_to_headlines_new.json", "w") as file_to_headlines_file:
        json.dump(file_to_headlines, file_to_headlines_file)
    print(fail_count, success_count)


oof, yee = 0, 0
for f in file_to_headlines:
    print(f)
    if len(file_to_headlines[f]) == 1955:
        yee += 1
    else:
        oof += 1
        print(len(file_to_headlines[f]))

print(oof, yee)
