import csv
import json
import random
import re
import spacy

# Loading spacy language package yeehaw
nlp = spacy.load("en_core_web_lg")

# All the paths
CONTEXTFREE_PATH = "../data/context_free_k1_p0_news_new_gpt2/satire-gpt2-large-decoder.json" # 1660
CONTEXTBASED_ABS_PATH = "../data/context_based_new/abs.json" # 1955
CONTEXTBASED_PATH = "../data/context_based_new/non_abs_dict.json" # 1955
NEWS_PATH = "../data/context_free_k1_p0_news_new_gpt2/news-gpt2-large-decoder.json" # 1660
ONION_TEST_PATH = "../data/all_test_heads.txt" # 1955
# NEW THING THAT WE ARE ADDING INTO OUR MODEL
NEW_OLD_HEADLINES_PATH = "new_headlines_new_old_model.json"
NEW_MODEL_NAME = "../data/data_gen_out_4_5_20/1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt"

with open(NEW_OLD_HEADLINES_PATH, "r") as new_old_headlines_file:
    new_old_headlines = list(json.load(new_old_headlines_file))[:270] # Limiting it to 270 - $76.5

JUICE_POSITION_DICT = {
    CONTEXTFREE_PATH: 1,
    NEWS_PATH: 1
}

EXPORT_PATH = "mturk_data.csv"

care_about = [
    "1_.._.._.._PreSumm_models_ENC_12K_LR02_model_step_2000.pt",
    "1_.._.._.._PreSumm_models_ENC_10xMORE_DEC_12K_LR02_model_step_2000.pt",
    # "1_.._.._.._PreSumm_models_ENC_ABS_DEC_12K_LR02_model_step_151000.pt", # Omitted out
    "1_.._.._.._PreSumm_models_ENC_ABS_10xDEC_12K_LR02_model_step_2250.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt",
    NEWS_PATH,
    CONTEXTFREE_PATH
]

# The information about the past/current generation
AMOUNT_PRODUCING = 750
STARTING = 0
NUM_MODELS = 6 # (previous was only 6)
PER_MODEL = int(AMOUNT_PRODUCING/NUM_MODELS)

# modified this for the
HEADLINE_TO_FILE_PATH = "mturk_data/headline_to_file_new.json"
headline_to_file = {}

PREVIOUSLY_PROCESSED_PATH = "mturk_data/headline_to_file.json"
with open(PREVIOUSLY_PROCESSED_PATH, "r") as previously_processed_file:
    previously_mturked = json.load(previously_processed_file)

FILE_TO_HEADLINES_PATH = "../generation_code/file_to_headlines_new.json"
with open(FILE_TO_HEADLINES_PATH, "r") as file_to_headlines_file:
    file_to_headlines = json.load(file_to_headlines_file)
    all_file_to_headlines_paths = [k[0] for k in file_to_headlines.items()]
    for p in all_file_to_headlines_paths:
        assert p in file_to_headlines
        if p not in care_about:
            del file_to_headlines[p]
            print("Deleting " + p + " from existing dict")



ALL_HEADLINES = []


# DECLARING HELPER FUNCTIONS

# Helper function to help clean the headlines
def clean(headline):
    headline = headline.replace("....", ".").replace("|endoftext|", "").replace("<>", "").strip()
    if headline[:len("News:")] == "News:":
        headline=headline[len("News:"):]
    headline = headline.replace("'''", "").replace("[unused2]", "").replace("[unused1]", "").strip()

    return re.sub(r'[^\x00-\x7f]', r'', headline)

# Helper function to help filter out the nasty ones
def filterer(headline, old_heads, new_heads):
    if headline in old_heads or headline in new_heads: return False
    doc = nlp(headline)
    return len(doc) >= 3

# Helper function to produce the csv file to upload onto mturk
def produce_csv(all_headlines, mode="w"):
    headers = ["input_"+str(i) for i in range(10)]
    with open(EXPORT_PATH, mode) as mturk_file:
        writer = csv.writer(mturk_file)
        # Write the first row
        writer.writerow(headers)
        for i, head in enumerate(all_headlines):
            # If it's time to move on to a new row
            if i%10 == 0:
                if i != 0:
                    assert(len(new_row) == 10)
                    writer.writerow(new_row)
                new_row = []
            # For all cases, you would append something new to the row
            new_row.append(head)
        assert len(new_row) == 10
        writer.writerow(new_row)
    return True


# Helper function to keep updating the dictionary of headlines that have already
# been considered
def update_id(new_head_to_file):
    try:
        with open(HEADLINE_TO_FILE_PATH, "r") as existing_file:
            existing_head_to_file = json.load(existing_file)
        # Updating the json
        for head in new_head_to_file:
            existing_head_to_file[head] = new_head_to_file[head]
        # Writing the file
        with open(HEADLINE_TO_FILE_PATH, "w") as write_file:
            json.dump(existing_head_to_file, write_file)
        return True
    except Exception as e:
        return False


with open(HEADLINE_TO_FILE_PATH, "r") as old_heads_file:
    old_heads = json.load(old_heads_file)



# Adding satire and news to the file_to_headlines dictionary
print("Adding satire and news to the file_to_headlines dictionary...")
for path, pos in JUICE_POSITION_DICT.items():
    print(path)
    filtered_out, not_set = 0, 0
    all = 0
    with open(path, "r") as open_file:
        # Sorting by headline
        sorted_loaded = list(json.load(open_file).items())
        print("Started out with: ", len(sorted_loaded))
        random.shuffle(sorted_loaded)
        headlines_from_this_model = []
        for key, value in sorted_loaded:
            # Getting the headlines
            if path == CONTEXTFREE_PATH or path == NEWS_PATH:
                headlines = value
            else:
                headlines = [key]
            # Clear everything from the headlin and choose only those that have length > 3
            all += len(headlines)
            headlines = [clean(h) for h in headlines if filterer(clean(h), old_heads, headline_to_file)]
            filtered_out += len([clean(h) for h in headlines if not filterer(clean(h), old_heads, headline_to_file)])
            not_set += len(headlines)
            # Putting it in all the headlines that we are outputting to mturk
            headlines_from_this_model.extend(headlines)
            headlines_from_this_model = list(set(headlines_from_this_model)) # making it to be distinct

        # Adding it to the file_to_headlines dict
        file_to_headlines[path] = headlines_from_this_model
        print("filtered out: ", filtered_out, "remaining: ", len(headlines_from_this_model), "we wish: ", not_set, "even better: ", all)

# Adding onion headlines stuff to this
# print(ONION_TEST_PATH)
# with open(ONION_TEST_PATH, "r") as onion_file:
#     onion_headlines = []
#     for line in onion_file:
#         cleaned = clean(line)
#         if filterer(cleaned, old_heads, headline_to_file) and clean(line) not in previously_mturked:
#             onion_headlines.append(clean(line))
#     file_to_headlines[ONION_TEST_PATH] = onion_headlines

# Checking error
if len(file_to_headlines) != 6:
    for f in file_to_headlines:
        print(f)
assert len(file_to_headlines) == 6
for fn in file_to_headlines:
    assert fn in care_about


print("Done adding... Ending with " + str(len(file_to_headlines)) + " models")

print("Selecting " + str(PER_MODEL) + " headlines from each model...")
# Getting the lengths of the number of models for each model
lengths = [len(file_to_headlines[h]) for h in file_to_headlines]
random_indices = random.sample(range(min(lengths)), PER_MODEL)
print("Random indices: ", random_indices[:10], "...")

def check_index(ind):
    for model_path in file_to_headlines:
        this_model_headlines = file_to_headlines[model_path]
        this_model_headlines = [clean(h) for h in this_model_headlines]
        if this_model_headlines[ind] in old_heads:
            return False
    return True

print("Fine tuning random indices...")
for i, indi in enumerate(random_indices):
    while not check_index(indi):
        indi = random.randint(0, min(lengths) - 1)
    random_indices[i] = indi

print("Selecting...")
for model_path in file_to_headlines:
    print(model_path)
    this_model_headlines = file_to_headlines[model_path]
    this_model_headlines = [clean(h) for h in this_model_headlines]
    for i, index in enumerate(random_indices):
        if this_model_headlines[index] not in old_heads:
            ALL_HEADLINES.append(this_model_headlines[index])
            if this_model_headlines[index] not in headline_to_file:
                headline_to_file[this_model_headlines[index]] = [model_path]
            else:
                headline_to_file[this_model_headlines[index]].append(model_path)

print("Done selecting.")
print("Exporting...")

print("Adding the previous model for the new thing that we care about...")
ALL_HEADLINES.extend(new_old_headlines)
for head in new_old_headlines:
    headline_to_file[head] = [NEW_MODEL_NAME]

# Shuffling AL_HEADLINES
random.shuffle(ALL_HEADLINES)
# print
print(len(ALL_HEADLINES))
print(len(headline_to_file))

# for h in ALL_HEADLINES:
#     print(h)


if len(ALL_HEADLINES) == AMOUNT_PRODUCING + 270:
    produce_csv(ALL_HEADLINES)
    update_id(headline_to_file)
    print("yeehaw")
else:
    print("Not enough :(")
