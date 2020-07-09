import csv
import json
import re
import random

################################ PATHS ########################################

# ALL PREVIOUS
NEW_MTURK_RESULT_FILE = "mturk_data/retrieved_from_mturk/result3-new.csv"
ALL_PREVIOUS_MTURK_RESULT_FILES = ["mturk_data/retrieved_from_mturk/result0.csv",
    "mturk_data/retrieved_from_mturk/result1.csv",
    "mturk_data/retrieved_from_mturk/result2.csv"] + [NEW_MTURK_RESULT_FILE]
ALL_MTURKED_PATH = "all_mturked_previous_test.json"

# ALL THE HEADLINE TO FILE FILES
HEADLINE_TO_FILE_PATH = "mturk_data/headline_to_file.json"
NEW_HEADLINE_TO_FILE_PATH = "mturk_data/headline_to_file_new.json"
# ALL THE FILE TO HEADLINES FILES
FILE_TO_HEADLINES_PATH = "../generation_code/file_to_headlines.json"
FILE_TO_HEADLINES_NEW_PATH = "../generation_code/file_to_headlines_new.json"
# All the paths - CONTEXTFREE - SATIRE
CONTEXTFREE_PATH = "../data/context_free_k1_p0/satire1.json" # 1660
CONTEXTFREE_NEW_PATH = "../data/context_free_k1_p0_news_new_gpt2/satire-gpt2-large-decoder.json" # 1660
# All the paths - CONTEXTBASED - ABS
CONTEXTBASED_ABS_PATH = "../data/context_based/abs.json" # 1955
CONTEXTBASED_NEW_PATH = "../data/context_based_new/abs.json" # 1955
# All the paths - CONTEXTBASED - NONABS
CONTEXTBASED_PATH = "../data/context_based/non_abs_dict.json" # 1955
CONTEXTBASED_NEW_PATH = "../data/context_based_new/non_abs_dict.json" # 1955
# All the paths - CONTEXTFREE - NEWS
NEWS_PATH = "../data/context_free_k1_p0/news1.json" # 1660
NEWS_NEW_PATH = "../data/context_free_k1_p0_news_new_gpt2/news-gpt2-large-decoder.json" # 1660
# All the paths - ONION
ONION_TEST_PATH = "../data/all_test_heads.txt" # 1955
# All the information about loading a new thing from an old model
PREVIOUS_INDICES_PATH = "chosen_indices_with_new_old_model.json"
PREVIOUS_NEW_OLD_MODEL_GENERATIONS_PATH = "new_headlines_new_old_model.json"


################################ LIST OF PATHS ################################

# LIST OF INFORMATION THAT WE CARE ABOUT
CARE_ABOUT_OLD = [
    "1_.._.._.._PreSumm_models_ENC_12K_LR02_model_step_2000.pt",
    "1_.._.._.._PreSumm_models_ENC_10xMORE_DEC_12K_LR02_model_step_2000.pt",
    # "1_.._.._.._PreSumm_models_ENC_ABS_DEC_12K_LR02_model_step_151000.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_10xDEC_12K_LR02_model_step_2250.pt",
    "../data/data_gen_out_4_5_20/1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt",
    NEWS_PATH,
    ONION_TEST_PATH,
    CONTEXTFREE_PATH
]

OLD_TEST_NEW_MODEL_PATH = "../data/data_gen_out_4_5_20/1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt"

CARE_ABOUT_NEW = [
    "1_.._.._.._PreSumm_models_ENC_12K_LR02_model_step_2000.pt",
    "1_.._.._.._PreSumm_models_ENC_10xMORE_DEC_12K_LR02_model_step_2000.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_10xDEC_12K_LR02_model_step_2250.pt",
    "1_.._.._.._PreSumm_models_ENC_ABS_100MORE_DEC_12K_LR02_model_step_3000.pt",
    NEWS_NEW_PATH,
    ONION_TEST_PATH,
    CONTEXTFREE_NEW_PATH
]

################################ DICTIONARIES ################################
# loading the indices of the contexts that we are controlling for
with open(PREVIOUS_INDICES_PATH, "r") as previous_indices_file:
    PREVIOUS_INDICES = list(json.load(previous_indices_file))

# file to headlines for the previous test set
with open(FILE_TO_HEADLINES_PATH, "r") as old_file_to_headlines_file:
    old_file_to_headlines = json.load(old_file_to_headlines_file)

# file to headlines for the new test set (sad coronavirus) + a pipe of old samples
# from the newly added model
with open(FILE_TO_HEADLINES_NEW_PATH, "r")  as new_file_to_headlines_file:
    new_file_to_headlines = json.load(new_file_to_headlines_file)

# head to files of previous test set
with open(HEADLINE_TO_FILE_PATH, "r") as old_head_to_files_file:
    old_head_to_files = json.load(old_head_to_files_file)

# head to files of new test set
with open(NEW_HEADLINE_TO_FILE_PATH, "r") as new_head_to_files_file:
    new_head_to_files = json.load(new_head_to_files_file)

############################ MUTABLE DATA STRUCTS ############################

master_dictionary = {}
headline_to_dump = {} # This would
PATH_TO_SPECIAL = {}
PATH_TO_MAJORITY_FUNNY = {}

############################## HELPER FUNCTIONS ##############################

def process_row(row: list, value_to_index_dict: dict, care_about_head_to_file: dict):
    """
    heads_domain = list of headlines
    """
    headlines = []
    for i in range(10):
        try:
            headline = row[value_to_index_dict["Input.input_"+str(i)]]
            coherence = int(row[value_to_index_dict["Answer.q"+str(i)+"0"]])
            onion = int(row[value_to_index_dict["Answer.q"+str(i)+"1"]])
            funny = int(row[value_to_index_dict["Answer.q"+str(i)+"2"]])
            if clean(headline) in care_about_head_to_file:
                add_to_master(headline, coherence, onion, funny)
                headlines.append(headline)
        except Exception as e:
            print("Empty submission found: ")
            print(row[value_to_index_dict["HITId"]])
            print(row[value_to_index_dict["WorkerId"]])
            if row[value_to_index_dict["AssignmentStatus"]] != 'Rejected':
                raise Exception
            break
    return headlines


def get_value_to_index_dict(first_row: list):
    value_to_index = {}
    for i, ele in enumerate(first_row):
        value_to_index[ele] = i
    return value_to_index


def add_to_master(head, coherence, onion, funny):
    # If head is not already in master_dictionary
    if head not in master_dictionary:
        new_dict = {
            "coherence": [coherence],
            "onion": [onion],
            "funny": [funny]
        }
        master_dictionary[head] = new_dict
    else:
        existing_dict = master_dictionary[head]
        existing_dict["coherence"].append(coherence)
        existing_dict["onion"].append(onion)
        existing_dict["funny"].append(funny)


# Helper to clean headline
def clean(headline):
    headline = headline.replace("....", ".").replace("|endoftext|", "").replace("<>", "").strip()
    if headline[:len("News:")] == "News:":
        headline=headline[len("News:"):]
    headline = headline.replace("'''", "").replace("[unused2]", "").replace("[unused1]", "").strip()

    return re.sub(r'[^\x00-\x7f]', r'', headline)


def add_to_aggregate(aggregate_dict, path, results):
    path_data = aggregate_dict[path]
    for feature in ["coherence", "onion", "funny"]:
        path_data[feature].append(results[feature])


def aggregate_to_models(care_about, headline_to_file, SPECIFIC=1):
    ### THE RETURN OF THIS: a dictionary
    aggregate_dict = {}
    seen = []
    PATHS = care_about
    for p in PATHS:
        data = {
            "coherence": [],
            "onion": [],
            "funny": []
        }
        aggregate_dict[p] = data
    # Looping through and adding to the aggregate dictionary
    for head in master_dictionary:
        results = master_dictionary[head]
        coherence, onion, funny = results["coherence"][:3], results["onion"][:3], results["funny"][:3]
        all_coherent = coherence == [1,1,1]
        all_oniony = onion == [1,1,1]
        majority_funny = len([f for f in funny if f == 3]) >= 2
        special = all_coherent and all_oniony and majority_funny
        if type(headline_to_file[head]) == str: paths = [headline_to_file[head]]
        else: paths = headline_to_file[head]
        for path in paths:
        # if SPECIFIC and path == CONTEXTBASED_PATH:
        #     path = realcontextbased_info[head]
        # if SPECIFIC and path == CONTEXTBASED_ABS_PATH:
        #     path = realcontextbased_abs_info[head]
            if path in PATHS and (head, path) not in seen:
                add_to_aggregate(aggregate_dict, path, results)
                seen.append((head, path))
                if special:
                    if path not in PATH_TO_SPECIAL:
                        PATH_TO_SPECIAL[path] = [head]
                    else:
                        PATH_TO_SPECIAL[path].append(head)
                if majority_funny:
                    if path not in PATH_TO_MAJORITY_FUNNY:
                        PATH_TO_MAJORITY_FUNNY[path] = [head]
                    else:
                        PATH_TO_MAJORITY_FUNNY[path].append(head)
    return aggregate_dict


def analyze_aggregated(aggregated_dict: dict):
    for path in aggregated_dict:
        overlap = 0
        skipped, coherence_agreement, coherence_majority = 0, 0, 0
        majority_onion, majority_funny, majority_couldfunny_or_mais = 0, 0, 0
        coherence = aggregated_dict[path]['coherence']
        onion = aggregated_dict[path]['onion']
        funny = aggregated_dict[path]['funny']
        assert len(coherence) == len(onion)
        assert len(coherence) == len(funny)
        for i, coherencevals in enumerate(coherence):
            if len(coherencevals) > 3:
                overlap += 1
                coherencevals = coherencevals[:3]
                this_onion = onion[i][:3]
                this_funny = funny[i][:3]
            if len([t for t in coherencevals if t]) >= 2:
                coherence_majority += 1
                if coherencevals == [1,1,1]:
                    coherence_agreement += 1
                    this_onion = onion[i]
                    this_funny = funny[i]
                else:
                    this_onion, this_funny = [], []
                    for id, coherencevals_rate in enumerate(coherencevals):
                        if coherencevals_rate == 1:
                            this_onion.append(onion[i][id])
                            this_funny.append(funny[i][id])
                    # print(this_onion, this_funny)
                    assert len(this_onion) == 2
                    assert len(this_funny) == 2
            else:
                skipped += 1
                continue
            if len([t for t in this_onion if t == 1]) >= 2: majority_onion += 1
            if len([t for t in this_funny if t == 3]) >= 2: majority_funny += 1
            if len([t for t in this_funny if t >= 2]) >= 2: majority_couldfunny_or_mais += 1
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print(path)
        print("Skipped: ", skipped, ", Coherence agreement (3): ", coherence_agreement, ", Coherence majority (2+): ", coherence_majority)
        print("Overlapped: ", overlap)
        if coherence_agreement != 0:
            print("Percentage skipped: ", skipped/len(coherence))
            print("Percentage all agreed on coherence: ", coherence_agreement/len(coherence))
            print("Percentage 2+ vote on coherence: ", coherence_majority/len(coherence))
            print("-=-=-=-=-=")
            print("Majority onion: ", majority_onion, ", Percentage: ", majority_onion/coherence_majority)
            print("Majority funny: ", majority_funny, ", Percentage: ", majority_funny/coherence_majority)
            print("majority funny, or can see how someone can see it: ", majority_couldfunny_or_mais, ", Percentage: ", majority_couldfunny_or_mais/coherence_majority)
        print("\n\n")
        print("Special snowflakes: (all coherent, all oniony, majority funny) -=-=-=-=-=-=-=-=-=")
        if path in PATH_TO_SPECIAL:
            for h in list(set(PATH_TO_SPECIAL[path])):
                print(h)
        print("\n\n")
        print("Majority funny ones: -=-=-=-=-=-=-=-=-=")
        if path in PATH_TO_MAJORITY_FUNNY:
            for h in list(set(PATH_TO_MAJORITY_FUNNY[path])):
                print(h)
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print("\n\n\n")


########################### CALCULATING OLD TEST ###########################
################################## STATS ###################################

OLD_MODELS_TO_HEADLINES = {f: [] for f in CARE_ABOUT_OLD}

def process_mturk_data_file(mturk_data_path, care_about_head_to_file):
    with open(mturk_data_path, "r") as results_file:
        csv_reader = list(csv.reader(results_file))
        headlines_to_dump = {}
        for i, row in enumerate(csv_reader):
            if i == 0:
                value_to_index = get_value_to_index_dict(row)
                continue
            # For each, we will process the row
            headlines = process_row(row, value_to_index, care_about_head_to_file)
            for h in headlines:
                   headlines_to_dump[h] = 1


def get_old_test_stuff():
    # Getting the list of all the things already put on mturk
    with open(ALL_MTURKED_PATH, "r") as previousmturked_file:
        previousmturked = json.load(previousmturked_file)
    # First, getting the chosen indices, and then loading the appropriate
    # sentences for each context that we care about
    file_to_headlines = {}
    # Load the new headlines from the new model from the old test set
    with open(PREVIOUS_NEW_OLD_MODEL_GENERATIONS_PATH, "r") as prevgenfile:
        file_to_headlines[OLD_TEST_NEW_MODEL_PATH] = list(json.load(prevgenfile))[:270]
        len_pulled = len(file_to_headlines[OLD_TEST_NEW_MODEL_PATH])
    # Load the previous indices
    with open(PREVIOUS_INDICES_PATH, "r") as previndifile:
        previous_indices = list(json.load(previndifile))[:len_pulled]
    # Load everything in the CARE_ABOUT_OLD TO THIS
    heads_domain = []
    for f in CARE_ABOUT_OLD:
        final_heads = []
        if f in file_to_headlines:
            continue
        if ".pt" in f and f not in file_to_headlines:
            # Extracting all the headlines associated with it from the file_to_headlines
            assert f in old_file_to_headlines
            associated_headlines = old_file_to_headlines[f]
            for index in previous_indices:
                final_heads.append(clean(associated_headlines[index]))
            print("yee")
        elif f == NEWS_PATH or f == CONTEXTFREE_PATH:
            final_heads = [clean(h) for h in old_head_to_files if (type(old_head_to_files[h]) == list and f in old_head_to_files[h]) or (type(old_head_to_files[h]) == str and f == old_head_to_files[h])]
            final_heads = [h for h in final_heads if h in previousmturked]
            random.shuffle(list(set(final_heads)))
            final_heads = final_heads[:len_pulled]
            print("yeee")
        elif f == ONION_TEST_PATH:
            with open(f, "r") as onionfile:
                for line in onionfile:
                    final_heads.append(clean(line))
            final_heads = [h for h in final_heads if h in previousmturked]
            random.shuffle(list(set(final_heads)))
            final_heads = final_heads[:len_pulled]
            print("yeeee")
        else:
            print(f)
            raise Exception
        assert len(final_heads) == len_pulled
        file_to_headlines[f] = final_heads
        heads_domain.extend(final_heads)
        heads_domain = list(set(heads_domain))
    return file_to_headlines, heads_domain


def process_old_test_stuff(old_test_info, heads_domain):
    care_about_head_to_file = {}
    # Getting the care about head to file that we care about
    for f in old_test_info:
        headlines = old_test_info[f]
        for h in headlines:
            if h not in care_about_head_to_file: care_about_head_to_file[h] = [f]
            else: care_about_head_to_file[h].append(f)
    # For each of the processed mturk file, we would process
    for mturk_path in ALL_PREVIOUS_MTURK_RESULT_FILES:
        process_mturk_data_file(mturk_path, care_about_head_to_file)
    # After this stage, we have already populated the master dictionary, so we
    # will run the analysis
    aggregate_dict = aggregate_to_models(CARE_ABOUT_OLD, care_about_head_to_file)
    analyze_aggregated(aggregate_dict)


def process_new_test_stuff():
    # Goal: To get the new path that made
    assert len(master_dictionary) == 0
    with open(NEW_HEADLINE_TO_FILE_PATH, "r") as new_headline_to_file_file:
        care_about_headline_to_file = json.load(new_headline_to_file_file)
    for h in care_about_headline_to_file:
        care_about_headline_to_file[h] = [f for f in care_about_headline_to_file[h] if f in CARE_ABOUT_NEW]
    process_mturk_data_file(NEW_MTURK_RESULT_FILE, care_about_headline_to_file)
    aggregate_dict = aggregate_to_models(CARE_ABOUT_NEW, care_about_headline_to_file)
    analyze_aggregated(aggregate_dict)



def clear_meta_dict():
    master_dictionary = {}


if __name__ == "__main__":
    old_test_info, heads_domain = get_old_test_stuff()
    # print(len(old_test_info[OLD_TEST_NEW_MODEL_PATH]))
    process_old_test_stuff(old_test_info, heads_domain)
    # process_new_test_stuff()






# # Helper function
#
# for f in ALL_MTURK_RESULT_FILES:


# # After having finished processing each entry, now aggregate to models
# aggregate_dict = aggregate_to_models()
# # print(aggregate_dict)
# analyze_aggregated(aggregate_dict)
# # Dump headlines out to a dict
# # with open("processed.json", "w") as processed_file:
#     # json.dump(headlines_to_dump, processed_file)
# print(len(headlines_to_dump))
