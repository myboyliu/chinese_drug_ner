import json
from data_utils import ssbs, load_data

def save_json_to_file(file_name,data):
    #保存dict to json file
    print('save .. ' + file_name)
    fp = open(file_name,"w")
    json.dump(data,fp)
    fp.close()

total_data, dfLabel = load_data()
train = ssbs( total_data, dfLabel, 500 )

words = set( [i for l in train['label'] for i in l] ) | set(["[CLS]", "[SEP]"])
category2id = {w: idx for idx, w in enumerate( words )}
save_json_to_file('cache/category2id.json',category2id)

