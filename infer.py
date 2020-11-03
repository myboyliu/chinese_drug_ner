#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf

from bert import tokenization
from model import entity_model
from conf import Config
from tensorflow.contrib.crf import viterbi_decode

from data_utils import load_testData, ssbsTest


def decode( logits, lengths, matrix, args ):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * args.relation_num + [0]])
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)
        
        paths.append(path[1:])

    return paths


def loadModel(args, mode):
    
    # 读取模型
    tf.reset_default_graph()
    session = tf.Session()
    model = entity_model(args)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
    path = os.path.join( args.model_dir, "{}".format(mode) )
    ckpt = tf.train.get_checkpoint_state( path )
    saver.restore(session, ckpt.model_checkpoint_path)
    
    return model, session


def read_json_from_file(file_name):
    #read json file
    print(file_name)
    print('load .. '+file_name)
    fp = open(file_name, "rb")
    data = json.load(fp)
    fp.close()
    return data


if __name__=='__main__':
        
        
    args = Config()
    
    category2id = read_json_from_file('cache/category2id.json')
    id2category = {j:i for i,j in category2id.items()}
    args.relation_num = len(category2id)
    args.category2id = category2id
    
    # model 可选择用第几折结果提交
    model, session = loadModel(args, mode=1) # 
    
    
    tokenizer = tokenization.FullTokenizer( vocab_file=args.vocab_file, do_lower_case=False)
    
    test = load_testData()
    test_data = ssbsTest( test, args.max_x_length )
    
    
    from utils import process
    
    for iid in tqdm( set(test_data['id']) ):
        
        sample = test_data[test_data['id'] == str(iid)]
        pred_list = []
        num = 1
        for d in sample.iterrows():
            d = d[1]
            tokens,t1, t2 = process( d['tx'], args, tokenizer )
            feed_dict = {
                model.input_x_word: [t1],
                model.input_mask: [t2],
                model.input_x_len: [ len(d['tx'])+2 ],
                model.keep_prob: 1,
                model.is_training: False,
            }
            
            lengths, logits, trans = session.run(
                fetches=[model.lengths, model.logits, model.trans],
                feed_dict=feed_dict
            )
            
            pred = decode(logits, lengths, trans, args)[0]
            pred = [ id2category[w] for w in pred ]
            for offset,p in enumerate( pred ):
                if p[0] == 'B':
                    if pred[offset+1][0]!='I':
                        continue
                    endPos = offset+1
                    for i in range(1,10):
                        if pred[offset+i][0]=='I':
                            endPos = offset+i
                        else:
                            break
                    startPos_ = d['lineStartPosition'] + offset-1
                    endPos_ = d['lineStartPosition'] + endPos
                    pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:endPos+1]) )] )
                    num += 1
                if p[0] == 'S':
                    startPos_ = d['lineStartPosition'] + offset-1
                    endPos_ = d['lineStartPosition'] + offset
                    pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:offset+1]) )] )
                    num += 1
        pred_list = pd.DataFrame(pred_list)
        
        import datetime
        path = os.path.join( '../submit/', "{}".format(datetime.datetime.now().strftime('%m%d')) )
        if not os.path.exists(path):
            os.makedirs(path)
        pred_list.to_csv( path + '/{0}.ann'.format(iid), encoding='utf8', header=False, sep='\t', index=False )
    


