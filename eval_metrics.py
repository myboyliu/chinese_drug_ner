#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from tensorflow.contrib.crf import viterbi_decode



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

def get_P_R_F( id2category, y_pred_list, y_true_list, ldct_list_tokens ):
    
    pred_list = []
    true_list = []
    for d, (pred, true_label, token) in enumerate( zip( y_pred_list, y_true_list, ldct_list_tokens ) ):
        pred = [ id2category[w] for w in pred ]
        for offset,p in enumerate( pred ):
            if p[0] == 'B':
                if pred[offset+1][0]!='I':
                    continue
#                    if pred[offset+2][0]=='I':
#                        endPos = offset+1
#                        for i in range(2,10):
#                            if pred[offset+i][0]=='I':
#                                endPos = offset+i
#                            else:
#                                break
#                        pred_list.extend( [( d, offset-1,endPos,p[2:],''.join(token[offset:endPos+1]) )] )
#                
#                    else:
#                        continue
                endPos = offset+1
                for i in range(1,10):
                    if pred[offset+i][0]=='I':
                        endPos = offset+i
                    else:
                        break
                pred_list.extend( [( d, offset-1,endPos,p[2:],''.join(token[offset:endPos+1]) )] )
                
            if p[0] == 'S':
                print (d)
                pred_list.extend( [( d, offset-1,offset,p[2:],''.join(token[offset:offset+1]) )] )
        true_list.extend( [(d,i[0],i[1],i[2],i[3]) for i in true_label] )
        
    rightNum = len( set(true_list) & set(pred_list) )
    P = rightNum / len(true_list)
    R = rightNum/len(pred_list)
    try:
        F1 = 2 * P * R / (P + R)
    except:
        F1 = 0
        
    return P, R, F1

def get_P_R_F2( id2category, y_pred_list, y_true_list, ldct_list_tokens ):
    
    pred_list = []
    true_list = []
    for d, (pred, true_label, token) in enumerate( zip( y_pred_list, y_true_list, ldct_list_tokens ) ):
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
                pred_list.extend( [( d, offset-1,endPos,p[2:],''.join(token[offset:endPos+1]) )] )
            if p[0] == 'S':
                pred_list.extend( [( d, offset-1,offset,p[2:],''.join(token[offset:offset+1]) )] )
        true_list.extend( [(d,i[0],i[1],i[2],i[3]) for i in true_label] )
        
    rightNum = len( set(true_list) & set(pred_list) )
    P = rightNum / len(true_list)
    R = rightNum/len(pred_list)
    try:
        F1 = 2 * P * R / (P + R)
    except:
        F1 = 0
        
    return P, R, F1


from utils import process
def getScore(model, dev_iter, session, args, tokenizer):

    y_pred_list = []
    y_true_list = []
    ldct_list_tokens = []
    id2category = {j:i for i,j in args.category2id.items()}
    
    for sample in tqdm( dev_iter ):
        tokens,t1, t2 = process( sample[0], args, tokenizer )
        y = list(sample[2])
        y = [ args.category2id["[CLS]"] ] + y + [ args.category2id["[SEP]"] ] + [ args.category2id["O"] ] * (args.max_x_length - len(y))
        feed_dict = {
            model.input_x_word: [t1],
            model.input_mask: [t2],
            model.input_x_len: [sample[1]+2],
            model.input_relation: [y],
            
            model.keep_prob: 1,
            model.is_training: False,
        }
        
        lengths, logits, trans = session.run(
            fetches=[model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )
        
        predict = decode(logits, lengths, trans,args)[0]
        
        y_pred_list.append(predict)
        y_true_list.append(sample[-1])
        ldct_list_tokens.append(tokens)
        
    precision, recall, f1 = get_P_R_F( id2category, y_pred_list, y_true_list, ldct_list_tokens )

    return precision, recall, f1



'''



sample = dev_iter[8]
tokens,t1, t2 = process( sample[0], args, tokenizer )
y = list(sample[2])
y = [ args.category2id["[CLS]"] ] + y + [ args.category2id["[SEP]"] ] + [ args.category2id["O"] ] * (args.max_x_length - len(y))
feed_dict = {
    model.input_x_word: [t1],
    model.input_mask: [t2],
    model.input_x_len: [sample[1]+2],
    model.input_relation: [y],
    
    model.keep_prob: 1,
    model.is_training: False,
}

lengths, logits, trans = session.run(
    fetches=[model.lengths, model.logits, model.trans],
    feed_dict=feed_dict
)

predict = decode(logits, lengths, trans,args)[0]

pred = [ id2category[w] for w in predict ]
pred_list = []
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
        pred_list.extend( [( offset-1,endPos,p[2:],''.join(tokens[offset:endPos+1]) )] )
    if p[0] == 'S':
        pred_list.extend( [( offset-1,offset,p[2:],''.join(tokens[offset:offset+1]) )] )
        
y = sample[-1]


chakan = pd.DataFrame({'pred':pred,'word':tokens})


'''










