#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random


def convert_single_example( tokenizer, text_a ):
    def _tokenize( token_dict,text ):
        R = []
        for c in text:
            if c in token_dict:
                R.append(c)
            elif c==' ':
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
    tokens_a = _tokenize( tokenizer.vocab, text_a)
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 2
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
    input_mask = [1] * len(input_ids) # 创建mask
    
    return tokens,input_ids,input_mask # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数

def process(text, args, tokenizer):
    
    tokens,t1,t2 = convert_single_example( tokenizer, text )
    t1 = t1+[0] * (args.max_x_length + 2 - len(t1))
    t2 = t2+[0] * (args.max_x_length + 2 - len(t2))
    
    return tokens,t1,t2



class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.input_x_word = []
        self.input_mask = []
        self.inputs_y = []
        self.sequence_lengths = []


def createBatch(samples, args, tokenizer):

    batch = Batch()
    for sample in samples:
        #将source PAD值本batch的最大长度
        
        tokens,t1, t2 = process( sample[0], args, tokenizer )
        batch.input_x_word.append( t1 )
        batch.input_mask.append( t2 )
        batch.sequence_lengths.append(sample[1]+2)
        
        y = list(sample[2])
        batch.inputs_y.append( [ args.category2id["[CLS]"] ] + y + [ args.category2id["[SEP]"] ] + 
                              [ args.category2id["O"] ] * (args.max_x_length - len(y)) )
        
    return batch


def getBatches(data, args, tokenizer):

    #每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, args.batch_size):
            yield data[i:min(i + args.batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch( samples, args, tokenizer )
        batches.append(batch)
    return batches
    



