#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
import os

global baseTrainPath, baseTestPath
baseTrainPath = '../data/round1_train/train/'
baseTestPath = '../data/round1_test/chusai_xuanshou/'

def readTxt( path, basePath ):
    
    path = basePath + path
    file = open( path, 'r', encoding='utf8').readlines()
    if len(file)!=1:
        return ''
    else:
        return file[0]

def readLabel( path, iid ):
    
    path = baseTrainPath + path
    dta = pd.read_csv(path, sep='\t', names=['id','entityInfo','entity'])
    dta['category'] = dta['entityInfo'].apply(lambda x:x.split(' ')[0])
    dta['pe1'] = dta['entityInfo'].apply(lambda x:x.split(' ')[1]).astype(int)
    dta['pe2'] = dta['entityInfo'].apply(lambda x:x.split(' ')[2]).astype(int)
    dta['id'] = iid
    dta = dta[['id','entity','category','pe1','pe2']]
    
    return dta

def load_data():
    
    df = {}
    dfLabel = pd.DataFrame()
    for path in tqdm( os.listdir( baseTrainPath ) ):
        cid = path[-3:]
        if cid=='txt':
            df[path[:-4]] = readTxt(path, baseTrainPath)
        else:
            dta = readLabel(path, path[:-4])
            dfLabel = pd.concat( [dfLabel, dta] )
    
    print ( '\n ... 训练集实体个数为{0}，实体类别个数为{1} ...'.format( len(set(dfLabel['entity'])), len(set(dfLabel['category'])) ) )
    
    return df, dfLabel

def load_testData():
    
    test = {}
    for path in tqdm( os.listdir( baseTestPath ) ):
        test[path[:-4]] = readTxt(path, baseTestPath)
    
    return test




def ssbs( df, dfLabel, maxLen ):
    '''
    # sample split by Symbol
    '''
    def _split(iid, df, dfLabel):
        
        text = df[str(iid)]
        lb = dfLabel[dfLabel['id'] == str(iid)]
        lines = text.split('。')
        
        # 记录分词后的每个短句对应的起止位置
        infos = []
        for offset, tx in enumerate( lines ):
            if offset==0:
                info = { 'offset':offset, 
                               'lineStartPosition':0, 
                               'lineEndPosition':len(tx)-1 }
                lastEndPos = info['lineEndPosition']
            else:
                info = { 'offset':offset, 
                               'lineStartPosition':lastEndPos+1+1, 
                               'lineEndPosition':lastEndPos+1+1+len(tx)-1  }
                lastEndPos = info['lineEndPosition']
            entityNum = len (lb[ (lb['pe1']>=info['lineStartPosition']) &
                          (lb['pe2']<=info['lineEndPosition']) ] )
            info['entityNum'] = entityNum
            infos.append( info )
        infos = pd.DataFrame(infos)[['offset', 'entityNum', 'lineStartPosition', 'lineEndPosition']]
        
        # 长句拆分为短句
        def oneSample( lineStartPosition, dta):
            
            dta_ = dta[dta['lineEndPosition']>=maxLen+lineStartPosition]
            dta = dta[dta['lineEndPosition']<maxLen+lineStartPosition]
            if len(dta)==0:
                return rs
            rs.append( {'offsetStart':dta['offset'].values[0],
             'offsetEnd':dta['offset'].values[-1],
             'lineStartPosition':dta['lineStartPosition'].values[0],
             'lineEndPosition':dta['lineEndPosition'].values[-1]} )
            
            lineStartPosition = dta['lineEndPosition'].values[-1] + 2
            oneSample( lineStartPosition, dta_)
            
            return rs
        rs = []
        lineStartPosition = 0
        rs = oneSample( lineStartPosition, infos.copy())
        
        #构造训练样本
        samples = []
        for s in rs:
            # 拆分为短句后的句子
            tx = text[ s['lineStartPosition']:s['lineEndPosition']+1 ]
            label = lb[ (lb['pe1']>=s['lineStartPosition']) &
                          (lb['pe2']<=s['lineEndPosition']) ]
            l_ = ['O'] * len(tx) # 训练的label
            l_other = []
            for left,right,cate,entity in zip(label['pe1'],label['pe2'],label['category'],label['entity']):
                left = left-s['lineStartPosition']
                right = right-s['lineStartPosition']
                l_other.append( (left,right,cate,entity) )
                if left == right:
                    l_[left] = 'S'+'-'+cate
                else:
                    l_[left] = 'B'+'-'+cate
                    for i in range(left+1,right-1):
                        l_[i] = 'I'
                    l_[right-1] = 'I'
            samples.append( {'text':tx, 'label':l_, 'l_other':l_other} )
            
        return pd.DataFrame(samples)
    
    
    train = pd.DataFrame()
    for iid in df.keys():
        train = pd.concat( [train, _split(iid, df, dfLabel)] )
    
    return train




def ssbsTest( df, maxLen ):
    '''
    # sample split by Symbol
    '''
    def _split( text ):
        
        lines = text.split('。')
        # 记录分词后的每个短句对应的起止位置
        infos = []
        for offset, tx in enumerate( lines ):
            if offset==0:
                info = { 'offset':offset, 
                               'lineStartPosition':0, 
                               'lineEndPosition':len(tx)-1 }
                lastEndPos = info['lineEndPosition']
            else:
                info = { 'offset':offset, 
                               'lineStartPosition':lastEndPos+1+1, 
                               'lineEndPosition':lastEndPos+1+1+len(tx)-1  }
                lastEndPos = info['lineEndPosition']
            infos.append( info )
        infos = pd.DataFrame(infos)[['offset', 'lineStartPosition', 'lineEndPosition']]
        
        # 长句拆分为短句
        def oneSample( lineStartPosition, dta):
            
            dta_ = dta[dta['lineEndPosition']>=maxLen+lineStartPosition]
            dta = dta[dta['lineEndPosition']<maxLen+lineStartPosition]
            if len(dta)==0:
                return rs
            rs.append( {'offsetStart':dta['offset'].values[0],
             'offsetEnd':dta['offset'].values[-1],
             'lineStartPosition':dta['lineStartPosition'].values[0],
             'lineEndPosition':dta['lineEndPosition'].values[-1]} )
            
            lineStartPosition = dta['lineEndPosition'].values[-1] + 2
            oneSample( lineStartPosition, dta_)
            
            return rs
        rs = []
        lineStartPosition = 0
        rs = oneSample( lineStartPosition, infos.copy())
        
        #构造训练样本
        samples = []
        for s in rs:
            # 拆分为短句后的句子
            tx = text[ s['lineStartPosition']:s['lineEndPosition']+1 ]
            s['tx'] = tx
            s['id'] = iid
            s['text'] = text
            samples.append( s )
            
        return pd.DataFrame(samples)
    
    test = pd.DataFrame()
    for iid in df.keys():
        test = pd.concat( [test, _split( df[str(iid)] )] )
    
    return test



#
#df, dfLabel = load_data()
#train = ssbs( df, dfLabel, maxLen=500 )
#






