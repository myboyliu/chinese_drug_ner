import json
import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf

from bert import tokenization
from model import entity_model
from utils import getBatches

from data_utils import ssbs, load_data
from conf import Config
from optimization import create_optimizer

def read_json_from_file(file_name):
    #read json file
    print(file_name)
    print('load .. '+file_name)
    fp = open(file_name, "rb")
    data = json.load(fp)
    fp.close()
    return data


def initLabel():
    if os.path.exists('cache/category2id.json'):
        category2id = read_json_from_file('cache/category2id.json')
    else:
        print ('error-category2id')
    return category2id


def trans2data(dta):
    
    dta['y'] = dta['label'].apply(lambda x: [ category2id[i] for i in x ])
    dta_ = []
    for d in tqdm(dta.iterrows()):
        d = d[1]
        text = d['text'][:args.max_x_length]
        lnt = len(text)
        dta_.append( [ text, lnt, d['y'], d['l_other'] ] )
    np.random.shuffle(dta_)
    
    return dta_




# 训练集样本长度分布 | 测试集样本长度分布 
total_data, dfLabel = load_data()

if not os.path.exists('./cache/random_order_train_dev.json'):
    random_order = list(range(len(total_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./cache/random_order_train_dev.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./cache/random_order_train_dev.json'))


args = Config()
tokenizer = tokenization.FullTokenizer( vocab_file=args.vocab_file, do_lower_case=False)

category2id = initLabel()
id2category = {j:i for i,j in category2id.items()}
args.relation_num = len(category2id)
args.category2id = category2id


print ('\n...training...')
k_folds = args.k_folds
for mode in range(k_folds):
    
    train_data = {}
    dev_data = {}
    for i in random_order:
        if i % k_folds != mode:
            train_data[str(i)] = total_data[str(i)]
        else:
            dev_data[str(i)] = total_data[str(i)]
            
            
    train_data = ssbs( train_data, dfLabel, args.max_x_length )
    dev_data = ssbs( dev_data, dfLabel, args.max_x_length )
    
    train_data = trans2data(train_data)
    batchesTrain = getBatches( train_data, args, tokenizer )
    
    dev_data = trans2data(dev_data)
    
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = entity_model( args )  # 读取模型结构图
            
            # 超参数设置
            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step,
                                                       args.decay_rate, staircase=True)
            
            normal_optimizer = tf.train.AdamOptimizer(learning_rate)  # 下接结构的学习率
            
            all_variables = graph.get_collection('trainable_variables')
            word2vec_var_list = [x for x in all_variables if 'bert' in x.name]  # BERT的参数
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]  # 下接结构的参数
            print('bert train variable num: {}'.format(len(word2vec_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int( len(train_data) / args.batch_size * args.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if word2vec_var_list:  # 对BERT微调
                print('word2vec trainable!!')
                word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                    model.loss, args.embed_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05) , use_tpu=False ,  variable_list=word2vec_var_list
                )
    
                train_op = tf.group(normal_op, word2vec_op)  # 组装BERT与下接结构参数
            else:
                train_op = normal_op
    
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            if args.continue_training:
                print('recover from: {}'.format(args.checkpoint_path))
                saver.restore(session, args.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            
            current_step = 0
            for e in range(args.train_epoch):
                print("----- Epoch {}/{} -----".format(e + 1, args.train_epoch))
                
                loss_ = 0
                ln_ = len(batchesTrain)
                for nextBatch in tqdm(batchesTrain, desc="Training"):
                    
                    feed = {
                            model.input_x_word: nextBatch.input_x_word,
                            model.input_mask: nextBatch.input_mask,
                            model.input_relation: nextBatch.inputs_y,
                            model.input_x_len: nextBatch.sequence_lengths,
                            model.keep_prob: args.keep_prob,
                            model.is_training: True
                            }
                    _, step, _, loss, lr = session.run(
                            fetches=[train_op,
                                     global_step,
                                     embed_step,
                                     model.loss,
                                     learning_rate
                                     ],
                            feed_dict=feed)
                    current_step += 1
                    loss_ += loss
                
                tqdm.write("----- Step %d -- train:  loss %.4f " % ( current_step, loss_/ln_))
                
                from eval_metrics import getScore
                P, R, F1 = getScore( model, dev_data, session, args, tokenizer )
                print('dev set : precision: {:.4f}, recall {:.4f}, f1 {:.4f}\n'.format(P, R, F1))
                
                if F1 > 0.67:  # 保存F1大于0的模型
                    out_dir = os.path.abspath(
                        os.path.join( args.model_dir, "{}".format(mode) ) )
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    saver.save(session, os.path.join( out_dir, 'model_{:.4f}_{:.4f}_{:.3f}_'.format(P, R, F1)),
                               global_step=step)



