class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        
        self.decay_rate = 0.85
        self.decay_step = 200
        self.num_checkpoints = 20 * 3
        
        self.train_epoch = 7
        self.max_x_length = 500
        
        self.learning_rate = 1e-4  # 下接结构的学习率
        self.embed_learning_rate = 5e-5  # BERT的微调学习率
        self.batch_size = 7
        self.k_folds = 7
        
        # BERT预训练模型的存放地址
        # self.data_root = './data/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
        # self.bert_file = './data/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        # self.bert_config_file = './data/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        # self.vocab_file = './data/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
        self.data_root = '/dataset/contest/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/'
        self.bert_file = '/dataset/contest/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
        self.bert_config_file = '/dataset/contest/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
        self.vocab_file = '/dataset/contest/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
        self.continue_training = False
        
        # 存放的模型名称，用以预测
        self.checkpointPath = './checkpointPath/'
        self.model_dir = './model_saved/right/'  # 模型存放地址
        
        #self.model_type = 'idcnn'  # 使用idcnn
        self.model_type = 'bilstm'  # 使用bilstm
        self.lstm_dim = 256
        self.dropout = 0.8
        self.use_origin_bert = False  # True:使用原生bert, False:使用动态融合bert









