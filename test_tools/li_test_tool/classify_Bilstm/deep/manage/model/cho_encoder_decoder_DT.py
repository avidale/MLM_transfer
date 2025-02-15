﻿# -*- encoding = gb18030 -*-
import os

from test_tools.li_test_tool.classify_Bilstm.deep.manage.manager import ModelManager
from test_tools.li_test_tool.classify_Bilstm.deep.util.parameter_operation import load_params_val, save_confs_val
from test_tools.li_test_tool.classify_Bilstm.deep.algorithms.networks.cho_encoder_decoder_DT import RnnEncoderDecoderNetwork
from test_tools.li_test_tool.classify_Bilstm.deep.dataloader.corpus_reader_dialog_topic import CorpusReaderDialogTopic



class RnnEncoderDecoder(ModelManager) :
    
    def __init__(self, dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                 train_rate, valid_rate, test_rate, algo_name, charset, mode) :
        """
        Need to set these attributes.
            1. conf_dict: configuration of the model.
            2. cr: CorpursReader for operate data.
            3. model: the network model.
        """
        self.conf_dict = {'algo_name': algo_name, 'batch_size': 256,
                          'train_valid_test_rate': [train_rate, valid_rate, test_rate],
                          'split_level': 'zi', 'pre_word_embedding': False,
                          'word_embedding_dim': 128, 'n_topics':5000, 'topic_embedding_dim' :256,'max_sentence_word_num': 150,
                          'min_sentence_word_num': 1, 'is_BEG': False, 'is_END': True,
                          'hidden_dim': 512, 'charset': charset, 'shuffle': False,
                          'save_freq': 100}
        def get_params_file_name(key):
            return 'fucking_toxic_name'
            if key == "amazon":
                return "ChoEncoderDecoderDT_e73ad85a8576a7a1f9a55e9e322d6313"
            elif key == "yelp":
                return "ChoEncoderDecoderDT_6ab0ac1b8974af9f2196a0b7edcf239a"

        self.param_path = os.path.join(dataset_folder, 'dialog', get_params_file_name(dict_file.split('.')[-1]) + '.model')
        #print(self.param_path)
        #self.param_path ='ChoEncoderDecoderTopic_5908276eb2ae513520ca72135e5b82d0.model83'
        #self.param_path='ChoEncoderDecoderDT_4575b6c5893c10a009e29b6eb2988387.model42'
        #self.param_path='ChoEncoderDecoderDT_cc7f5ed5d9e9fe5a90a012f4e017106a.model'
        param_dict = load_params_val(self.param_path)
        self.conf_path = os.path.join(dataset_folder, 'model', 'dialog', get_params_file_name(dict_file.split('.')[-1]) + '.conf')
        #save_confs_val(self.conf_dict, self.conf_path)
        # set corpus reader
        if mode == 'train' :
            self.cr = CorpusReaderDialogTopic(dataset_file=dataset_file,
                                         stopwords_file=stopwords_file,
                                         dict_file=dict_file, 
                                         word_embedding_file=None, 
                                         train_valid_test_rate=self.conf_dict['train_valid_test_rate'], 
                                         charset=self.conf_dict['charset'], 
                                         max_sentence_word_num=self.conf_dict['max_sentence_word_num'], 
                                         min_sentence_word_num=self.conf_dict['min_sentence_word_num'],
                                         is_BEG_available=self.conf_dict['is_BEG'],
                                         is_END_available=self.conf_dict['is_END'])
        else:
            self.cr = CorpusReaderDialogTopic(dataset_file=None,
                                         stopwords_file=stopwords_file,
                                         dict_file=dict_file, 
                                         word_embedding_file=None,
                                         train_valid_test_rate=self.conf_dict['train_valid_test_rate'], 
                                         charset=self.conf_dict['charset'], 
                                         max_sentence_word_num=self.conf_dict['max_sentence_word_num'], 
                                         min_sentence_word_num=self.conf_dict['min_sentence_word_num'],
                                         is_BEG_available=self.conf_dict['is_BEG'],
                                         is_END_available=self.conf_dict['is_END'])
        # set model
        self.model = RnnEncoderDecoderNetwork(n_words=len(self.cr.get_word_dictionary()),
                                              hidden_status_dim=self.conf_dict['hidden_dim'],
                                              word_embedding_dim=self.conf_dict['word_embedding_dim'],
                                              n_topics=self.conf_dict['n_topics'],
                                              topic_embedding_dim=self.conf_dict['topic_embedding_dim'],
                                              input_params=param_dict)
