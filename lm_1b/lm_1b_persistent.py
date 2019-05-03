import os
import sys
import numpy as np
from six.moves import xrange
import tensorflow as tf
from google.protobuf import text_format
import data_utils
import time
import pandas as pd
import glob
import math

class LM1B_model(object):
    
    def __init__(self, metadata):
        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50
        self.metadata =  metadata
        self._LoadModel(self.metadata['modelParameters']['pbtxt_path'], self.metadata['modelParameters']['ckpt_path'])
        self.vocab = data_utils.CharsVocabulary(self.metadata['modelParameters']['vocab_path'], self.MAX_WORD_LEN)        


    def _LoadModel(self, gd_file, ckpt_file):
      """Load the model from GraphDef and Checkpoint.

      Args:
        gd_file: GraphDef proto text file.
        ckpt_file: TensorFlow Checkpoint file.

      Returns:
        TensorFlow session and tensors dict.
      """
      with tf.Graph().as_default():
        sys.stderr.write('Recovering graph.\n')
        with tf.gfile.FastGFile(gd_file, 'r') as f:
          s = f.read().decode()
          gd = tf.GraphDef()
          text_format.Merge(s, gd)

        tf.logging.info('Recovering Graph %s', gd_file)
        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
        ] = tf.import_graph_def(gd, {}, ['states_init',
                                         'lstm/lstm_0/control_dependency:0',
                                         'lstm/lstm_1/control_dependency:0',
                                         'softmax_out:0',
                                         'class_ids_out:0',
                                         'class_weights_out:0',
                                         'log_perplexity_out:0',
                                         'inputs_in:0',
                                         'targets_in:0',
                                         'target_weights_in:0',
                                         'char_inputs_in:0',
                                         'all_embs_out:0',
                                         'Reshape_3:0',
                                         'global_step:0'], name='')

        sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

      self.sess = sess
      self.t = t
      


    def _EvalSentences(self, sentences):
        """Evaluate the log probability of the input sentences in the directory

            Args:
            vocab: vocabulary object.
            sentences: list of strings
        """
        print('Evaluating sentences')
        start_time = time.time()
        current_step = self.t['global_step'].eval(session=self.sess)
        sys.stderr.write('Loaded step %d.\n' % current_step)

        # instantiate a dataset generator       
        dataset = data_utils.LM1BDataset(vocab=self.vocab)
        
        result_dfs = []
        for sentence in sentences:
            # set the sentence first
            dataset.sentence = sentence
            # then the call to batch with method "list" converts the sentence object            
            data_gen = dataset.get_batch(self.BATCH_SIZE, self.NUM_TIMESTEPS, method='list', forever=False)
            word_probabilities = []
            words = []
        
            for i, (inputs, char_inputs, _, targets, weights) in enumerate(data_gen):
              
                input_dict = {self.t['inputs_in']: inputs,
                            self.t['targets_in']: targets,
                            self.t['target_weights_in']: weights}
                if 'char_inputs_in' in self.t:
                    input_dict[self.t['char_inputs_in']] = char_inputs
              
                log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)
                softmax = self.sess.run(self.t['softmax_out'], feed_dict=input_dict) 
            
                log10_probability = -1 * np.log10(softmax[0,targets[0][0]] / np.sum(softmax))
                sys.stderr.write(self.vocab.id_to_word(targets[0][0])+' '+str(log10_probability)+'\n')
          
                words.append(self.vocab.id_to_word(targets[0][0]))          
                word_probabilities.append(log10_probability)

            sys.stderr.write('Sentence perplexity: %s\n' % str(np.sum(word_probabilities)))      
            sys.stderr.write('Elapsed: %s\n' % str(time.time() - start_time))      
        
            rdf = pd.DataFrame({'prob':word_probabilities, 'word':words})
            result_dfs.append(rdf)
            #!!! different serialization here

        return(pd.concat(result_dfs))

    def _EvalSentencesDicts(self, input_dict_list, base):
        """Evaluate the log probability of the input sentences in the directory

            Args:
            vocab: vocabulary object.
            input_dict_list: list of dictionaries
        """
        print('Evaluating sentences')
        start_time = time.time()
        current_step = self.t['global_step'].eval(session=self.sess)
        sys.stderr.write('Loaded step %d.\n' % current_step)

        # instantiate a dataset generator       
        dataset = data_utils.LM1BDataset(vocab=self.vocab)
        
        utterance_dfs = []
        for input_row in input_dict_list:
            # set the sentence first
            # !!! drop the eos and bos
            eval_utterance = [x for x in input_row['utterance_list'] if not (x in ('<s>','</s>'))]
            eval_utterance_string = ' '.join(eval_utterance)
            print('Evaluating:')
            print(eval_utterance_string)
            dataset.sentence = eval_utterance_string
            # then the call to batch with method "list" converts the sentence object            
            data_gen = dataset.get_batch(self.BATCH_SIZE, self.NUM_TIMESTEPS, method='list', forever=False)
            
            word_by_word = []
            word_by_word.append( #append a dummy start of sentence
                {
                    'token_id': 0,
                    'word': '<S>',
                    'log_prob': np.nan
            })

        
            for i, (inputs, char_inputs, _, targets, weights) in enumerate(data_gen):
              
                input_dict = {self.t['inputs_in']: inputs,
                            self.t['targets_in']: targets,
                            self.t['target_weights_in']: weights}
                if 'char_inputs_in' in self.t:
                    input_dict[self.t['char_inputs_in']] = char_inputs
              
                log_perp = self.sess.run(self.t['log_perplexity_out'], feed_dict=input_dict)
                softmax = self.sess.run(self.t['softmax_out'], feed_dict=input_dict) 
            
                log10_probability = -1 * np.log10(softmax[0,targets[0][0]] / np.sum(softmax))
                sys.stderr.write(self.vocab.id_to_word(targets[0][0])+' '+str(log10_probability)+'\n')

                word_by_word.append({
                    'token_id': i+1,
                    'word': self.vocab.id_to_word(targets[0][0]),
                    'log_prob': math.log(10. ** log10_probability, base)
                })

            utterance_df = pd.DataFrame(word_by_word) 
            utterance_df['utterance_id'] = input_row['utterance_id']

            sys.stderr.write('Sentence perplexity: %s\n' % str(np.sum(utterance_df.log_prob)))      
                  
            utterance_dfs.append(utterance_df)
        
        sys.stderr.write('Elapsed: %s\n' % str(time.time() - start_time))
        return(pd.concat(utterance_dfs))

