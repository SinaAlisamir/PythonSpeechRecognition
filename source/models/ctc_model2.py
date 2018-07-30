#!/usr/bin/python3
import os
import time
import sys
sys.path.append('../..')
import shutil
import pickle
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from source.utils.utils import pad_sequences
from source.utils.utils import sparse_tuple_from
from source.utils.utils import EditDistance


class ctc_model():
    def __init__(self, save_dir='./model', run_name='default'):
        super(ctc_model, self).__init__()
        self.model_dir = save_dir
        self.save_dir = save_dir + '/' + run_name
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.graphs_dir = save_dir + '/graphs' + '/' + run_name
        self.conv_kernels = [10]
        self.conv_strides = [10]
        self.num_filters = 10
        self.pooling_size = 1
        self.cnn_hidden_nums = 256


    def setup(self, lexicon, num_features, use_cnn=False, num_hidden=128, num_layers=1, mlp_hiddens=[512], learning_rate=1e-2, max_len=400):
        self.lexicon = lexicon
        self.dictionary = []
        self.delimiter = '|'
        self.num_features = num_features
        self.max_len = max_len
        self.mlp_hiddens = mlp_hiddens
        self.learning_rate = learning_rate
        self.setup_model(use_cnn=use_cnn, num_hidden=num_hidden, num_layers=num_layers)


    def setup_model(self, use_cnn=False, num_hidden=128, num_layers=1):
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_phns = len(self.lexicon)
        self.num_classes = self.num_phns+1 # +1 for blank states in CTC
        self.keep_prob = tf.Variable(1.0, trainable = False)

        self.sess = tf.Session()
        self.model_input = tf.placeholder(tf.float32, [None, self.max_len, self.num_features])
        self.model_targets = tf.sparse_placeholder(tf.int32)
        self.model_seqlen = tf.placeholder(tf.int32, [None])

        if use_cnn:
            self.model_matricized_x = tf.reshape(self.model_input, [-1, self.num_features])
            self.model_matricized_unary_scores, dim = self.cnn_model(self.model_matricized_x , self.num_features, self.num_phns)
            self.model_unary_scores = tf.reshape(self.model_matricized_unary_scores,[-1, self.max_len, dim])
            inputs = self.model_unary_scores
        else:
            inputs = self.model_input

        # cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.8)
        # rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(self.num_hidden) for _ in range(self.num_layers)]
        rnn_layers = [tf.contrib.rnn.GRUCell(self.num_hidden) for _ in range(self.num_layers)]
        stack = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, self.model_seqlen, dtype=tf.float32)
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.GRUCell(self.num_hidden), tf.contrib.rnn.GRUCell(self.num_hidden), inputs, self.model_seqlen, dtype=tf.float32)
        # outputs = tf.concat(outputs, 2)

        # rnn_layers2 = [tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True) for _ in range(self.num_layers)]
        # stack2 = tf.nn.rnn_cell.MultiRNNCell(rnn_layers2, state_is_tuple=True)
        # outputs2, _ = tf.nn.dynamic_rnn(stack2, outputs, self.model_seqlen, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        # outputs = tf.concat([outputs,outputs2], 2)
        outputs = tf.reshape(outputs, [-1, self.num_hidden])

        # W = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes], stddev=0.1))
        # b = tf.Variable(tf.constant(0., shape=[self.num_classes]))
        # logits = tf.matmul(outputs, W) + b
        logits, _ = self.mlp_model(outputs , self.num_hidden, self.num_classes, self.mlp_hiddens)
        logits = tf.reshape(logits, [batch_s, self.max_len, self.num_classes])
        self.logits = logits
        self.ctc_probs = tf.reshape(tf.nn.softmax(logits), [batch_s, self.max_len, self.num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(self.model_targets, logits, self.model_seqlen)
        self.cost = tf.reduce_mean(loss)

        # optimizer = tf.train.MomentumOptimizer(1e-02, 0.9).minimize(cost)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.model_seqlen)
        # self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,sequence_length=self.model_seqlen,beam_width=3,merge_repeated=False, top_paths=1)
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.model_targets))

        self.saver = tf.train.Saver(max_to_keep=None)
        tf.summary.scalar('ler',self.ler)
        tf.summary.scalar('cost',self.cost)
        self.summeries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.graphs_dir)
        self.num_of_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('num_of_params',self.num_of_params)


    def define_paths(self, inputs_train_path, outputs_train_path, inputs_test_path, outputs_test_path):
        self.inputs_train_path = inputs_train_path
        self.outputs_train_path = outputs_train_path
        self.inputs_test_path = inputs_test_path
        self.outputs_test_path = outputs_test_path

    def batch_append(self, batch):
        txt = '_' + str(batch)
        return txt

    def restore_model(self, epoch=-1):
        epoch_here = epoch
        if os.path.isdir(self.save_dir):
            if epoch_here == -1:
                with open(self.save_dir+'/last_epoch', "rb") as fp:   # Unpickling
                    epoch_here = pickle.load(fp)
            self.saver.restore(self.sess, self.model_path_for(epoch))
        else:
            print("path", self.save_dir, "not found!")

    def train_model(self, num_epochs=20, num_batches=10, num_batches_test=1, keep_prob=1, data_usage=1, valid_proportion=0.1, valid_patience=5, continue_model=False):

        last_epoch = 0
        if continue_model:
            with open(self.save_dir+'/last_epoch', "rb") as fp:   # Unpickling
                last_epoch = pickle.load(fp) + 1
            self.restore_model(epoch=last_epoch-1)
        else:
            self.sess.run(tf.global_variables_initializer())
            shutil.rmtree(self.graphs_dir); self.writer = tf.summary.FileWriter(self.graphs_dir)

        for curr_epoch in range(last_epoch, num_epochs):
            train_cost = train_ler = 0
            all_batches_lengths = 0
            batches = list(range(0,num_batches))
            random.shuffle(batches)
            for batch in batches[1:]:
                sys.stdout.write("\repoch %d " % int(curr_epoch+1))
                sys.stdout.write("%d%%" % int(100*batches.index(batch)/num_batches))
                with open(self.inputs_train_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    batch_train_inputs = pickle.load(fp)
                with open(self.outputs_train_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    batch_train_targets = sparse_tuple_from(pickle.load(fp))
                # batch_train_inputs = self.add_noise_to_batch(batch_train_inputs)
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs, maxlen=self.max_len)
                feed = {self.model_input: batch_train_inputs, self.model_targets: batch_train_targets, self.model_seqlen: batch_train_seq_len, self.keep_prob: keep_prob}
                batch_cost, _ = self.sess.run([self.cost, self.optimizer], feed)
                train_cost += batch_cost*len(batch_train_seq_len)
                train_ler += self.sess.run(self.ler, feed_dict=feed)*len(batch_train_seq_len)
                all_batches_lengths += len(batch_train_seq_len)
            train_cost /= all_batches_lengths
            train_ler /= all_batches_lengths
            test_ler = self.test_ler(num_batches_test)
            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, test_ler = {:.3f}"
            sys.stdout.write("\r")
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, test_ler))
            self.write_summary_for(curr_epoch, self.inputs_train_path + self.batch_append(batches[0]), self.outputs_train_path + self.batch_append(batches[0]))
            self.save_model(curr_epoch)


    def add_noise_to_batch(self, batch):
        result = batch
        for i,utter in enumerate(batch):
            for j,feat in enumerate(utter):
                noise = np.random.normal(loc=0,scale=0.075,size=len(feat))
                result[i][j] = result[i][j]+noise
        return result

    def save_model(self, epoch, keep_prob=1):
        self.saver.save(self.sess, self.model_path_for(epoch))
        with open(self.save_dir+'/last_epoch', "wb") as fp:   #Pickling
            pickle.dump(epoch, fp)

    def model_path_for(self, epoch):
        models_path = self.save_dir+'/model'+'_'+str(epoch)
        if not os.path.exists(models_path): os.makedirs(models_path)
        model_path = models_path +'/'+'model'+'.ckpt'
        return model_path

    def write_summary_for(self, epoch, inputs_path, outputs_path):
        with open(inputs_path, "rb") as fp:   # Unpickling
            inputs = pickle.load(fp)
        with open(outputs_path, "rb") as fp:   # Unpickling
            outputs = pickle.load(fp)
        batch_test_inputs, batch_test_seq_len = pad_sequences(inputs, maxlen=self.max_len)
        batch_test_targets = sparse_tuple_from(outputs)
        feed = {self.model_input: batch_test_inputs, self.model_targets: batch_test_targets, self.model_seqlen: batch_test_seq_len, self.keep_prob: 1}
        _, summ = self.sess.run([self.ler,self.summeries], feed_dict=feed)
        self.writer.add_summary(summ, global_step = epoch+1)
        _, summ = self.sess.run([self.cost,self.summeries], feed_dict=feed)
        self.writer.add_summary(summ, global_step = epoch+1)


    def test_ler(self, num_batches):
        all_batches_lengths = 0
        test_ler = 0
        for batch in range(num_batches):
            sys.stdout.write("\rtesting progress %d%%" % int(100*batch/num_batches))
            with open(self.inputs_test_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                batch_test_inputs = pickle.load(fp)
            with open(self.outputs_test_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                batch_test_outputs = pickle.load(fp)
            batch_len = len(batch_test_outputs)
            all_batches_lengths += batch_len
            test_ler += self.test_model(batch_test_inputs, batch_test_outputs)*batch_len
        sys.stdout.write("\r")
        test_ler /= all_batches_lengths
        return test_ler

    def test_model(self, inputs, outputs, keep_prob=1):
        batch_test_inputs, batch_test_seq_len = pad_sequences(inputs, maxlen=self.max_len)
        batch_test_targets = sparse_tuple_from(outputs)
        feed = {self.model_input: batch_test_inputs, self.model_targets: batch_test_targets, self.model_seqlen: batch_test_seq_len, self.keep_prob: keep_prob}
        d = self.sess.run(self.decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=self.sess)
        error = self.sess.run(self.ler, feed_dict=feed)
        return error

    def print_outputs(self, inputs, outputs, keep_prob=1):
        batch_test_inputs, batch_test_seq_len = pad_sequences(inputs, maxlen=self.max_len)
        batch_test_targets = sparse_tuple_from(outputs)
        feed = {self.model_input: batch_test_inputs, self.model_targets: batch_test_targets, self.model_seqlen: batch_test_seq_len, self.keep_prob: keep_prob}
        d = self.sess.run(self.decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=self.sess)
        all_originals = []
        all_decodeds = []
        for i, seq in enumerate(dense_decoded):
            seq = [s for s in seq if s != -1]
            all_originals.append(self.number_to_lex(outputs[i]))
            all_decodeds.append(self.number_to_lex(seq))
        return [self.no_delimiters(array) for array in all_originals], [self.no_delimiters(array) for array in all_decodeds]

    def no_delimiters(self, array):
        result = []
        for a in array:
            if a != self.delimiter:
                result.append(a)
        return result


    def number_to_lex(self, sequence):
        lex_seq = []
        for i in range(len(sequence)):
            if sequence[i] < len(self.lexicon):
                lex_seq.append(self.lexicon[sequence[i]])
            else:
                continue
                # lex_seq.append(' ')
        # lex_seq = [self.lexicon[sequence[i]] for i in range(len(sequence))]
        return lex_seq


    def mlp_model(self, theInput, num_features, num_tags, hidden_layers):
        print('theInput',theInput.get_shape().as_list())
        Weight = self.weight_variable([num_features, hidden_layers[0]])
        Bias = self.bias_variable([hidden_layers[0]])
        y_out = tf.nn.relu(tf.matmul(theInput, Weight) + Bias)
        y_out = tf.nn.dropout(y_out, keep_prob=self.keep_prob)
        for i in range(len(hidden_layers)-1):
            Weight = self.weight_variable([hidden_layers[i], hidden_layers[i+1]])
            Bias = self.bias_variable([hidden_layers[i+1]])
            y_out = tf.nn.relu(tf.matmul(y_out, Weight) + Bias)
            y_out = tf.nn.dropout(y_out, keep_prob=self.keep_prob)
        Weight = self.weight_variable([hidden_layers[len(hidden_layers)-1], num_tags])
        Bias = self.bias_variable([num_tags])
        y_out = tf.matmul(y_out, Weight) + Bias
        # y_out = tf.nn.softmax(y_out)
        dim = np.prod(y_out.get_shape().as_list()[1:])
        print('y_out',y_out.get_shape().as_list())
        return y_out, dim

    def weight_variable(self, shape):
        initial = tf.random_normal(shape,mean=0.0,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.random_normal(shape,mean=0.1,stddev=0.1)
        return tf.Variable(initial)

    def cnn_model(self, theInput, num_features, num_tags):
        conv_kernels = self.conv_kernels
        conv_strides = self.conv_strides
        num_filters = self.num_filters
        pooling_size = self.pooling_size

        x_reshaped = tf.reshape(theInput, [-1, num_features, 1, 1])
        print(x_reshaped.get_shape().as_list())

        conv = tf.layers.conv2d(inputs=x_reshaped,
                                 filters=num_filters,
                                 kernel_size=[conv_kernels[0], 1],
                                 name='conv0',
                                 strides=[conv_strides[0], 1],
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0,0.1),
                                 padding='valid')

        pool = tf.layers.max_pooling2d(inputs=conv,
                                 pool_size=[pooling_size, 1],
                                 strides=[pooling_size, 1],
                                 padding='valid')
        pool = tf.nn.relu(pool)
        print(pool.get_shape().as_list())

        for i in range(1,len(conv_kernels)):
            conv = tf.layers.conv2d(inputs=pool,
                                     filters=num_filters,
                                     kernel_size=[conv_kernels[i], 1],
                                     strides=[conv_strides[i], 1],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0,0.1),
                                     padding='valid')

            pool = tf.layers.max_pooling2d(inputs=conv,
                                     pool_size=[pooling_size, 1],
                                     strides=[pooling_size, 1],
                                     padding='valid')
            pool = tf.nn.relu(pool)
            print(pool.get_shape().as_list())

        dim = np.prod(pool.get_shape().as_list()[1:])
        pool_flat = tf.reshape(pool, [-1, dim])
        # return pool_flat, dim

        size = self.cnn_hidden_nums
        Weight = self.weight_variable([dim, size])
        Bias = self.bias_variable([size])
        y_out = (tf.matmul(pool_flat, Weight) + Bias)
        y_out = tf.nn.dropout(y_out, keep_prob=self.keep_prob)

        return y_out, size
