#!/usr/bin/python3
import sys
sys.path.append('../..')
from source.preprocess.data_reader import data_reader
from source.models.ctc_model import ctc_model
import pickle

def main():
    timit_path = 'timit_data'
    feat_type = 'mfcc'
    frame_length = 0.025
    frame_shift = 0.010
    num_batches = 80
    context = 0
    over_ride_setups = False

    timit = data_reader(timit_path, save_dir='./model_data', over_ride=False)
    timit.getFiles(num_batches=num_batches, test_path='/test_core')
    timit.getAllSegments(win_length=frame_length, win_shift=frame_shift)
    timit.load_lexicon(timit_path+'/lexicon')
    timit.getAllMFCCs()
    # timit.getAllFBs()
    timit.setupInputsForModel(feat_type=feat_type, context=context, over_ride=False)
    if over_ride_setups: timit.kmeans_fit(100)
    timit.setupInputsForKmeans(over_ride=over_ride_setups)
    input_size = timit.get_input_size(kmeans=True)

    myModel = ctc_model(save_dir='./models/model_timit_mfcc_kmeans', run_name='timit_mfcc_kmeans')
    myModel.conv_kernels = [39*2]
    myModel.conv_strides = [39]
    myModel.num_filters = 20
    myModel.pooling_size = 1
    myModel.cnn_hidden_nums = 256

    myModel.define_paths(timit.inputs_train_path + '_k', timit.outputs_train_path, timit.inputs_test_path + '_k', timit.outputs_test_path)
    myModel.setup(timit.lexicon, input_size, use_cnn=False, num_hidden=256, num_layers=1, mlp_hiddens=[512,256], learning_rate=1e-3, max_len=800)
    myModel.train_model(num_epochs=450, num_batches=num_batches, num_batches_test=num_batches, keep_prob=0.8, data_usage=1, continue_model=True)

    myModel.restore_model(epoch=51)
    print('LER:', myModel.test_ler(num_batches))

if __name__ == "__main__":
    main()
