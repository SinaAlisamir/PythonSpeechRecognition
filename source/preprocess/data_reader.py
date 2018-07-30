#!/usr/bin/python3
import os
import sys
import pickle
import glob
import soundfile as sf
import numpy as np
import python_speech_features
import random
import shutil
from sklearn.cluster import MiniBatchKMeans

class data_reader():
    def __init__(self, path, save_dir='./data', over_ride=False):
        super(data_reader, self).__init__()
        self.lexicon = []
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if over_ride: shutil.rmtree(save_dir); os.makedirs(save_dir)
        self.lexicon_path = save_dir+"/lexicon"
        self.files_train_path = save_dir+"/files_train"
        self.files_test_path = save_dir+"/files_test"
        self.lexs_train_path = save_dir+"/lexs_train"
        self.lexs_test_path = save_dir+"/lexs_test"
        self.data_train_path = save_dir+"/data_train"
        self.data_test_path = save_dir+"/data_test"
        self.data_train_normalized_path = save_dir+"/data_train_normalized"
        self.data_test_normalized_path = save_dir+"/data_test_normalized"
        self.data_train_feats_path = save_dir+"/data_train_feats"
        self.data_test_feats_path = save_dir+"/data_test_feats"
        self.inputs_train_path = save_dir+"/inputs_train"
        self.outputs_train_path = save_dir+"/outputs_train"
        self.inputs_test_path = save_dir+"/inputs_test"
        self.outputs_test_path = save_dir+"/outputs_test"
        self.path = path #path = "/usr/username/TIMIT/TIMIT"
        self.win_length = 0.025
        self.win_shift = 0.01


    def getFiles(self, amount=1e10, num_batches=1, test_path='test', train_path='train', audio_format='.wav'):
        self.num_batches = num_batches
        txts_train = glob.glob(self.path+"/"+train_path+"/*.txt", recursive=True)
        txts_train = txts_train[:min(len(txts_train),amount)]
        print('number of utterances for train:', len(txts_train))
        txts_trains = self.make_batches(txts_train)
        wavs_trains = [self.get_wavs_from_txts(txt, audio_format) for txt in txts_trains]
        self.txts_train = txts_train
        txts_test = glob.glob(self.path+"/"+test_path+"/*.txt", recursive=True)
        txts_test = txts_test[:min(len(txts_test),amount)]
        print('number of utterances for test:', len(txts_test))
        txts_tests = self.make_batches(txts_test)
        wavs_tests = [self.get_wavs_from_txts(txt, audio_format) for txt in txts_tests]
        self.txts_test = txts_test
        self.get_waves_and_lexs(wavs_trains, txts_trains, self.files_train_path, self.lexs_train_path)
        self.get_waves_and_lexs(wavs_tests, txts_tests, self.files_test_path, self.lexs_test_path)

    def get_waves_and_lexs(self, wavs, texts, text_path, lexs_path):
        for b in range(len(texts)):
            if (os.path.isfile(text_path + self.batch_append(b)) and os.path.isfile(lexs_path + self.batch_append(b))): continue
            files = []; lexs = []
            for i in range(len(texts[b])):
                line = open(texts[b][i]).read()
                labels = line.split()
                lexs.append(labels)
                wave, _ = sf.read(wavs[b][i])
                files.append(wave)
            with open(text_path + self.batch_append(b), "wb") as fp:   #Pickling
                pickle.dump(files, fp)
            with open(lexs_path + self.batch_append(b), "wb") as fp:   #Pickling
                pickle.dump(lexs, fp)

    def make_batches(self, array):
        batches = []
        max_l = len(array); data_l = max_l//self.num_batches
        for i in range(self.num_batches):
            index1 = i*data_l
            index2 = min((i+1)*data_l,max_l)
            batches.append(array[index1:index2])
        return batches

    def get_wavs_from_txts(self, wavs, ext):
        ext_list = []
        for path in wavs:
            new_path = path[0:len(path)-4]+ext
            ext_list.append(new_path)
        return ext_list

    def batch_append(self, batch):
        txt = '_' + str(batch)
        return txt

    def getAllSegments(self, win_length=0.025, win_shift=0.01):
        self.win_length = win_length
        self.win_shift = win_shift
        print('segmenting waves for train data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rsegmenting waves %d%%" % int(100*b/(self.num_batches)))
            self.get_segments_for_batch(self.data_train_path, self.data_train_normalized_path, self.files_train_path, b)
        sys.stdout.write("\r")
        print('segmenting waves for test data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rsegmenting waves %d%%" % int(100*b/(self.num_batches)))
            self.get_segments_for_batch(self.data_test_path, self.data_test_normalized_path, self.files_test_path, b)
        sys.stdout.write("\r")

    def get_segments_for_batch(self, path, path_norm, files_path, batch):
        if not (os.path.isfile(path + self.batch_append(batch)) and os.path.isfile(path_norm + self.batch_append(batch))):
            with open(files_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                files = pickle.load(fp)
                data, data_normalized = self.getSegments(files, win_length=self.win_length, win_shift=self.win_shift)
                with open(path + self.batch_append(batch), "wb") as fp:   #Pickling
                    pickle.dump(data, fp)
                with open(path_norm + self.batch_append(batch), "wb") as fp:   #Pickling
                    pickle.dump(data_normalized, fp)

    def getSegments(self, theFiles, win_length=0.025, win_shift=0.01):
        self.win_length = win_length
        self.win_shift = win_shift
        sample_length = win_length*16000
        sample_shift = win_shift*16000
        theData = []
        theData_normalized = []
        for wave in theFiles:
            theFileData = []
            theFileData_normed = []
            wavs = python_speech_features.sigproc.framesig(wave, sample_length, sample_shift)
            for wav in wavs:
                wav_normed = self.normalize(wav)
                theFileData.append(wav)
                theFileData_normed.append(wav_normed)
            theData.append(theFileData)
            theData_normalized.append(theFileData_normed)
        return theData, theData_normalized


    def getAllMFCCs(self):
        print('getting MFCCs for train data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rgetting MFCCs %d%%" % int(100*b/(self.num_batches)))
            self.get_MFCCs_for_batch(self.data_train_feats_path, self.files_train_path, b)
        sys.stdout.write("\r")
        print('getting MFCCs for test data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rgetting MFCCs %d%%" % int(100*b/(self.num_batches)))
            self.get_MFCCs_for_batch(self.data_test_feats_path, self.files_test_path, b)
        sys.stdout.write("\r")

    def get_MFCCs_for_batch(self, path, files_path, batch):
        if not os.path.isfile(path + self.batch_append(batch)):
            with open(files_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                files = pickle.load(fp)
                Data_feats = self.getMFCCs(files)
                with open(path + self.batch_append(batch), "wb") as fp:   #Pickling
                    pickle.dump(Data_feats, fp)

    def getMFCCs(self, theData, context_size=4):
        theData_feats = []
        for i,utterance in enumerate(theData):
            theSegmentData = []
            mfccs = []
            mfcc_norms = []
            labels = []
            mfccs = python_speech_features.base.mfcc(utterance, winlen=self.win_length, winstep=self.win_shift, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
            mfcc_norms = [self.normalize(mfcc) for mfcc in mfccs]
            delta1 = python_speech_features.base.delta(mfccs, context_size)
            delta1_norms = [self.normalize(delta) for delta in delta1]
            delta2 = python_speech_features.base.delta(delta1, context_size)
            delta2_norms = [self.normalize(delta) for delta in delta2]
            for i in range(len(mfccs)):
                feats = np.append(mfcc_norms[i], np.append(delta1_norms[i],delta2_norms[i]))
                theSegmentData.append(feats)
            theData_feats.append(theSegmentData)
        return theData_feats



    def getAllFBs(self):
        print('getting FilterBanks for train data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rgetting FilterBanks %d%%" % int(100*b/(self.num_batches)))
            self.get_FBs_for_batch(self.data_train_feats_path, self.files_train_path, b)
        sys.stdout.write("\r")
        print('getting FilterBanks for test data ...')
        for b in range(self.num_batches):
            sys.stdout.write("\rgetting FilterBanks %d%%" % int(100*b/(self.num_batches)))
            self.get_FBs_for_batch(self.data_test_feats_path, self.files_test_path, b)
        sys.stdout.write("\r")

    def get_FBs_for_batch(self, path, files_path, batch):
        if not os.path.isfile(path + self.batch_append(batch)):
            with open(files_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                files = pickle.load(fp)
                Data_feats = self.getFBs(files)
                with open(path + self.batch_append(batch), "wb") as fp:   #Pickling
                    pickle.dump(Data_feats, fp)

    def getFBs(self, theData, context_size=4):
        theData_feats = []
        for i,utterance in enumerate(theData):
            theSegmentData = []
            fbs = []
            fb_norms = []
            labels = []
            fbs = python_speech_features.base.fbank(utterance, winlen=self.win_length, winstep=self.win_shift, nfilt=40, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, winfunc=np.hamming)
            fbs2 = []
            for i,fb in enumerate(fbs[0]):
                fb2 = np.append(fbs[0][i],fbs[1][i])
                fbs2.append(fb2)
            fbs = fbs2
            fb_norms = [self.normalize(fb) for fb in fbs]
            delta1 = python_speech_features.base.delta(fbs, context_size)
            delta1_norms = [self.normalize(delta) for delta in delta1]
            delta2 = python_speech_features.base.delta(delta1, context_size)
            delta2_norms = [self.normalize(delta) for delta in delta2]
            for i in range(len(fbs)):
                feats = np.append(fb_norms[i], np.append(delta1_norms[i],delta2_norms[i]))
                theSegmentData.append(feats)
            theData_feats.append(theSegmentData)
        return theData_feats


    def normalize(self, feats):
        result = feats
        if feats.std(axis=0) != 0: result = (feats - feats.mean(axis=0)) / feats.std(axis=0)
        return result


    def load_dictionary(self, path):
        with open(path, "rb") as fp:   # Unpickling
            self.dictionary = pickle.load(fp)
        print('number of lexs:', len(self.dictionary))

    def load_lexicon(self, path):
        with open(path, "rb") as fp:   # Unpickling
            self.lexicon = pickle.load(fp)
        print('number of lexicon:', len(self.lexicon))

    def get_lexs_from_data(self, ignored='NOTHING'):
        if os.path.isfile(self.lexicon_path):
            with open(self.lexicon_path, "rb") as fp:   # Unpickling
                self.lexicon = pickle.load(fp)
        else:
            lexs_files = self.txts_train
            Files_lbls = []
            for i in range(len(lexs_files)):
                line = open(lexs_files[i]).read()
                labels = line.split()
                Files_lbls.append(labels)
            self.lexicon = []
            for labels in Files_lbls:
                for label in labels:
                    if not label in self.lexicon:
                        if label == ignored: continue
                        self.lexicon.append(label)
            with open(self.lexicon_path, "wb") as fp:   #Pickling
                pickle.dump(self.lexicon, fp)
        print('number of lexs:', len(self.lexicon))

    def transferlexs(self, theData):
        theRData = []
        for utterance in theData:
            theSegmentData = []
            for segment in utterance:
                label = segment
                if not label in self.lexicon: continue
                index = self.lexicon.index(label)
                theSegmentData.append(index)
            theRData.append(theSegmentData)
        return theRData


    def get_input_size(self, kmeans=False):
        batch = 0
        path = self.inputs_train_path
        if kmeans: path = path + '_k'
        with open(path + self.batch_append(batch), "rb") as fp:   # Unpickling
            data_train = pickle.load(fp)
        input_size = len(data_train[0][0])
        print('input_size:',input_size)
        return input_size


    def setupInputsForModel(self, feat_type='mfcc', context=0, over_ride=False):
        print('making data ready for use...')
        self.context = context
        for batch in range(self.num_batches):
            sys.stdout.write("\rProcessing data %d%%" % int(100*batch/self.num_batches))
            if os.path.isfile(self.outputs_test_path + self.batch_append(batch)) and (not over_ride): continue
            if feat_type.lower() == 'mfcc':
                with open(self.data_train_feats_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    data_train = pickle.load(fp)
                with open(self.data_test_feats_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    data_test = pickle.load(fp)
            else:
                with open(self.data_train_normalized_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    data_train = pickle.load(fp)
                with open(self.data_test_normalized_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    data_test = pickle.load(fp)
            with open(self.lexs_train_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                lexs_train = self.transferlexs(pickle.load(fp))
            with open(self.lexs_test_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                lexs_test = self.transferlexs(pickle.load(fp))

            inputs, outputs = self.transform_data(data_train, lexs_train)
            # inputs, outputs = self.shuffle_data(inputs, outputs)
            inputs_test, outputs_test = self.transform_data(data_test, lexs_test)

            with open(self.inputs_train_path + self.batch_append(batch), "wb") as fp:   #Pickling
                pickle.dump(inputs, fp)
            with open(self.outputs_train_path + self.batch_append(batch), "wb") as fp:   #Pickling
                pickle.dump(outputs, fp)
            with open(self.inputs_test_path + self.batch_append(batch), "wb") as fp:   #Pickling
                pickle.dump(inputs_test, fp)
            with open(self.outputs_test_path + self.batch_append(batch), "wb") as fp:   #Pickling
                pickle.dump(outputs_test, fp)
        sys.stdout.write("\rdata ready for neural net\n")

    def transform_data(self, input_data, output_data):
        inputs = []; outputs = [];
        for i in range(len(input_data)):
            data = self.get_context_on_utter(input_data[i])
            # data = [np.array(data).astype(np.float32) for data in data_train[i]]
            inputs.append(data)
            # outputs.append(np.array([data[1] for data in data_train[i]]).astype(np.int64))
            outputs.append(np.array(output_data[i]).astype(np.int64))
            # outputs.append(np.array(lexs_train[i]).astype(np.int64))
        return np.array(inputs), np.array(outputs)

    def get_context_on_utter(self, utter):
        result = []
        for index in range(len(utter)):
            context = []
            for i in range(-self.context,self.context+1):
                j = i+index
                if j < 0: j = 0
                if j >= len(utter): j = len(utter)-1
                context += list(utter[j])
            result.append(np.array(context).astype(np.float32))
        return result

    def shuffle_data(self, inputs, outputs):
        indexes = random.sample(range(len(inputs)), len(inputs))
        return inputs[indexes], outputs[indexes]



    def kmeans_fit(self, num_clusters):
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
        for batch in range(self.num_batches):
            sys.stdout.write("\rFitting kmeans %d%%" % int(100*batch/self.num_batches))
            with open(self.inputs_train_path + self.batch_append(batch), "rb") as fp:   # Unpickling
                data_train = pickle.load(fp)
                for utter in data_train:
                    self.kmeans.partial_fit(utter)
                    # print(self.kmeans.predict(utter))
        sys.stdout.write("\rKmeans ready for use")

    def setupInputsForKmeans(self, over_ride=False):
        print('Making kmeans data ready for use...')
        for batch in range(self.num_batches):
            sys.stdout.write("\rProcessing data %d%%" % int(100*batch/self.num_batches))
            if os.path.isfile(self.inputs_train_path + self.batch_append(batch)) and (not over_ride): continue

            for path in [self.inputs_train_path, self.inputs_test_path]:
                inputs = []
                with open(path + self.batch_append(batch), "rb") as fp:   # Unpickling
                    theData = pickle.load(fp)
                    for i,utter in enumerate(theData):
                        utt = []
                        # print(len(utter))
                        for j,segment in enumerate(utter):
                            seg = self.segment_kmeans(segment)
                            utt.append(np.append(segment,seg))
                        # print(len(utt))
                        inputs.append(utt)
                with open(path + '_k' + self.batch_append(batch), "wb") as fp:   #Pickling
                    pickle.dump(inputs, fp)
        sys.stdout.write("\rKmeans data ready for neural net\n")

    def dist(self, a, b):
        return np.sqrt(((a-b)**2).sum())

    def segment_kmeans(self, segment):
        cntrs = self.kmeans.cluster_centers_
        result = np.zeros(len(cntrs))
        Zs = []
        for _, c in enumerate(cntrs):
            z = self.dist(segment,c)
            Zs.append(z)
        miu = np.mean(Zs,0)
        for i, c in enumerate(cntrs):
            result[i] = np.max([0, miu-Zs[i]])
        return result
