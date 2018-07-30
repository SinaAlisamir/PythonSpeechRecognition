#!/usr/bin/python3
import os
import glob
import subprocess
import pickle
from shutil import copyfile

def main():
    path = '/home/sina/Documents/TIMIT/TIMIT'
    dic1 = copy_all_to_one(path+'/TRAIN', './train/')
    dic2 = copy_all_to_one(path+'/TEST', './test/')
    dic3 = copy_all_to_one(path+'/TEST_CORE', './test_core/')

    make_files('./train/', 'train.PHN', 'train.WRD')
    make_files('./test/', 'test.PHN', 'test.WRD')
    make_files('./test_core/', 'test_core.PHN', 'test_core.WRD')

    dictionary = dic1
    for word in dic2:
        if not word in dictionary: dictionary.append(word)
    for word in dic3:
        if not word in dictionary: dictionary.append(word)
    content = " ".join(map(str, dictionary))
    with open('dictionary.txt', 'w') as f:
        f.write(content)
    with open('dictionary', "wb") as fp:   #Pickling
        pickle.dump(dictionary, fp)
    create_lexicon()


def copy_all_to_one(path, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    aud_files = glob.glob(path+"/**/SX*.WAV", recursive=True) + glob.glob(path+"/**/SI*.WAV", recursive=True)
    txt_files = glob.glob(path+"/**/SX*.PHN", recursive=True) + glob.glob(path+"/**/SI*.PHN", recursive=True)
    wrd_files = glob.glob(path+"/**/SX*.WRD", recursive=True) + glob.glob(path+"/**/SI*.WRD", recursive=True)
    print(len(aud_files), len(txt_files))
    counters={};
    for audio in aud_files: audio_name = audio.split('/')[-1][:-4]; counters[audio_name] = 0
    for audio in aud_files:
        audio_name = audio.split('/')[-1][:-4]
        counters[audio_name] += 1
        if counters[audio_name] > 1:
            audio_name += '_'+str(counters[audio_name])
        copyfile(audio, save_dir+audio_name+'.wav')
    counters={};
    for txt in txt_files: file_name = txt.split('/')[-1][:-4]; counters[file_name] = 0
    for txt in txt_files:
        file_name = txt.split('/')[-1][:-4]
        lines = [line.rstrip('\n') for line in open(txt)]
        lines = [line.split() for line in lines]
        # print(lines)
        transcripts = []
        for line in lines:
            transcript = line[2]
            if not transcript in lexicon:
                if transcript == 'q': continue
                label = folder[transcript]
            else:
                label = transcript
            transcripts.append(label)
        content = " ".join(map(str, transcripts))
        counters[file_name] += 1
        if counters[file_name] > 1:
            file_name += '_'+str(counters[file_name])
        with open(save_dir+file_name+'.txt', 'w') as f:
            f.write(content)

    counters={};
    dictionary = [];
    for wrd in wrd_files: file_name = wrd.split('/')[-1][:-4]; counters[file_name] = 0
    for wrd in wrd_files:
        file_name = wrd.split('/')[-1][:-4]
        lines = [line.rstrip('\n') for line in open(wrd)]
        lines = [line.split() for line in lines]
        # print(lines)
        transcripts = []
        for line in lines:
            label = line[2]
            if not label in dictionary: dictionary.append(label)
            transcripts.append(label)
        content = " ".join(map(str, transcripts))
        counters[file_name] += 1
        if counters[file_name] > 1:
            file_name += '_'+str(counters[file_name])
        with open(save_dir+file_name+'.WRD', 'w') as f:
            f.write(content)

    return dictionary


def create_lexicon():
    with open('lexicon', "wb") as fp:   #Pickling
        pickle.dump(lexicon, fp)
    content = " ".join(map(str, lexicon))
    with open('lexicon.txt', 'w') as f:
        f.write(content)

lexicon = ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey',
                'ay', 'oy', 'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 'n',
                'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k',
                'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil']

folder = {'ao':'aa', 'ax':'ah', 'ax-h':'ah', 'axr':'er', 'hv':'hh', 'ix':'ih',
              'el':'l', 'em':'m', 'en':'n', 'nx':'n', 'eng':'ng', 'zh':'sh', 'ux':'uw',
              'pcl':'sil', 'tcl':'sil', 'kcl':'sil', 'bcl':'sil', 'dcl':'sil', 'gcl':'sil',
              'h#':'sil', '#h':'sil', 'pau':'sil', 'epi':'sil'}


def make_files(path, phn_path, wrd_path):
    txt_files = glob.glob(path+"/**/*.txt", recursive=True)
    wrd_files = []
    for txt in txt_files: wrd_files.append(txt[:-3] + 'WRD')
    txts = []; wrds = []
    for i in range(len(txt_files)):
        txt = open(txt_files[i]).read() + ' .'
        wrd = open(wrd_files[i]).read() + ' .'
        if not wrd in wrds:
            txts.append(txt)
            wrds.append(wrd)
    content_phn = "\n".join(map(str, txts))
    content_wrd = "\n".join(map(str, wrds))
    with open(phn_path, 'w') as f:
        f.write(content_phn)
    with open(wrd_path, 'w') as f:
        f.write(content_wrd)


if __name__ == "__main__":
    main()
