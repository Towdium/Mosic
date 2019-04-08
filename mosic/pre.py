import os, copy, tqdm, sys
import shutil
import numpy as np
import pandas as pd
from scipy import misc
from collections import defaultdict
import random
import librosa as rosa

'''
This file contains steps to fetch and pre-process the dataset.
It relies on wget, unzip, ffmpeg to execute, please have them installed.
You can simply call this script directly with the args:

cache: 1) download dataset; 2) unzip files; 3) convert wav audio to mp3
generate: load cached data to generate audio info and spectrograms, then sort and pack
delete: delete cache files

When called with no arg, it will run all the steps
'''


def cache():
    os.makedirs('data/tmp/wav', exist_ok=True)
    os.system('wget http://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip -P data/tmp')
    os.system('wget http://cvml.unige.ch/databases/DEAM/DEAM_audio.zip -P data/tmp')

    dir_tmp = 'data/tmp'
    for f in os.listdir(dir_tmp):
        if f.endswith('.zip'):
            os.system('unzip {0}/{1} -d {0}'.format(dir_tmp, f))

    dir_audio = "data/tmp/MEMD_audio"
    for f in os.listdir(dir_audio):
        id = int(f[:-4])
        if id <= 2000:
            os.system('ffmpeg -i "{1}/{0}.mp3" "{2}/wav/{0}.wav"'.format(id, dir_audio, dir_tmp))


def generate():
    data = defaultdict(dict)
    output = defaultdict(lambda: defaultdict(list))
    keys = ['valence_mean', 'valence_std', 'arousal_mean', 'arousal_std',
            'arousal_cont', 'valence_cont', 'amp_stft', 'amp_cqt', 'pha_stft', 'pha_cqt', 'song_id']
    dir_out = 'data'
    dir_avg = "data/tmp/annotations/annotations averaged per song" \
                     "/song_level/static_annotations_averaged_songs_1_2000.csv"
    dir_con_aro = "data/tmp/annotations/annotations averaged per song/" \
                        "dynamic (per second annotations)/arousal.csv"
    dir_con_val = "data/tmp/annotations/annotations averaged per song/" \
                        "dynamic (per second annotations)/valence.csv"

    avg = pd.read_csv(dir_avg)
    avg.rename(columns=lambda x: x.strip(), inplace=True)
    for _, i in avg.iterrows():
        record = data[int(i['song_id'])]
        for j in keys[:4]:
            record[j] = i[j]

    for file, name in zip((dir_con_aro, dir_con_val), keys[4:6]):
        csv = pd.read_csv(file)
        for _, i in csv.iterrows():
            id = int(i['song_id'])
            if id > 2000:
                continue
            record = data[id]
            record[name] = i.values[1:61]

    dir_wav = 'data/tmp/wav'
    print('Generating spectrum')
    for f in tqdm.tqdm(os.listdir(dir_wav)):
        if f.endswith('.wav'):
            index = int(f[:-4])
            y, sr = rosa.load(os.path.join(dir_wav, f))
            cqt = rosa.core.cqt(y=y, sr=sr, hop_length=2048, n_bins=48 * 8, bins_per_octave=48)
            amp_cqt = rosa.amplitude_to_db(np.abs(cqt))[::-1, :]
            pha_cqt = np.angle(cqt)[::-1, :]
            stft = rosa.core.stft(y=y, n_fft=1024, hop_length=2048)
            amp_stft = rosa.amplitude_to_db(np.abs(stft))[::-1, :]
            pha_stft = np.angle(stft)[::-1, :]
            misc.imsave('dump/amp-cqt/' + str(index) + '.bmp', amp_cqt)
            misc.imsave('dump/amp-stft/' + str(index) + '.bmp', amp_stft)
            misc.imsave('dump/pha-cqt/' + str(index) + '.bmp', pha_cqt)
            misc.imsave('dump/pha-stft/' + str(index) + '.bmp', pha_stft)
            record = data[index]
            record['amp_stft'] = amp_stft
            record['amp_cqt'] = amp_cqt
            record['pha_stft'] = pha_stft
            record['pha_cqt'] = pha_cqt

    print('Indexing data')
    dest = ['test', 'valid', 'train']
    data = [{'song_id': k, **v} for k, v in data.items()]
    data = sorted(data, key=lambda i: i['valence_mean'] + i['arousal_mean'])
    buffer = defaultdict(list)
    for i, v in enumerate(data):  # iterate through all items
        if 'amp_cqt' not in v:
            continue
        buffer[dest[min(i % 8, 2)]].append(v)  # distribute buffers for different sets
    for i in dest:
        random.shuffle(buffer[i])
        container = output[i]
        for j in buffer[i]:
            container['song_id'].append(j['song_id'])
            for k in keys:
                container[k].append(j[k])  # split data rows to column index

    print('Writing data files')
    for i in [1, 2, 4]:  # for each size
        for k, v in output.items():  # iterate each set
            if len(v) == 0:  # skip empty set in testing
                continue
            o = copy.copy(v)
            for s in keys[-5:-1]:
                size = (int(v[s][0].shape[0] / i), int(v[s][0].shape[1] / i))  # size of one spec class
                o[s] = [misc.imresize(j, size, interp='bilinear') for j in v[s]]  # resize each spec
            np.savez(os.path.join(dir_out, 'deam-{1}-{0}'.format(k, i)), **o)


def clean():
    shutil.rmtree('data/tmp')


if len(sys.argv) == 2:
    if sys.argv[1] in locals():
        locals()[sys.argv[1]]()
    else:
        print('Function not found')
else:
    cache()
    generate()
    clean()
