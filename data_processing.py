import numpy as np
import pandas as pd
import os
import librosa
import warnings
import shutil

from uuid import uuid4
from PIL import Image
from sklearn.preprocessing import minmax_scale

import argparse

"""
データを準備する関数
"""

parser = argparse.ArgumentParser(
    description = "data preprocessing"
)

parser.add_argument("--data_folder", default = '/kaggle/input/birdsong-recognition', type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
args = parser.parse_args()

#===============================
# read_data
#===============================
def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')[['ebird_code', 'filename', 'duration', 'rating']]
    #データを減らす
    train = train.query("rating >= 4")

    return train

#==============================
# resample
#==============================
def resample(df):
    train = pd.DataFrame(columns = ['ebird_code', 'filename', 'duration', 'rating'])
    for bird_name in df.ebird_code.unique():
        rows = df[df.ebird_code == bird_name].sample(10, replace = True)
        train = train.append(rows)
        train.drop_duplicates(subset = ['ebird_code', 'filename'], inplace = True)
        train.reset_index(drop = True, inplace = True)
    
    return train

#===========================================================
# get_sample
#===========================================================
def get_sample(file_name, bird, output_folder):
    try:
        wave_data, wave_rate = librosa.load(file_name)
        wave_data, _ = librosa.effects.trim(wave_data)
        song_sample = []
        sample_length = 5 * wave_rate
        samples_from_file = []
    
        for idx in range(0, len(wave_data), sample_length):
            song_sample = wave_data[idx:idx + sample_length]
            if(len(song_sample) >= sample_length):
                mel = librosa.feature.melspectrogram(song_sample, n_mels = 216)
                db = librosa.power_to_db(mel)
                normalized_db = minmax_scale(db)
                filename = str(uuid4()) + ".jpg" #ランダムでユニークなパスを作成
                db_array = (np.asarray(normalized_db) * 255).astype(np.uint8)
                db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)
                db_image.save(output_folder + '/' + filename)
            
                samples_from_file.append({"song_sample" : "{}".format(filename), "bird" : bird})

        return samples_from_file

    except:
        print(file_name)

#===========================
# create_data
#===========================
def create_data():
    samples_df = pd.DataFrame(columns = ['song_sample', 'bird'])
    sample_list = []

    output_dir = args.output_folder + "/melspectrogram_dataset"
    os.mkdir(output_dir)

    train = read_data(args.data_folder)
    train = resample(train)

    for idx, row in train.iterrows():
        if row.filename == "XC195038.mp3":
            pass
        else:
            audio_file_path = args.data_folder + "/train_audio/" + row.ebird_code + "/" + row.filename
            sample_list += get_sample(audio_file_path, row.ebird_code, output_dir)

    samples_df = pd.DataFrame(sample_list)

    shutil.make_archive(args.output_folder + '/output', 'zip', root_dir = output_dir)
    samples_df.to_csv(args.output_folder + '/sample_df.csv', index = False)

    shutil.rmtree(output_dir)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    create_data()