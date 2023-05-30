import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt
import resampy
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

#音声ファイルからMFCC特徴量を抽出する関数
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

os.environ['KAGGLE_CONFIG_DIR'] = "./content"

#音声ファイルの読み込みと表示
file_name='./fold5/100263-2-0-121.wav'
audio_data, sampling_rate = librosa.load(file_name)
librosa.display.waveshow(audio_data,sr=sampling_rate)
ipd.Audio(file_name)
print(audio_data)

#音声データセットのメタデータを読み込む
audio_dataset_path='./content/'
metadata=pd.read_csv('UrbanSound8K.csv')
print(metadata.head())
print(metadata['class'].value_counts())

#音声データからMFCCを抽出
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
print(mfccs)

#全データセットから特徴量を抽出
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
    
#抽出した特徴量とラベルをDataFrameに格納
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(10)
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

#ラベルをエンコード
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

#データセットをトレーニングセットとテストセットに分割
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

num_labels=y.shape[1]
#ニューラルネットワークモデルの構築
model=Sequential()
#レイヤー(4層)
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

#モデルのコンパイル
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

#バッチサイズとエポック数の設定
num_epochs = 200
num_batch_size = 32

#モデルの保存設定
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

#モデルトレーニング開始
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

#トレーニング時間計測
duration = datetime.now() - start
print("Training completed in time: ", duration)

#新しい音声ファイルから特徴量を抽出
filename="fold8/103076-3-0-0.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)

# 特徴量をモデルに適した形状に整形
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)

# モデルを用いて予測
predicted_label=model.predict(mfccs_scaled_features)
print(predicted_label)

# 予測ラベルを人間が理解可能な形にデコード
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
prediction_class