import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from datetime import datetime 
import pickle

#音声ファイルからMFCC特徴量を抽出する関数
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

#音声データセットのメタデータを読み込む
metadata=pd.read_csv('UrbanSound8K.csv')

#全データセットから特徴量を抽出
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join('fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

#抽出した特徴量とラベルをDataFrameに格納
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
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

#モデルのトレーニング開始
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

#トレーニング時間の計測
duration = datetime.now() - start
print("Training completed in time: ", duration)

# モデルの保存
model.save('saved_models/audio_classification.h5')

# ラベルエンコーダの保存
with open('label_encoder.pkl', 'wb') as le_dump_file:
    pickle.dump(labelencoder, le_dump_file)

print("Model and label encoder saved successfully!")

