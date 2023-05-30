#use_model.py
import librosa
import numpy as np
import tensorflow as tf
import pickle

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

#モデルの読み込み
new_model = tf.keras.models.load_model('saved_models/audio_classification.h5')

#ラベルエンコーダの読み込み
with open('label_encoder.pkl', 'rb') as le_dump_file:
    labelencoder = pickle.load(le_dump_file)

#新しい音声ファイルから特徴量を抽出
filename="fold5/100852-0-0-1.wav"
mfccs_scaled_features = features_extractor(filename)

#特徴量をモデルに適した形状に整形
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

#モデルを用いて予測
predicted_label=new_model.predict(mfccs_scaled_features)

#予測ラベルを人間が理解可能な形にデコード
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
print(prediction_class)
