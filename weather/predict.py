#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# config
test_data_dirs = [
    '/home/weather/dataset/alien_test',
    '/home/weather/dataset/alien_test_foggy',
    '/home/weather/dataset/alien_test_rain'
]  
test_csv_path = '/home/weather/dataset/test2.csv'           
model_path = '/home/weather/model/model/resnet50_long_512_weights.h5' 
img_size = 512

# class
class_names = ['Cloud', 'Foggy', 'Rain', 'Shine', 'Sunrise']

# model
print("model loading...")
model = load_model(model_path)

# 读取test.csv获取图片标签
df = pd.read_csv(test_csv_path)
image_to_label = dict(zip(df['Image_id'], df['labels']))

# predict
print("predicting...")
results = []

for test_data_dir in test_data_dirs:  
    if not os.path.exists(test_data_dir):
        print(f"Warning: {test_data_dir} not found, skipping...")
        continue
    
    # 获取当前文件夹中的所有图片
    for img_name in os.listdir(test_data_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(test_data_dir, img_name)  
        
        # 检查图片是否在test.csv中
        if img_name not in image_to_label:
            print(f"  Warning: {img_name} not found in test.csv, skipping...")
            continue
        
        # 加载并预处理图片
        img = load_img(img_path, target_size=(img_size, img_size))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # predict
        pred = model.predict(img_array, verbose=0)
        pred_label = np.argmax(pred)
        
        results.append({
            'Image_id': img_name,
            'True_Label': image_to_label[img_name],
            'Pred_Label': pred_label,
            'True_Class': class_names[image_to_label[img_name]],
            'Pred_Class': class_names[pred_label]
        })

# acc
results_df = pd.DataFrame(results)
correct = results_df[results_df['True_Label'] == results_df['Pred_Label']]
wrong = results_df[results_df['True_Label'] != results_df['Pred_Label']]

accuracy = len(correct) / len(results_df) if len(results_df) > 0 else 0

print(f"\n{'='*50}")
print(f"正确率: {accuracy:.2%} ({len(correct)}/{len(results_df)})")
print(f"{'='*50}")

# output error
if len(wrong) > 0:
    print(f"\n预测错误的样本 ({len(wrong)}个):")
    print("-" * 60)
    for _, row in wrong.iterrows():
        print(f"  {row['Image_id']}: 真实={row['True_Class']}({row['True_Label']}) -> 预测={row['Pred_Class']}({row['Pred_Label']})")
else:
    print("\n 全部正确！")