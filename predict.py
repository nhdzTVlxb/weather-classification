#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===================== GPU 设置 =====================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ===================== 配置 =====================
test_data_dir = '/home/cyp/weather/dataset/alien_test'  # 测试集图片目录
test_csv_path = '/home/cyp/weather/dataset/test.csv'            # 答案文件
model_path = '/home/cyp/weather/model/model/resnet50_long_648_weights.h5' # 模型路径
img_size = 648

# 类别名称（根据你的训练集顺序）
class_names = ['Cloud', 'Foggy', 'Rain', 'Shine', 'Sunrise']

# ===================== 加载模型 =====================
print("加载模型中...")
model = load_model(model_path)

# ===================== 加载答案 =====================
df = pd.read_csv(test_csv_path)
image_to_label = dict(zip(df['Image_id'], df['labels']))

# ===================== 预测 =====================
print("预测中...")
results = []

for img_name in df['Image_id']:
    img_path = os.path.join(test_data_dir, img_name)
    
    # 加载并预处理图片
    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 预测
    pred = model.predict(img_array, verbose=0)
    pred_label = np.argmax(pred)
    
    results.append({
        'Image_id': img_name,
        'True_Label': image_to_label[img_name],
        'Pred_Label': pred_label,
        'True_Class': class_names[image_to_label[img_name]],
        'Pred_Class': class_names[pred_label]
    })

# ===================== 计算准确率 =====================
results_df = pd.DataFrame(results)
correct = results_df[results_df['True_Label'] == results_df['Pred_Label']]
wrong = results_df[results_df['True_Label'] != results_df['Pred_Label']]

accuracy = len(correct) / len(results_df)

print(f"\n{'='*50}")
print(f"正确率: {accuracy:.2%} ({len(correct)}/{len(results_df)})")
print(f"{'='*50}")

# ===================== 输出错误样本 =====================
if len(wrong) > 0:
    print(f"\n预测错误的样本 ({len(wrong)}个):")
    print("-" * 60)
    for _, row in wrong.iterrows():
        print(f"  {row['Image_id']}: 真实={row['True_Class']}({row['True_Label']}) -> 预测={row['Pred_Class']}({row['Pred_Label']})")
else:
    print("\n🎉 全部正确！")