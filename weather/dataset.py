#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
from shutil import copyfile


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import pathlib


# In[4]:

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('/home/weather/dataset/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(10, 8))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes of Dataset')
plt.show()


# In[5]:


image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

for i in image_folder:
    sample_images = list(pathlib.Path('/home/weather/dataset/'+i+'/').glob('*'))  
    np.random.seed(42)
    
    if len(sample_images) >= 100:
        rand_imgs = np.random.choice(sample_images, size=100, replace=False)
    else:
        rand_imgs = np.random.choice(sample_images, size=len(sample_images), replace=False)
        print(f"警告: {i} 类别只有 {len(sample_images)} 张图片，少于100张")

    shapes = []
    for img in rand_imgs:
        try:
            shapes.append(imread(str(img)).shape)
        except Exception as e:
            print(f"读取图片失败 {img}: {e}")
            continue
    
    if shapes:  # 只有当有成功读取的图片时才绘制
        shapes = pd.DataFrame().assign(X=pd.Series(shapes).map(lambda s: s[0]), Y=pd.Series(shapes).map(lambda s: s[1]))
        
        plt.figure(figsize=(12, 8))
        sns.set_context("notebook", font_scale=1.5)
        sns.kdeplot(shapes['X'], bw_method=75)
        sns.kdeplot(shapes['Y'], bw_method=75)
        plt.title('Distribution of {}_image Sizes'.format(i))
        ax = plt.gca()
        ax.set_xlim(0, ax.get_xlim()[1])
        plt.show()
    else:
        print(f"无法读取 {i} 类别的图片")


# In[6]:


# 创建目录 - 使用 makedirs 以确保父目录存在
BASE_DIR = '/home/weather/weather_pred'

try:
    # 创建所有训练子目录
    os.makedirs(os.path.join(BASE_DIR, 'Data/training/cloudy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/training/foggy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/training/rainy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/training/shine'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/training/sunrise'), exist_ok=True)
    
    # 创建所有验证子目录
    os.makedirs(os.path.join(BASE_DIR, 'Data/validation/cloudy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/validation/foggy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/validation/rainy'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/validation/shine'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'Data/validation/sunrise'), exist_ok=True)
    
    print("所有目录创建成功")
    
except Exception as e:
    print(f"创建目录时出错: {e}")
    print(f"请检查权限或路径: {BASE_DIR}")


# In[7]:


def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    """分割数据集到训练和验证目录"""
    
    # 检查源目录是否存在
    if not os.path.exists(SOURCE):
        print(f"错误: 源目录不存在: {SOURCE}")
        return False
    
    # 检查目标目录是否存在，如果不存在则创建
    for dir_path in [TRAINING, VALIDATION]:
        if not os.path.exists(dir_path):
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # 收集所有有效的图片文件
    files = []
    for filename in os.listdir(SOURCE):
        file_path = os.path.join(SOURCE, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            # 只处理常见的图片格式
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                files.append(filename)
        else:
            if os.path.isfile(file_path):
                print(f"{filename} 是空文件，跳过")
            else:
                print(f"{filename} 不是文件，跳过")

    if len(files) == 0:
        print(f"警告: 在 {SOURCE} 中没有找到有效的图片文件")
        return False

    # 计算训练和验证集的大小
    training_length = int(len(files) * SPLIT_SIZE)
    valid_length = len(files) - training_length
    
    # 随机打乱文件列表
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    print(f"\n处理目录: {SOURCE}")
    print(f"总文件数: {len(files)}")
    print(f"训练集: {len(training_set)} 个文件")
    print(f"验证集: {len(valid_set)} 个文件")

    # 复制训练集文件
    success_count = 0
    for filename in training_set:
        source_file = os.path.join(SOURCE, filename)
        dest_file = os.path.join(TRAINING, filename)
        try:
            copyfile(source_file, dest_file)
            success_count += 1
        except Exception as e:
            print(f"复制文件失败 {filename}: {e}")

    # 复制验证集文件
    for filename in valid_set:
        source_file = os.path.join(SOURCE, filename)
        dest_file = os.path.join(VALIDATION, filename)
        try:
            copyfile(source_file, dest_file)
            success_count += 1
        except Exception as e:
            print(f"复制文件失败 {filename}: {e}")
    
    print(f"成功复制 {success_count} 个文件")
    return True


# In[8]:


# 设置源目录和目标目录
CLOUDY_SOURCE_DIR = '/home/weather/dataset/cloudy/'
TRAINING_CLOUDY_DIR = '/home/weather/weather_pred/Data/training/cloudy/'
VALID_CLOUDY_DIR = '/home/weather/weather_pred/Data/validation/cloudy/'

FOGGY_SOURCE_DIR = '/home/weather/dataset/foggy/'
TRAINING_FOGGY_DIR = '/home/weather/weather_pred/Data/training/foggy/'
VALID_FOGGY_DIR = '/home/weather/weather_pred/Data/validation/foggy/'

RAINY_SOURCE_DIR = '/home/weather/dataset/rainy/'
TRAINING_RAINY_DIR = '/home/weather/weather_pred/Data/training/rainy/'
VALID_RAINY_DIR = '/home/weather/weather_pred/Data/validation/rainy/'

SHINE_SOURCE_DIR = '/home/weather/dataset/shine/'
TRAINING_SHINE_DIR = '/home/weather/weather_pred/Data/training/shine/'
VALID_SHINE_DIR = '/home/weather/weather_pred/Data/validation/shine/'

SUNRISE_SOURCE_DIR = '/home/weather/dataset/sunrise/'
TRAINING_SUNRISE_DIR = '/home/weather/weather_pred/Data/training/sunrise/'
VALID_SUNRISE_DIR = '/home/weather/weather_pred/Data/validation/sunrise/'


# In[9]:


# 设置分割比例
split_size = 0.85

# 执行数据分割
print("开始数据分割...")
print("=" * 50)

split_data(CLOUDY_SOURCE_DIR, TRAINING_CLOUDY_DIR, VALID_CLOUDY_DIR, split_size)
split_data(FOGGY_SOURCE_DIR, TRAINING_FOGGY_DIR, VALID_FOGGY_DIR, split_size)
split_data(RAINY_SOURCE_DIR, TRAINING_RAINY_DIR, VALID_RAINY_DIR, split_size)
split_data(SHINE_SOURCE_DIR, TRAINING_SHINE_DIR, VALID_SHINE_DIR, split_size)
split_data(SUNRISE_SOURCE_DIR, TRAINING_SUNRISE_DIR, VALID_SUNRISE_DIR, split_size)

print("=" * 50)
print("数据分割完成！")


# In[10]:


# 显示训练集分布
image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
nimgs = {}
for i in image_folder:
    train_path = f'/home/weather/weather_pred/Data/training/{i}/'
    if os.path.exists(train_path):
        nimages = len([f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        nimgs[i] = nimages
    else:
        nimgs[i] = 0
        print(f"警告: 训练目录不存在 {train_path}")

plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Training Dataset')
plt.show()


# In[11]:


# 打印训练集详细信息
print("\n训练集统计:")
for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
    train_path = f'/home/weather/weather_pred/Data/training/{i}/'
    if os.path.exists(train_path):
        count = len([f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f'Training {i} images are: {count}')
    else:
        print(f'Training {i} directory not found')


# In[12]:


# 显示验证集分布
nimgs = {}
for i in image_folder:
    valid_path = f'/home/weather/weather_pred/Data/validation/{i}/'
    if os.path.exists(valid_path):
        nimages = len([f for f in os.listdir(valid_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        nimgs[i] = nimages
    else:
        nimgs[i] = 0
        print(f"警告: 验证目录不存在 {valid_path}")

plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Validation Dataset')
plt.show()


# In[13]:


# 打印验证集详细信息
print("\n验证集统计:")
for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
    valid_path = f'/home/weather/weather_pred/Data/validation/{i}/'
    if os.path.exists(valid_path):
        count = len([f for f in os.listdir(valid_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f'Valid {i} images are: {count}')
    else:
        print(f'Valid {i} directory not found')


print("\n数据预处理完成！")