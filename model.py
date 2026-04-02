#!/usr/bin/env python
# coding: utf-8

import os
import urllib.request
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    VGG16, VGG19,
    InceptionV3, InceptionResNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    MobileNetV2, MobileNetV3Large, MobileNetV3Small,
    NASNetMobile, NASNetLarge
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess

# TensorFlow Hub 的权重通常会自动下载，不需要手动指定URL
# Keras 内置的权重会在首次使用时自动下载到 ~/.keras/models/

# 如果确实需要手动管理权重，可以定义备用下载链接
model_urls = {
    # ResNet 系列 (TensorFlow/Keras)
    'resnet50': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'resnet101': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'resnet152': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5',
    
    # VGG 系列
    'vgg16': 'https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'vgg19': 'https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    
    # Inception 系列
    'inception_v3': 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'inception_resnet_v2': 'https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    
    # DenseNet 系列
    'densenet121': 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'densenet169': 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'densenet201': 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
    
    # EfficientNet 系列
    'efficientnet_b0': 'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
    'efficientnet_b1': 'https://storage.googleapis.com/keras-applications/efficientnetb1_notop.h5',
    'efficientnet_b2': 'https://storage.googleapis.com/keras-applications/efficientnetb2_notop.h5',
    'efficientnet_b3': 'https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5',
    'efficientnet_b4': 'https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5',
    
    # MobileNet 系列
    'mobilenet_v2': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
    'mobilenet_v3_large': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_large_224_1.0_float_no_top.h5',
}

def download_weight(url, save_path):
    """下载预训练权重文件"""
    print(f"正在下载预训练权重: {url}")
    print(f"保存路径: {save_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, save_path, reporthook=download_progress)
        print(f"\n✅ 下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("将使用 Keras 默认方式加载权重（会自动下载）")
        return False

def download_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        if percent % 10 == 0:  # 每10%显示一次
            print(f"\r下载进度: {percent:.1f}%", end='')

def get_preprocess_function(model_name):
    """根据模型名称返回对应的预处理函数"""
    preprocess_functions = {
        'resnet50': resnet_preprocess,
        'resnet101': resnet_preprocess,
        'resnet152': resnet_preprocess,
        'vgg16': vgg_preprocess,
        'vgg19': vgg_preprocess,
        'inception_v3': inception_preprocess,
        'inception_resnet_v2': inception_preprocess,
        'densenet121': densenet_preprocess,
        'densenet169': densenet_preprocess,
        'densenet201': densenet_preprocess,
        'efficientnet_b0': efficientnet_preprocess,
        'efficientnet_b1': efficientnet_preprocess,
        'efficientnet_b2': efficientnet_preprocess,
        'efficientnet_b3': efficientnet_preprocess,
        'efficientnet_b4': efficientnet_preprocess,
        'mobilenet_v2': mobilenet_preprocess,
        'mobilenet_v3_large': mobilenet_preprocess,
        'mobilenet_v3_small': mobilenet_preprocess,
        'nasnet_mobile': nasnet_preprocess,
        'nasnet_large': nasnet_preprocess,
    }
    return preprocess_functions.get(model_name, resnet_preprocess)

def get_input_size(model_name):
    """获取模型默认输入尺寸"""
    input_sizes = {
        'resnet50': 224,
        'resnet101': 224,
        'resnet152': 224,
        'vgg16': 224,
        'vgg19': 224,
        'inception_v3': 299,
        'inception_resnet_v2': 299,
        'densenet121': 224,
        'densenet169': 224,
        'densenet201': 224,
        'efficientnet_b0': 224,
        'efficientnet_b1': 240,
        'efficientnet_b2': 260,
        'efficientnet_b3': 300,
        'efficientnet_b4': 380,
        'mobilenet_v2': 224,
        'mobilenet_v3_large': 224,
        'mobilenet_v3_small': 224,
        'nasnet_mobile': 224,
        'nasnet_large': 331,
    }
    return input_sizes.get(model_name, 224)

def load_backbone(model_name='resnet50', input_shape=None, weights='imagenet', 
                  include_top=False, weight_path=None, verbose=True):
    """
    加载 backbone 模型
    
    参数:
        model_name: 模型名称 (resnet50, resnet101, vgg16, inception_v3, densenet121, efficientnet_b0, etc.)
        input_shape: 输入形状，如果为None则使用默认尺寸
        weights: 'imagenet' 或 None
        include_top: 是否包含全连接层
        weight_path: 自定义权重文件路径
        verbose: 是否打印详细信息
    
    返回:
        base_model: Keras 模型
        preprocess_func: 预处理函数
        input_size: 输入尺寸
    """
    
    # 获取输入尺寸
    if input_shape is None:
        input_size = get_input_size(model_name)
        input_shape = (input_size, input_size, 3)
    else:
        input_size = input_shape[0]
    
    if verbose:
        print(f"加载 backbone: {model_name}")
        print(f"输入形状: {input_shape}")
        print(f"使用预训练权重: {weights}")
    
    # 模型映射
    models = {
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'vgg16': VGG16,
        'vgg19': VGG19,
        'inception_v3': InceptionV3,
        'inception_resnet_v2': InceptionResNetV2,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'efficientnet_b0': EfficientNetB0,
        'efficientnet_b1': EfficientNetB1,
        'efficientnet_b2': EfficientNetB2,
        'efficientnet_b3': EfficientNetB3,
        'efficientnet_b4': EfficientNetB4,
        'mobilenet_v2': MobileNetV2,
        'mobilenet_v3_large': MobileNetV3Large,
        'mobilenet_v3_small': MobileNetV3Small,
        'nasnet_mobile': NASNetMobile,
        'nasnet_large': NASNetLarge,
    }
    
    if model_name not in models:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(models.keys())}")
    
    # 处理权重路径
    if weight_path and os.path.exists(weight_path):
        if verbose:
            print(f"使用本地权重文件: {weight_path}")
        base_model = models[model_name](
            include_top=include_top,
            weights=None,
            input_shape=input_shape
        )
        base_model.load_weights(weight_path)
        if verbose:
            print("✅ 本地权重加载成功")
    else:
        # 如果指定了权重路径但文件不存在，尝试下载
        if weight_path and model_name in model_urls:
            if verbose:
                print(f"本地权重文件不存在，尝试下载: {weight_path}")
            download_weight(model_urls[model_name], weight_path)
            if os.path.exists(weight_path):
                base_model = models[model_name](
                    include_top=include_top,
                    weights=None,
                    input_shape=input_shape
                )
                base_model.load_weights(weight_path)
                if verbose:
                    print("✅ 下载的权重加载成功")
            else:
                if verbose:
                    print("使用 Keras 默认方式加载权重")
                base_model = models[model_name](
                    include_top=include_top,
                    weights=weights,
                    input_shape=input_shape
                )
        else:
            # 使用 Keras 默认方式（会自动下载到 ~/.keras/models/）
            if verbose:
                print("使用 Keras 自动下载的权重")
            base_model = models[model_name](
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )
    
    # 获取预处理函数
    preprocess_func = get_preprocess_function(model_name)
    
    if verbose:
        print(f"✅ 模型加载完成")
        print(f"可训练参数数量: {base_model.count_params():,}")
    
    return base_model, preprocess_func, input_size

# 可用模型列表
AVAILABLE_MODELS = list(model_urls.keys()) + [
    'mobilenet_v3_small', 'nasnet_mobile', 'nasnet_large'
]

def list_available_models():
    """列出所有可用的模型"""
    print("可用的 backbone 模型:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        input_size = get_input_size(model)
        print(f"  {i}. {model:25s} (输入尺寸: {input_size}x{input_size})")

if __name__ == "__main__":
    # 测试代码
    list_available_models()
    
    # 测试加载模型
    print("\n" + "="*50)
    model, preprocess_func, input_size = load_backbone('resnet50', verbose=True)