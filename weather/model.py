#!/usr/bin/env python
# coding: utf-8

import os
import urllib.request

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


MODEL_NAME = 'resnet50'
INPUT_SIZE = 224

model_urls = {
    MODEL_NAME: 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
}

AVAILABLE_MODELS = [MODEL_NAME]


def download_weight(url, save_path):
    """下载预训练权重文件"""
    print(f"正在下载预训练权重: {url}")
    print(f"保存路径: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=download_progress)
        print(f"\n下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("将使用 Keras 默认方式加载权重（会自动下载）")
        return False


def download_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        if percent % 10 == 0:
            print(f"\r下载进度: {percent:.1f}%", end='')


def _validate_model_name(model_name):
    if model_name != MODEL_NAME:
        raise ValueError(f"不支持的模型: {model_name}. 仅支持模型: {MODEL_NAME}")


def get_preprocess_function(model_name=MODEL_NAME):
    """返回 ResNet50 对应的预处理函数"""
    _validate_model_name(model_name)
    return resnet_preprocess


def get_input_size(model_name=MODEL_NAME):
    """获取 ResNet50 默认输入尺寸"""
    _validate_model_name(model_name)
    return INPUT_SIZE


def load_backbone(model_name=MODEL_NAME, input_shape=None, weights='imagenet',
                  include_top=False, weight_path=None, verbose=True):
    """
    加载 ResNet50 backbone 模型

    参数:
        model_name: 模型名称，仅支持 resnet50
        input_shape: 输入形状，如果为 None 则使用默认尺寸
        weights: 'imagenet' 或 None
        include_top: 是否包含全连接层
        weight_path: 自定义权重文件路径
        verbose: 是否打印详细信息

    返回:
        base_model: Keras 模型
        preprocess_func: 预处理函数
        input_size: 输入尺寸
    """
    _validate_model_name(model_name)

    if input_shape is None:
        input_size = get_input_size(model_name)
        input_shape = (input_size, input_size, 3)
    else:
        input_size = input_shape[0]

    if verbose:
        print(f"加载 backbone: {model_name}")
        print(f"输入形状: {input_shape}")
        print(f"使用预训练权重: {weights}")

    if weight_path and os.path.exists(weight_path):
        if verbose:
            print(f"使用本地权重文件: {weight_path}")
        base_model = ResNet50(
            include_top=include_top,
            weights=None,
            input_shape=input_shape
        )
        base_model.load_weights(weight_path)
        if verbose:
            print("本地权重加载成功")
    else:
        if weight_path:
            if verbose:
                print(f"本地权重文件不存在，尝试下载: {weight_path}")
            download_weight(model_urls[model_name], weight_path)

        if weight_path and os.path.exists(weight_path):
            base_model = ResNet50(
                include_top=include_top,
                weights=None,
                input_shape=input_shape
            )
            base_model.load_weights(weight_path)
            if verbose:
                print("下载的权重加载成功")
        else:
            if verbose:
                print("使用 Keras 自动下载的权重")
            base_model = ResNet50(
                include_top=include_top,
                weights=weights,
                input_shape=input_shape
            )

    preprocess_func = get_preprocess_function(model_name)

    if verbose:
        print("模型加载完成")
        print(f"可训练参数数量: {base_model.count_params():,}")

    return base_model, preprocess_func, input_size


def list_available_models():
    """列出所有可用的模型"""
    print("可用的 backbone 模型:")
    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
        input_size = get_input_size(model_name)
        print(f"  {i}. {model_name:25s} (输入尺寸: {input_size}x{input_size})")


if __name__ == "__main__":
    list_available_models()

    print("\n" + "=" * 50)
    model, preprocess_func, input_size = load_backbone(MODEL_NAME, verbose=True)
