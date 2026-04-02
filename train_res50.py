#!/usr/bin/env python
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  
# ===================== GPU 设置 =====================
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"可用GPU数量: {len(gpus)}")
    except RuntimeError as e:
        print(e)

# ===================== 导入模块 =====================
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# 添加model.py路径
sys.path.append('/home/cyp/weather/Keras-MultiClass-Image-Classification-master/scripts')
from model import load_backbone, list_available_models

# ===================== 自定义进度条回调（只显示进度，不显示指标） =====================
class ProgressBarCallback(Callback):
    """只显示训练进度的进度条"""
    def __init__(self, total_epochs, steps_per_epoch):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.epoch_bar = None
        self.batch_bar = None
        
    def on_train_begin(self, logs=None):
        print("\n开始训练...")
        self.epoch_bar = tqdm(total=self.total_epochs, desc='总体进度', unit='epoch', position=0)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_bar = tqdm(total=self.steps_per_epoch, 
                              desc=f'Epoch {epoch+1}/{self.total_epochs}', 
                              unit='batch', 
                              leave=False,
                              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
    def on_batch_end(self, batch, logs=None):
        self.batch_bar.update(1)
        if batch == self.steps_per_epoch - 1:
            self.batch_bar.close()
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        
    def on_train_end(self, logs=None):
        self.epoch_bar.close()

# ===================== 自定义详细指标回调（每个epoch结束后打印） =====================
class VerboseMetricsCallback(Callback):
    """每个epoch结束后打印详细的训练和验证指标"""
    def __init__(self, validation_generator=None):
        super().__init__()
        self.validation_generator = validation_generator
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        print(f"\n{'='*70}")
        print(f" Epoch {epoch+1}/{self.params['epochs']} finished")
        print(f"{'='*70}")
        
        # 训练指标
        print(f"\n train:")
        print(f"   ├─ Loss:     {logs.get('loss', 0):.6f}")
        print(f"   └─ Acc: {logs.get('accuracy', 0):.4f} ({logs.get('accuracy', 0)*100:.2f}%)")
        
        # 验证指标
        print(f"\n val:")
        print(f"   ├─ Loss:     {logs.get('val_loss', 0):.6f}")
        print(f"   └─ Acc: {logs.get('val_accuracy', 0):.4f} ({logs.get('val_accuracy', 0)*100:.2f}%)")
        
        # 学习率
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"\n🔧 学习率: {lr:.2e}")
        
        # 最佳准确率
        if hasattr(self, 'best_accuracy'):
            self.best_accuracy = max(self.best_accuracy, logs.get('val_accuracy', 0))
        else:
            self.best_accuracy = logs.get('val_accuracy', 0)
        print(f"\n🏆 最佳验证准确率: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        
        print(f"{'='*70}\n")

# ===================== 自定义 F1 Score 回调 =====================
class F1ScoreCallback(Callback):
    """在每个epoch结束后计算并记录F1分数"""
    def __init__(self, validation_generator, validation_steps=None):
        super().__init__()
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.val_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 获取验证集预测结果
        y_true = []
        y_pred = []
        
        # 重置验证集生成器
        self.validation_generator.reset()
        
        # 确定步数
        steps = self.validation_steps if self.validation_steps else len(self.validation_generator)
        
        print(f"正在计算详细指标...")
        
        # 批量预测（带进度条）
        for _ in tqdm(range(steps), desc='计算F1分数', leave=False):
            try:
                x_batch, y_batch = next(self.validation_generator)
                pred_batch = self.model.predict(x_batch, verbose=0)
                
                y_true.extend(np.argmax(y_batch, axis=1))
                y_pred.extend(np.argmax(pred_batch, axis=1))
            except StopIteration:
                break
        
        # 计算各种指标
        from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        # 保存到历史记录
        self.val_f1_scores.append(f1)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        
        # 添加到logs中
        logs['val_f1_score'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall
        
        # 打印详细指标
        print(f"\n 详细验证指标:")
        print(f"   ├─ F1 Score:  {f1:.4f}")
        print(f"   ├─ Precision: {precision:.4f}")
        print(f"   └─ Recall:    {recall:.4f}")
        
        # 可选：打印每个类别的详细报告
        # print("\n📋 分类报告:")
        # target_names = list(self.validation_generator.class_indices.keys())
        # report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        # print(report)

# ===================== 配置参数 =====================
class Config:
    # 数据集参数
    img_width = 256
    img_height = 256
    batch_size = 16
    num_classes = 5
    
    # 模型参数
    backbone = 'resnet50'
    
    # 训练参数
    epochs = 50
    learning_rate = 0.0001
    patience = 4
    dropout_rate = 0.3
    l2_reg = 0.0001
    
    # 路径配置
    data_dir = '/home/cyp/weather/weather_pred/Data/'
    training_dir = os.path.join(data_dir, 'training/')
    validation_dir = os.path.join(data_dir, 'validation/')
    save_dir = '/home/cyp/weather/model'
    see_data_dir = '/home/cyp/weather/see_data'
    weight_dir = '/home/cyp/weather/weights'
    
    # 是否冻结backbone
    freeze_backbone = True
    freeze_layers_count = -7

# ===================== 创建目录 =====================
os.makedirs(os.path.join(Config.save_dir, 'model'), exist_ok=True)
os.makedirs(Config.see_data_dir, exist_ok=True)
os.makedirs(Config.weight_dir, exist_ok=True)

# ===================== 加载 Backbone =====================
print("="*60)
print("初始化模型")
print("="*60)

list_available_models()
print(f"\n选择 backbone: {Config.backbone}")

weight_path = os.path.join(Config.weight_dir, f'{Config.backbone}_notop.h5')
base_model, preprocess_func, input_size = load_backbone(
    model_name=Config.backbone,
    weights='imagenet',
    include_top=False,
    weight_path=weight_path,
    input_shape=(Config.img_height, Config.img_width, 3),
    verbose=True
)

Config.img_width = input_size
Config.img_height = input_size
print(f"\n使用输入尺寸: {Config.img_width}x{Config.img_height}")

# ===================== 数据增强 =====================
print("\n" + "="*60)
print("准备数据加载器")
print("="*60)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_func,
    rotation_range=30,
    zoom_range=0.4,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_directory(
    Config.training_dir,
    batch_size=Config.batch_size,
    class_mode='categorical',
    target_size=(Config.img_height, Config.img_width),
    shuffle=True
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

validation_generator = validation_datagen.flow_from_directory(
    Config.validation_dir,
    batch_size=Config.batch_size,
    class_mode='categorical',
    target_size=(Config.img_height, Config.img_width),
    shuffle=False
)

print(f"训练集类别: {train_generator.class_indices}")
print(f"类别数量: {Config.num_classes}")

# ===================== 构建分类模型 =====================
print("\n" + "="*60)
print("构建分类头")
print("="*60)

output = base_model.output
output = GlobalAveragePooling2D()(output)
output = Dense(512, activation="relu", kernel_regularizer=l2(Config.l2_reg))(output)
output = BatchNormalization()(output)
output = Dropout(Config.dropout_rate)(output)
output = Dense(256, activation="relu", kernel_regularizer=l2(Config.l2_reg))(output)
output = BatchNormalization()(output)
output = Dropout(Config.dropout_rate)(output)
output = Dense(Config.num_classes, activation='softmax')(output)

model = Model(base_model.input, output)

if Config.freeze_backbone:
    print(f"冻结backbone层 (保留最后 {abs(Config.freeze_layers_count)} 层可训练)")
    for layer in base_model.layers:
        layer.trainable = False

print(f"\n总层数: {len(model.layers)}")
print(f"可训练层数: {sum(1 for layer in model.layers if layer.trainable)}")
print(f"可训练参数数量: {model.count_params():,}")

# ===================== 编译模型 =====================
model.compile(
    optimizer=Adam(learning_rate=Config.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===================== 回调函数 =====================
best_model_file = os.path.join(Config.save_dir, 'model', f'{Config.backbone}_best_weights.h5')

# 计算每个epoch的步数
steps_per_epoch = len(train_generator)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=Config.patience, verbose=1, restore_best_weights=True),
    ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ProgressBarCallback(total_epochs=Config.epochs, steps_per_epoch=steps_per_epoch),
    VerboseMetricsCallback(),
    F1ScoreCallback(validation_generator)
]

# ===================== 训练模型 =====================
print("\n" + "="*60)
print("开始训练")
print("="*60)

history = model.fit(
    train_generator,
    epochs=Config.epochs,
    verbose=0,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ===================== 保存最终模型 =====================
print("\n" + "="*60)
print("保存模型")
print("="*60)

model.save(os.path.join(Config.save_dir, f'{Config.backbone}_model.h5'))
model.save_weights(os.path.join(Config.save_dir, f'{Config.backbone}_weights.h5'))
print(f"✅ 模型已保存到: {Config.save_dir}")

# ===================== 绘制训练曲线 =====================
print("\n" + "="*60)
print("生成训练图表")
print("="*60)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

has_f1 = 'val_f1_score' in history.history
if has_f1:
    val_f1 = history.history['val_f1_score']
    val_precision = history.history['val_precision']
    val_recall = history.history['val_recall']

epochs_range = range(len(acc))

# 绘制综合曲线
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, 'r', label="Training Accuracy", linewidth=2)
plt.plot(epochs_range, val_acc, 'b', label="Validation Accuracy", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{Config.backbone} - Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, 'r', label="Training Loss", linewidth=2)
plt.plot(epochs_range, val_loss, 'b', label="Validation Loss", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{Config.backbone} - Training and Validation Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

if has_f1:
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, val_f1, 'g', label="Validation F1 Score", linewidth=2)
    plt.plot(epochs_range, val_precision, 'orange', label="Validation Precision", linewidth=2)
    plt.plot(epochs_range, val_recall, 'purple', label="Validation Recall", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'{Config.backbone} - Validation Metrics')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.see_data_dir, f'{Config.backbone}_training_curves.jpg'), dpi=300, bbox_inches='tight')
plt.close()

# 单独保存各曲线
plt.figure(figsize=(20, 10))
plt.plot(epochs_range, acc, 'r', label="Training Accuracy", linewidth=2)
plt.plot(epochs_range, val_acc, 'b', label="Validation Accuracy", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{Config.backbone} - Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(Config.see_data_dir, f'{Config.backbone}_Accuracy_curve.jpg'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(epochs_range, loss, 'r', label="Training Loss", linewidth=2)
plt.plot(epochs_range, val_loss, 'b', label="Validation Loss", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{Config.backbone} - Training and Validation Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(Config.see_data_dir, f'{Config.backbone}_Loss_curve.jpg'), dpi=300, bbox_inches='tight')
plt.close()

if has_f1:
    plt.figure(figsize=(20, 10))
    plt.plot(epochs_range, val_f1, 'g', label="Validation F1 Score", linewidth=2)
    plt.plot(epochs_range, val_precision, 'orange', label="Validation Precision", linewidth=2)
    plt.plot(epochs_range, val_recall, 'purple', label="Validation Recall", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'{Config.backbone} - Validation F1, Precision, Recall')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(Config.see_data_dir, f'{Config.backbone}_F1_curve.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

# ===================== 打印最终结果 =====================
print("\n" + "="*60)
print("训练完成！最终结果")
print("="*60)
print(f"最佳验证准确率: {max(val_acc):.4f}")
print(f"最佳验证损失: {min(val_loss):.4f}")
if has_f1:
    print(f"最佳验证F1分数: {max(val_f1):.4f}")
    print(f"最佳验证精确率: {max(val_precision):.4f}")
    print(f"最佳验证召回率: {max(val_recall):.4f}")

print(f"\n📊 图表已保存到: {Config.see_data_dir}")
print(f"   - {Config.backbone}_training_curves.jpg")
print(f"   - {Config.backbone}_Accuracy_curve.jpg")
print(f"   - {Config.backbone}_Loss_curve.jpg")
if has_f1:
    print(f"   - {Config.backbone}_F1_curve.jpg")

print(f"\n💾 模型已保存到: {Config.save_dir}")
print(f"   - model/{Config.backbone}_best_weights.h5 (最佳权重)")
print(f"   - {Config.backbone}_model.h5 (完整模型)")
print(f"   - {Config.backbone}_weights.h5 (模型权重)")

print("\n✅ 全部完成！")