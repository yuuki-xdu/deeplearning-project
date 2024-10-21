import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.utils import to_categorical,plot_model
from keras.api.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
from keras.api.optimizers import Adam
from sklearn.preprocessing import StandardScaler
'''
读取数据并处理
'''
#读取CSV文件
#这里的csv文件由matlab仿真生成，为5n行10列的矩阵，前9列为脉冲描述字，10列为标签0-4，共5种
data = pd.read_csv('signal_data_with_labels.csv', header=None)
X = data.iloc[:, [3, 4, 8]].values  # 第4列(索引为3)、第5列(索引为4)、第9列(索引为8)
y = data.iloc[:, -1].values         # 最后一列(标签)
# 将标签进行独热编码（one-hot encoding）
y = to_categorical(y, num_classes=5)

#进行数据的标准化
scaler = StandardScaler()
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''
模型创建与编译
'''
learning_rate=0.00025
# 使用Adam优化器并设置学习率为0.001
optimizer = Adam(learning_rate)
# 创建Keras模型
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 输出5个类别，对应5种信号模式

# 编译模型
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# 在每个 epoch 结束时保存模型


'''
训练模型
'''


checkpoint = ModelCheckpoint('my_model.keras', save_best_only=True)
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True) 
# 在训练过程中监测验证集上的性能，如果性能不再提升则提前停止训练
early_stopping = EarlyStopping(patience=15)
# 训练模型
model.fit(X_train_scaled, 
          y_train, 
          epochs=100, 
          batch_size=16, 
          validation_data=(X_test_scaled, y_test),
          callbacks=[checkpoint, early_stopping])

# 评估模型
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {accuracy}")


