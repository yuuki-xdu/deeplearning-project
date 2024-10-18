import tensorflow as tf
#import keras
from keras import Sequential 
from keras import LSTM, Dense
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
#此处需要定义数据集 x,y_train x,y_test

# 创建RNN模型
model = Sequential()

# 添加LSTM层，输入形状为 (sequence_length, feature_size)
# 假设 feature_size = 1（例如只有幅值），sequence_length = 100
model.add(LSTM(64, input_shape=(100, 1), return_sequences=False))

# 添加输出层，根据你要分类的信号类别数进行调整
# 这里假设有6类雷达信号
model.add(Dense(6, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()
#print(keras.__version__)
# X_train 是形状为 (num_samples, sequence_length, feature_size) 的训练数据
# Y_train 是形状为 (num_samples, num_classes) 的标签数据（已进行one-hot编码）

 
# 在每个 epoch 结束时保存模型
checkpoint = ModelCheckpoint('my_model', save_best_only=True)
 
# 在训练过程中监测验证集上的性能，如果性能不再提升则提前停止训练
early_stopping = EarlyStopping(patience=3)
 
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint, early_stopping])
#model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2)
# 在测试数据上评估模型
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")
#实现模型的保存和加载
#model.save('my_model')
#model = keras.models.load_model('my_model')
# 使用训练好的模型对新的雷达信号进行分类 X_new
predictions = model.predict(X_new)