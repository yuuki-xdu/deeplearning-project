import pandas as pd
import numpy as np
from keras.api.models import load_model
from sklearn.preprocessing import StandardScaler
loadedModel= load_model('my_model.keras')
scaler = StandardScaler()
new_data=pd.read_csv('tbd', header=None)
new_data1=new_data.iloc[:, [3, 4, 8]].values
new_data_scaled=scaler.fit_transform(new_data1)
predictions=loadedModel.predict(new_data_scaled)
# 打印预测结果 (这是一个概率分布)
print(predictions)
# 获取预测类别 (取概率最大的类别)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
