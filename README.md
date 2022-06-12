# Deeplabv3_plus
Deeplabv3+语义分割网络
## 测试文件演示
左边为输入图片[256, 256]，右边为预测输出图片[256, 256]</br>
![road](https://user-images.githubusercontent.com/77096562/173211851-ace636a2-5fcb-4b7b-bc3a-696d7e067e2a.jpg)
![tmpeyccgqyj](https://user-images.githubusercontent.com/77096562/173211863-ddfb2e1d-ed94-441c-9204-5405b26fc4a4.PNG)
## 工程目录说明
main</br>
|----model </span></span> —— </span></span> 存放模型用文件</br>
|----utils </span></span> —— </span></span> 存放一些工具函数</br>
config.json </span></span> —— </span></span> 训练/预测配置文件</br>
predict.py </span></span> —— </span></span> 预测脚本</br>
road.jpg </span></span> —— </span></span> 预测测试图片</br>
train.py </span></span> —— </span></span> 训练脚本</br>
## 模型训练损失/验证损失/测试正确率
![loss](https://user-images.githubusercontent.com/77096562/173211877-8044588e-a728-49be-8788-45c7f6cbe161.png)</br>
train</span>loss</span>——训练损失
val</span>loss</span>——验证损失
cor</span>——验证准确率
