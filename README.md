# Deeplabv3_plus
Deeplabv3+语义分割网络</br>
配置都写在config.json文件当中</br>
网络采用pytorch框架进行编写</br>
## 测试文件演示
左边为输入图片[256, 256]，右边为预测输出图片[256, 256]</br>
![road](https://user-images.githubusercontent.com/77096562/173211851-ace636a2-5fcb-4b7b-bc3a-696d7e067e2a.jpg)
![tmpeyccgqyj](https://user-images.githubusercontent.com/77096562/173211863-ddfb2e1d-ed94-441c-9204-5405b26fc4a4.PNG)
## 训练数据集链接
示例当中采用自主设计的仿真环境中采集的图片作为数据集，随后使用Labelme进行数据集标注。</br>
![2](https://user-images.githubusercontent.com/77096562/182009686-89c61724-0ec4-4ffb-b783-c9162e1dabd3.jpg)
![2](https://user-images.githubusercontent.com/77096562/182009687-3ab3963e-3ddb-4c08-ac16-22955106ad0e.png)
链接：https://pan.baidu.com/s/1P68ZbL538a6m73DLG2LHdQ </br>
提取码：7969 </br>
## 工程目录说明
main</br>
|----model &emsp; —— &emsp; 存放模型用文件</br>
&emsp;&emsp;|----backbone.py &emsp; —— &emsp; 模型主干特征提取网络文件</br>
&emsp;&emsp;|----deeplabv3.py &emsp; —— &emsp; deeplabv3模型文件</br>
|----utils &emsp; —— &emsp; 存放一些工具函数</br>
&emsp;&emsp;|----loss.py &emsp; —— &emsp; 损失函数</br>
&emsp;&emsp;|----dataloader.py &emsp; —— &emsp; 数据集加载相关文件</br>
&emsp;&emsp;|----utils.py &emsp; —— &emsp; 一些其他工具类API文件</br>
config.json &emsp; —— &emsp; 训练/预测配置文件</br>
predict.py &emsp; —— &emsp; 预测脚本</br>
road.jpg &emsp; —— &emsp; 预测测试图片</br>
train.py &emsp; —— &emsp; 训练脚本</br>
val.py &emsp; —— &emsp; 验证脚本</br>
## 模型训练损失/验证损失/测试正确率
![loss](https://user-images.githubusercontent.com/77096562/173211877-8044588e-a728-49be-8788-45c7f6cbe161.png)</br>
train&nbsp;loss &emsp; —— &emsp; 训练损失</br>
val&nbsp;loss &emsp; —— &emsp; 验证损失</br>
cor &emsp; —— &emsp; 验证准确率（准确率=图片正确的像素总数/图片像素总数）</br>
## 模型详细结构示意图
![image](https://user-images.githubusercontent.com/77096562/173212818-87605b81-c577-4d99-9a69-485f4fb6207d.png)
![image](https://user-images.githubusercontent.com/77096562/173212832-87a8263b-f139-4c41-8c41-c11f4467dfd3.png)
![image](https://user-images.githubusercontent.com/77096562/173212868-b12fe62a-97ff-4d42-a2a8-e0c417ec7282.png)
![image](https://user-images.githubusercontent.com/77096562/173212961-73c305e0-b881-4422-b2ce-d2f8530495df.png)
