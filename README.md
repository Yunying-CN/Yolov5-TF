# Yolov5-TF
本文介绍了基于TensorFlow 2.3.0来实现Yolov5的训练以及通过C++进行OpenVINO的部署




## 模型训练
1. 获取工程并进入工程         
```
git clone https://github.com/Yunying-CN/Yolov5-TF·git
cd Yolov5
```               

2. 准备数据集             
本文基于Pascal Voc2012数据集来进行训练，通过在终端上用以下命令直接下载，也可以制作自己的数据集。               
```
bash get_voc.sh
```                  

3. 生成TFRecord文件             
为了便于TensorFlow读取数据集，先将数据信息保存到.TFRecord，减少数据读取时间。
```
python3 TF_VOC.py
```            

4. 训练              
准备好数据集后，可以修改在/models/文件夹下的.yaml文件来修改网络预测类别数nc和网络结构来训练
```
python3 train.py  python3 train.py --num_classes 20 --epochs 30 --yaml_dir ./models/yolov5s.yaml --saved_dir ./weights
```    
- num_classes: 模型预测的类别数               
- epoch: 总训练轮数        
- yaml_dir: 模型文件           
- saved_dir: 训练后的模型参数保存路径                 



## OpenVINO部署
1. 模型转换                                        
训练得到的模型原始存储格式为.pb格式，为了实现OpenVINO部署，再转化为OpenVINO需要的.xml和.bin的存储格式
```
python mo_tf.py --saved_model_dir <.pb文件夹路径> --input_shape [1,640,640,3] --output_dir <输出文件夹路径> --data_type FP32
```        

- saved_model_dir 为训练得到的模型路径        
- input_shape 为指定图像尺寸          
- output_dir 为转换后的输出路径           
- data_type 为设置数据格式       

2. 推理部署                            
通过执行main.cpp工程可以实现模型部署，其中在部署中包括几部分：      
- 推理引擎的初始化
- 数据准备和输入
- 执行推理
- NMS及阈值处理

## 检测结果
![Alt text](https://github.com/Yunying-CN/Yolov5-TF/blob/main/img/test.jpg "Input")
![Alt text](https://github.com/Yunying-CN/Yolov5-TF/blob/main/img/output.jpg "Onput")
