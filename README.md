# SegNetwork
这是一个用于图像分割的深度学习网络模型

# 安装需求
### 依赖项
- Anaconda3 环境
- Python 3.8+
- Pytorch 2.0.0+
- pycocotools

### 环境搭建
- 配置环境安装 python 所需依赖库文件
```shell
pip install requirements.txt
```

### 数据集准备
- 下载COCO数据集，官网地址：https://cocodataset.org/#download 

    下载：http://images.cocodataset.org/zips/train2017.zip ；http://images.cocodataset.org/zips/val2017.zip ；
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- 将COCO数据集解压为如下格式：
```
data/coco2017/
    annotations/
        ...
    train2017/
        ...
    val2017/
        ...
```

### 网络训练
将```data-dir```设置为数据集路径，运行```train.py```文件
```
python train.py --data-dir /data/coco2017
```
具体参数可以在```train.py```文件中修改

### 模型测试
运行```test.py```，可以在文件内修改详细参数
```
python test.py --data-dir /data/coco2017
```

### 结果测试
需要在```inference.py```头部修改预训练文件路径和数据集路径
```
python inference.py
```
结果将保存到```./image/```路径下






