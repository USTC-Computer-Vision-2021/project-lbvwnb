基于图像分割的指定视频对象移除

1、成员及分工：
张驰
    coding
赵天翔
    coding

2、问题描述
背景：随着移动互联网时代的来临，视频成为了信息传播的重要媒介，每时每刻都有海量的视频内容被上传到互联网。
产生的问题：拍摄者上传的视频中，常有与视频无关的人出镜，而视频上传者往往未经其授权。即使拍摄者隐私权意识较强，与所有无关出镜者沟通以获取授权也是困难重重。于是侵犯隐私权的现象便屡见不鲜。
技术角度的解决方案：编写程序，绘制一个边界框，对边界框中的选定视频对象进行移除。

3、运行说明
已验证的运行环境：Ubuntu 16.04、Python 3.5、Pytorch 0.4.0、CUDA 8.0、GTX1080Ti GPU 

请先下载训练好的模型：https://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth 和 https://drive.google.com/file/d/1KAi9oQVBaJU9ytr7dYr2WwEcO5NLiJvo/view
然后把模型放到cp/文件夹中；
运行以下代码：
python demo.py --data data/Human6

画一个边框：
(https://user-images.githubusercontent.com/94847760/143771358-2b15d256-d427-4af2-9a87-4c43ea069f02.gif)

被边框选定的对象会被移除，处理后的视频将保存在results/inpainting文件夹中：
处理前后：<img width="591" alt="sgif-1-1-1" src="https://user-images.githubusercontent.com/94847760/143881800-5c1c44a6-f202-42c9-859a-7c9d02d39a95.PNG">
![image](https://user-images.githubusercontent.com/94847760/143882060-0e536e29-2b5e-42b2-930d-f0414e5abad8.png)
![image](https://user-images.githubusercontent.com/94847760/143882275-ad47415f-cfd2-49b6-8998-cf9eb8dd9e3e.png)
<img width="584" alt="sgif-1-1-4" src="https://user-images.githubusercontent.com/94847760/143883019-c0c60f37-242b-4d9c-bbbd-562e3c855622.PNG">










