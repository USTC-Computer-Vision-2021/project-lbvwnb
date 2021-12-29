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
3、原理分析：
视频对象擦除，顾名思义，就是要让视频中的某个特定对象（如：移动的汽车、行走的路人等）从视频的每一帧画面中消失，取而代之的是该对象所在画面位置处的背景。可见整个过程主要涉及到两部分操作：一是找出选定对象在每一帧的具体位置，即找出一个Mask将对象给“框选”出来；然后便是将Mask中的每一个像素都用预测的背景像素替代，即对该Mask圈出的画面的消失部分进行修补（inpainting）。因此，需要解决的核心问题也就变成了：如何追踪某个选定对象在视频中不断变化位置，以及如何根据视频相邻帧画面之间的关联给出背景像素的预测结果。事实上，两个问题的关键都在于利用视频中的每一帧数据在时间上的连续性，借此来训练对应的网络模型进行相关预测。在本项目中，前者主要通过区域生成网络（RPN）实现，而后者则通过视频修补网络（VInet）实现。

3.1 Mask的获得（get_mask）——
Mask的获得严格意义上介于传统的视频对象追踪（video object tracking）以及视频对象分离（video object segmentation）任务之间，本项目中采用的SiamMask便是对两者解决方法的一个整合与优化。其具体运行流程可以简述如下：在视频初始帧中框选感兴趣对象后，自动识别其中真正属于目标的像素（因为框选部分可能存在背景像素），并对后续帧中该目标的位置进行估计和预测。而为了实现这一目标，SiamMask的训练过程也被划分为三步：首先，学习构建一套相似性准则用以度量目标对象和以滑动窗口形式送入的众多候选对象之间的相似性。因为这样得到的输出只能显示对象的位置，为了进一步确定其空间上的信息，将对象准确分离出来，接下来便要通过RPN实现边框回归（bounding box regression）,并进行类不可知的二元分割（class-agnostic binary segmentation）。
另一方面，SiamMask网络模型是基于已有的全卷积暹罗网络构建的，主要是给原来的暹罗追踪器引入了一个额外的分支和损失函数。而这一改变将得到两种SiamMask变体：三分支变体和双分支变体。其结构简图如下：
![image](https://user-images.githubusercontent.com/94847760/147634865-defadc54-a195-4edc-9baf-2eb9be56ef8e.png)
二者将分别优化多任务损失函数![image](https://user-images.githubusercontent.com/94847760/147635617-d4409682-f574-4058-b02d-2f8510873f7d.png)和![image](https://user-images.githubusercontent.com/94847760/147635638-5ba9df28-ce79-4d07-ac9f-c980e2b38ed6.png)，网络的训练也将围绕该目标展开。
3．2 视频修补（video_inpainting）——
设包含有消失区域的视频各帧为![image](https://user-images.githubusercontent.com/94847760/147635147-080ae3da-f633-42e6-84ea-9fda63b08bd2.png)，而填充了消失区域的真实视频各帧为![image](https://user-images.githubusercontent.com/94847760/147635187-06dfbe1f-b423-4a7e-9eb4-e6c401cbf073.png)，那么inpainting的目的就是学习一个从![image](https://user-images.githubusercontent.com/94847760/147635241-ed1168d2-1573-461c-bb2c-b20902842a89.png)到![image](https://user-images.githubusercontent.com/94847760/147635270-6273eb47-e3a3-44fd-bf44-522cd16b0a53.png)的映射，使得条件分布![image](https://user-images.githubusercontent.com/94847760/147635306-35046485-21e7-4a44-9666-373d8b9005bb.png)，![image](https://user-images.githubusercontent.com/94847760/147635405-140e75d2-785f-4122-b53a-48c3084be843.png)由下式计算得到：![image](https://user-images.githubusercontent.com/94847760/147635564-5a1931d4-83cd-4254-a478-c01860d2e09f.png)
其中，N取2，意为取预测帧的前后各两帧数据，而采样步长为3，亦即![image](https://user-images.githubusercontent.com/94847760/147635688-b47ffbdd-fb10-4a10-84ca-e6d7324b93f3.png)，![image](https://user-images.githubusercontent.com/94847760/147635710-79bfeec7-105c-4fd8-9c9d-bd17c095fde2.png)
为由之前帧得到的记忆信息，以确保预测帧的时间一致性。
这一学习过程则主要通过VInet实现。其设计初衷是期望通过改造一个前馈深度网络来实现视频修补，而这个新的深度CNN模型所具有的最为核心的两个功能，便是时间特征聚合以及时间连续性保留。对于前者，VInet设计者将视频修补任务转换成了一个“多到一”的连续帧的修补问题，借此引入一个新的基于图像编码解码模型的3D-2D前馈神经网络，用于收集并提炼视频相邻帧之间的潜在关联，从而据此在时间和空间两个维度合成语义一致的视频内容。而对于后者，设计者则采用了一个递归反馈层和一个记忆层来实现。除此之外，模型中还分别设置了流动损失函数和规整损失函数，分别用于学习先前合成帧的规整和加强inpainting结果中的长短期一致性。
VInet网络的大致框架如下图所示：
![image](https://user-images.githubusercontent.com/94847760/147635737-42c7a659-6aff-4514-a4dd-a224a131a069.png)
其中的Mask-sum运算由下式定义（⨀为逐元素相乘运算符）：
![image](https://user-images.githubusercontent.com/94847760/147635768-91155915-939b-4fd1-bae4-afd41c3d7a37.png)
网络训练用到的损失函数定义如下：
![image](https://user-images.githubusercontent.com/94847760/147635808-5e80e0f7-3a96-471b-937c-ecc7da9de28a.png)
其中，![image](https://user-images.githubusercontent.com/94847760/147635830-08635a52-8db0-4559-a5cc-882dcd1f43c9.png)、![image](https://user-images.githubusercontent.com/94847760/147635855-2ce2d4b0-fcb3-4acd-b2fc-9413311cce6e.png)、![image](https://user-images.githubusercontent.com/94847760/147635868-36d9999b-40c8-4f2a-93bc-91ce4bc7085f.png)分别为重建损失，流动损失和规整损失，权重因子分别为1、10、1.
3.3 代码实现：
Get_mask部分主要由下面的函数实现——
mask.py里的mask函数

Inpainting部分主要由下面的函数实现——
inpaint.py里的inpaint函数

上述函数中用到的现有模型和工具函数分别在get_mask和inpainting文件夹的models和utils中定义，此处略去。
4、运行说明
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
