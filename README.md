<img width="421" alt="捕获1" src="https://user-images.githubusercontent.com/94847760/147587070-63499f95-10f7-4ef1-a297-ab4fc26165e7.PNG">
<img width="418" alt="捕获2" src="https://user-images.githubusercontent.com/94847760/147587082-76dd69c9-eda9-45f4-b902-162e9094f4fa.PNG">
<img width="421" alt="捕获3" src="https://user-images.githubusercontent.com/94847760/147587089-e499fb8a-9ef1-4d53-9f53-08ea900a01d9.PNG">
<img width="420" alt="捕获4" src="https://user-images.githubusercontent.com/94847760/147587100-3bdfc2a7-a9a6-4ab0-a1c2-2c64a301d018.PNG">
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
