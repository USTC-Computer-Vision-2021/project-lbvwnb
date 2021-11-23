Prerequisites:Python 3.8.5
Please run the following commands at first:

conda create -n FGVC
conda activate FGVC
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib scipy
pip install -r requirements.txt

Next, please download the model weight and demo data using the following command:

chmod +x download_data_weights.sh
./download_data_weights.sh



Object removal:

cd tool
python video_completion.py \
       --mode object_removal \
       --path ../data/tennis \
       --path_mask ../data/tennis_mask \
       --outroot ../result/tennis_removal \
       --seamless
       
FOV extrapolation:

cd tool
python video_completion.py \
       --mode video_extrapolation \
       --path ../data/tennis \
       --outroot ../result/tennis_extrapolation \
       --H_scale 2 \
       --W_scale 2 \
       --seamless
       
Besides,we need prepare:

opencv-python
opencv-contrib-python
imageio
imageio-ffmpeg
scipy
scikit-image


