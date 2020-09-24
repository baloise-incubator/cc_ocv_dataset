# cc_ocv_dataset

A dataset for the codecamp identity card detection

## How to get a trained model with tensorflow

### Prepare Os (Ubuntu 20.04.1)

1. Install bare metal Ubuntu 20.04
2. Change /etc/sudoers to eliminate password for "sudo" commands
```
sudo visudo
```

Change line
```
%sudo   ALL=(ALL:ALL) ALL
```
to
```
%sudo   ALL=(ALL:ALL) NOPASSWD:ALL
```
3. Upgrade Ubuntu
```
sudo apt -y update 
sudo apt -y upgrade
sudo apt -y dist-upgrade
```
4. Install additional packages
```
sudo apt install python3-venv unzip protobuf-compiler build-essential python3-dev libgl1
```

5. Create new user for your work with tensorflow and provide a password
```
sudo adduser code-camp
sudo addgroup code-camp sudo
```


### Prepare directory structure and python env 

Do all the following commands as your preferred user for working with tensorflow !!!

1. Create Project infrastructure
```
cd $HOME
mkdir -p Projects
```

2. Create Python venv
```
cd $HOME/Projects
python3 -m venv tensorflow
mkdir -p $HOME/Projects/tensorflow/usr/src
```

3. Add automatic activation for the python venv
```
echo '. $HOME/Projects/tensorflow/bin/activate' >> $HOME/.bashrc
echo 'cd $HOME/Projects/tensorflow/usr/src' >> $HOME/.bashrc
```

Now you are ready to work with Python in an isolated environment
Now relogin 

### Install tensorflow the PIP way :-)

1. Install tensorflow
```
pip install --upgrade pip
pip install --upgrade tensorflow
```

2. Download the ZIP from tensorflow/models, do not clone (to much space needed)
```
cd $HOME/Projects/tensorflow/usr/src
wget https://github.com/tensorflow/models/archive/master.zip -O tensorflow-models-master.zip
```

3. unzip 
```
unzip tensorflow-models-master.zip
mv models-master models
```

4. Install additional requirements from rep
```
pip install -r models/official/requirements.txt
```

5. Compile proto
```
ls -l $HOME/Projects/tensorflow/usr/src/models/research/object_detection/protos/*.py
cd $HOME/Projects/tensorflow/usr/src/models/research
protoc object_detection/protos/*.proto --python_out=.
ls -l $HOME/Projects/tensorflow/usr/src/models/research/object_detection/protos/*.py
```

6. Prepare python path
```
echo "export PYTHONPATH=:$HOME/Projects/tensorflow/usr/src/models/research:$HOME/Projects/tensorflow/usr/src/models/research/slim:$HOME/Projects/tensorflow/usr/src/models" >> $HOME/.bashrc
```

7. Relogin :-)

### Clone Dataset for image creation and create coco dataset
This repo has a script to download images and prepare the coco-dataset annotations

```
cd $HOME/Projects/tensorflow/usr/src
git clone https://github.com/baloise-incubator/cc_ocv_dataset.git
cd cc_ocv_dataset
pip install scikit-image
pip install Shapely
pip install tqdm
```

Now you can edit the images and background Urls for your own purpose
```
background/images.tst
idcards/images.txt
```

and run the script

```
python generate_coco_dataset.py
sed -Ei 's#"category_id": 1#"category_id": 91#g' annotations.json
sed -Ei 's#"id": 1#"id": 91#g' annotations.json
rm -f images/*mask*
```

Now you should have quite a lot of images in sub images and a file annotations.json
```
ls annot* images/*
```

### Modify coco dataset to XML
This is necessary because many tutorials work XML for each image as labelImg would create it, so we simulate this :-)

```
cd $HOME/Projects/tensorflow/usr/src
wget https://github.com/mhiyer/coco-annotations-to-xml/archive/master.zip -O coco-annotations-to-xml-master.zip
unzip coco-annotations-to-xml-master.zip
mv coco-annotations-to-xml-master coco-annotations-to-xml
cd coco-annotations-to-xml
```

It is necessary to modify the python script coco_get_annotations_xml_format.py to your setup. It was a quick and dirty script so it has parameters and so on :-)
Change
```
        image_name = '{0:012d}.jpg'.format(image_id)
	annotations_path = '...'
	image_folder = '...'

```
to
```
        image_name = '{0:0d}.jpg'.format(image_id)                                                                                                                                                               
	annotations_path = '/home/code-camp/Projects/tensorflow/usr/src/cc_ocv_dataset/annotations.json'
	image_folder = '/home/code-camp/Projects/tensorflow/usr/src/cc_ocv_dataset/images'
```

Add category
```
echo "card,91,idcard" >> coco_categories.csv
```

and run it
```
python coco_get_annotations_xml_format.py
cd saved
sed -Ei 's#filename>([[:digit:]]*)</filename#filename>\1.jpg</filename#g' *.xml
cp *xml ../../cc_ocv_dataset/images/

```

### Prepare workspace

```
cd $HOME/Projects/tensorflow/usr/src
mkdir -p workspace/training_demo 
cd workspace/training_demo
mkdir -p annotations exported-models pre-trained-models images models script/preprocessing scripts/preprocessing
```

### Split dataset to train and test

```
cd $HOME/Projects/tensorflow/usr/src
wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/d0e545609c5f7f49f39abc7b6a38cec3/partition_dataset.py
python partition_dataset.py -i cc_ocv_dataset/images -o workspace/training_demo/images -r 0.1 -x
```

### Create label
Create a file annotations/label_map.pbtxt in your workspace
```
cat >$HOME/Projects/tensorflow/usr/src/workspace/training_demo/annotations/label_map.pbtxt << EOF
item {
    id: 91
    name: 'idcard'
}
EOF
```

### Create tfrecord

```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo/scripts/preprocessing
wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py
python generate_tfrecord.py -x ../../images/test -l ../../annotations/label_map.pbtxt -o ../../annotations/test.record
python generate_tfrecord.py -x ../../images/train -l ../../annotations/label_map.pbtxt -o ../../annotations/train.record

```


### Download Pre-Trained-model

```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo/pre-trained-models
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar xzvf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

### Copy pipeline_config
```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo/models
mkdir my_ssd_resnet_50_v1_fpn                                                                                                                                                                                      
cd my_ssd_resnet_50_v1_fpn/                                                                                                                                                                                        
cp ../../pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config .                                                                                                                             
```
See below diff for changes to config file
```
3c3
<     num_classes: 90
---
>     num_classes: 1
131c131
<   batch_size: 64
---
>   batch_size: 4
161c161
<   fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED"
---
>   fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pre-trained model
167,168c167,168
<   fine_tune_checkpoint_type: "classification"
<   use_bfloat16: true
---
>   fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
>   use_bfloat16: false # Set this to false if you are not training on a TPU
172c172
<   label_map_path: "PATH_TO_BE_CONFIGURED"
---
>   label_map_path: "annotations/label_map.pbtxt" # Path to label map file
174c174
<     input_path: "PATH_TO_BE_CONFIGURED"
---
>     input_path: "annotations/train.record" # Path to training TFRecord file
182c182
<   label_map_path: "PATH_TO_BE_CONFIGURED"
---
>   label_map_path: "annotations/label_map.pbtxt" # Path to label map file
186c186
<     input_path: "PATH_TO_BE_CONFIGURED"
---
>     input_path: "annotations/test.record" # Path to testing TFRecord
```

### Train
```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo
cp ../../models/research/object_detection/model_main_tf2.py .
time python model_main_tf2.py --model_dir=models/my_ssd_resnet_50_v1_fpn --pipeline_config_path=models/my_ssd_resnet_50_v1_fpn/pipeline.config --alsologtostderr --num_train_steps=50     --sample_1_of_n_eval_examples=1
```

### Watch tensorboard
Do this in a separate windows

```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo                                                                                                                                                       
tensorboard --logdir=models/my_ssd_resnet_50_v1_fpn --bind_all
```

### Export
```
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo                                                                                                                                                       
cp ../../models/research/object_detection/exporter_main_v2.py .
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_resnet_50_v1_fpn/pipeline.config  --trained_checkpoint_dir models/my_ssd_resnet_50_v1_fpn --output_directory exported-models/my_model
```

### Create mobile JSON
```
pip install tensorflow-js
cd $HOME/Projects/tensorflow/usr/src/workspace/training_demo
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='MobilenetV1/Predictions/Reshape_1' --saved_model_tags=serve     exported-models/my_model/saved_model exported-models/my_model/web_model
```
