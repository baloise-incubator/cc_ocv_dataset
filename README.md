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
sudo adduser tensorflow
sudo addgroup tensorflow sudo
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
python -m venv tensorflow
mkdir -p $HOME/tensorflow/usr/src
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
wget https://github.com/tensorflow/models/archive/master.zip -O tensorflow-i-models-master.zip
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
ls -l $HOME/tensorflow/usr/src/models/research/object_detection/protos/*.py
protoc $HOME/tensorflow/usr/src/models/research/object_detection/protos/*.proto --python_out=.
ls -l $HOME/tensorflow/usr/src/models/research/object_detection/protos/*.py
```

6. Prepare python path
```
echo "export PYTHONPATH=:$HOME/tensorflow/Projects/tensorflow/usr/src/models/research:$HOME/tensorflow/Projects/tensorflow/usr/src/models/research/slim:$HOME/tensorflow/Projects/tensorflow/usr/src/models" >> $HOME/.bashrc
```

7. Relogin :-)

### Clone Dataset for image creation and create coco dataset
This repo has a script to download images and prepare the coco-dataset annotations

```
cd $HOME/Projects/tensorflow/usr/src/addons
git clone https://github.com/baloise-incubator/cc_ocv_dataset.git
cd cc_ocv_dataset
```

Now you can edit the images and background Urls for your own purpose
```
background/images.tst
idcards/images.txt
```

and run the script

```
python generate_coco_dataset.py
rm -f images/*mask*
```

Now you shoul have quite a lot of images in sub images and a file annotations.json
```
ls -l annot* images/*
```

### Modify coco dataset to XML
This is necessary because many tutorials work XML for each image as labelImg would create it, so we simulate this :-)

```
cd $HOME/Projects/tensorflow/usr/src/addons
wget https://github.com/mhiyer/coco-annotations-to-xml/archive/master.zip -O coco-annotations-to-xml-master.zip
unzip coco-annotations-to-xml-master.zip
mv coco-annotations-to-xml-master coco-annotations-to-xml
```

It is necessary to modify the python script to your setup. It was a quick and dirty script so it has parameters and so on :-)
Change
```
        image_name = '{0:012d}.jpg'.format(image_id)
```
to
```
        image_name = '{0:0d}.jpg'.format(image_id)                                                                                                                                                               
```

and run it
```
python coco_get_annotations_xml_format.py
```

### Prepare workspace

```
cd $HOME/Projects/tensorflow/usr/src
mkdir -p workspace/tensorflow 
cd workspace/tensorflow
mkdir -p annotations exported-models pre-trained-models images models script/preprocessing
```

### Split dataset to train and test

```
cd $HOME/Projects/tensorflow/usr/src/addons                                                                                                                                                                        
wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/d0e545609c5f7f49f39abc7b6a38cec3/partition_dataset.py
python partition_dataset.py -i cc_ocv_dataset/images -o ../workspace/training_demo/images -r 0.1 -x
```

### Create label
Create a file annotations/label_map.pbtxt in your workspace
```
cat >$HOME/Projects/tensorflow/usr/src/workspace/tensorflow/annotations/label_map.txt << EOF
item {
    id: 91
    name: 'idcard'
}
EOF
```

### Create tfrecord

```
cd $HOME/Projects/tensorflow/usr/src/scripts/preprocessing
wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py
python generate_tfrecord.py -x ../../workspace/training_demo/images/test -l ../../workspace/training_demo/annotations/label_map.pbtxt -o ../../workspace/training_demo/annotations/test.record

```


### Download Pre-Trained-model

```
cd $HOME/Projects/tensorflow/usr/src/workspace/tensorflow/pre-trained-models
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar xzvf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

### Copy pipeline_config
### Copy train.py
### train
### Watch tensorboard
