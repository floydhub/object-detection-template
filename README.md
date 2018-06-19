# Object Detection

Recently the Chinese Police was on the first page of the most famous newspapers because they have succesfully detected a criminal in a pop concert with 60k people from security cameras using a facial recognition system. Here's a [link of the news](http://www.bbc.com/news/world-asia-china-43751276) if you missed it. This is not the first time that they used the same technology to catch criminals, but it is certainly quite amazing what this technology is able to achieve.

### Try it now

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/floydhub/object-detection-template)

Click this button to open a Workspace on FloydHub that will train this model.

### "Where is Syd?"

In this project, we will use [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to detect Syd. It turns out that Syd participated to several Sport events of the last years, mostly during the opening ceremony. Unfortunately, we were able to get only some images of him during the ceremonies... but don't worry, Transfer Learning comes to help. We will use a pre-trained SSD model on COCO dataset to find Syd. If the trained model we will be good enough we will be able to detect Syd in the next games and tell to your family & friends: 'Look there, it's Syd! I found him ;)'.

![wanted Syd](https://raw.githubusercontent.com/floydhub/object-detection-template/master/images/wanted-syd.jpg)
*Are you able to catch Syd in this image?*

The project is structered around 3 notebooks that have to been run in this order:

- `1_create_data` where we build a TFRecords dataset from the images with bounding box annotations,
- `2_training` where we perform the training, evaluation and model exportation,
- `3_prediction` where we evaluate the model on new data.

#### Template Structure

- `install/` contains the instruction to replicate the installation of Tensorflow Object Detection - API,
- `models/` contains the pretrained `ssdlite_mobilenet_v2` model on COCO dataset,
- `object_detection/` is the Tensorflow Object Detection framework,
- `slim/`, the TF Object Detection module is built upon [TF-SLIM](https://github.com/tensorflow/models/tree/master/research/slim) an high-level API of TensorFlow (tensorflow.contrib.slim) for defining, training and evaluating complex models (expecially CNN models),
- `tfrecors_data/` will contains the dataset in TFRecord format after the execution of the `1_create_data` notebook,
- `trained_models` will contains the model trained for the task after the execution of the `2_training` notebook. 
 