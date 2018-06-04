#! /bin/bash

# Commit: ea6d6aabe5c121102a645d3f08cf819fa28d2a03
# Clone Models, Compile protobuff and export the PATH variable
git clone https://github.com/tensorflow/models.git
cd models/research && \
git reset --hard ea6d6aa && \
/usr/local/bin/protoc object_detection/protos/*.proto --python_out=. && \
cp -R object_detection /floyd/code && cp -R slim /floyd/code

rm -rf /floyd/code/models

export PYTHONPATH=$PYTHONPATH:/floyd/code/object_detection/:/floyd/code/slim

# Test if everything works
cd /floyd/code/ && python object_detection/builders/model_builder_test.py
