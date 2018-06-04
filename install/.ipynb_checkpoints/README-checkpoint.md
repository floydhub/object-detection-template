# How to run Tensorflow Object Detection API


Before going through the installation, we have to upgrade the protobuf version by running the `bash upgrade-protobuf.sh`script, then we can install and test the package by running: `bash install.sh`.

Unfortunately there are some bugs that have to be to be fixed manually:

- Python3 error when reading from config file.
- Double log info: https://github.com/tensorflow/models/issues/3137

