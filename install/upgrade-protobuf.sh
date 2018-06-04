#! /bin/bash

# Work on tmp
cd /tmp/

# Update Protocol Buffer
# Get the latest version from there: https://github.com/google/protobuf/releases
curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
unzip protoc-3.5.1-linux-x86_64.zip -d protoc3

# Move to Bin and Include
mv protoc3/bin/* /usr/local/bin/
mv protoc3/include/* /usr/local/include/

chown `whoami` /usr/local/bin/protoc 
chown -R `whoami` /usr/local/include/google