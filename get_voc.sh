#!/bin/bash
# chmod +x get_voc.sh
echo '**' `date +%H:%M:%S` 'start **'

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012.tar
mkdir -p ./data/voc
tar -xf ./data/voc2012.tar -C ./data/voc

echo 'Dataset Downloaded'
