#!/bin/bash

echo DOWNLOADING

# curl https://drive.google.com/file/d/1BmLdIE0rDiUJmx-OnNd67sIchv8Ze3M5/view?usp=sharing

filename="blacklist.zip"

base_filename=$(echo ${filename} | cut -d '.' -f1)

fileid="1BmLdIE0rDiUJmx-OnNd67sIchv8Ze3M5"

curl -L -o ${filename} "https://drive.google.com/uc?export=download&id=${fileid}"


echo "UNZIPPING"

unzip ${filename}

echo "CLEANING"

rm ${filename}

echo "MOVING"

mv ${base_filename} src/

echo "DONE!"
