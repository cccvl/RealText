#!/bin/bash
pip install -r requirements.txt
curl -O https://download.blender.org/release/Blender3.6/blender-3.6.16-linux-x64.tar.xz
tar -xf blender-3.6.16-linux-x64.tar.xz
./blender-3.6.16-linux-x64/3.6/python/bin/python3.10 -m pip install opencv-python==4.10.0.84
./blender-3.6.16-linux-x64/3.6/python/bin/python3.10 -m pip install pillow==10.4.0
mv blender.py blender-3.6.16-linux-x64/