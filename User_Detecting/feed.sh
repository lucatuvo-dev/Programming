#!/bin/bash

ffmpeg -loglevel error -y -f avfoundation -pixel_format uyvy422 -framerate 30 -video_size 1280x720 -i "0" -frames:v 1 ~/User_Detecting/snapshot.jpg 2>/dev/null
