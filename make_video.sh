ffmpeg -pattern_type glob -i 'data/output/frames/11_frame_averaged/*.jpg' -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -an 11_frame_averaged.mp4  
