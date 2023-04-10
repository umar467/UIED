ffmpeg -pattern_type glob -i 'data/output/frames/11_SIFT_merged/*.jpg' -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y -an 11_SIFT_merged_Full.mp4 
