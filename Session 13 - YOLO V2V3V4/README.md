## Analysis

### Target:
1. OpenCV Yolo: [SOURCE](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

  - Run this above code on your laptop or Colab. 
  - Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
  -  Run this image through the code above. 
  - Upload the link to GitHub implementation of this
  - Upload the annotated image by YOLO. 
  
2. Share your NEWLY annotated (same as assignment 12, but annotated using new tool) images with Zoheb by Wednesday at midnight. Take the set back for training on Thursday.

3. Training Custom Dataset on Colab for YoloV3

  - Refer to this Colab File: [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
  - Refer to this GitHub [Repo](https://github.com/theschoolofai/YoloV3)
  - Collect a dataset from the last assignment and re-annotate them. Steps are explained in the readme.md file on GitHub.
  
4. Once done:
  
  - [Download](https://www.y2mate.com/en19) a very small (~10-30sec) video from youtube which shows your classes. 
  - Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to extract frames from the video. 
  - Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
  - Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
    `python detect.py --conf-thres 0.3 --output output_folder_name`
  - Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to convert the files in your output folder to video
  - Upload the video to YouTube. 
  
5. Share the link to your GitHub project with the steps as mentioned above
6. Share the link to your YouTube video
7. Share the link of your YouTube video on LinkedIn, Instagram, etc! You have no idea how much you'd love people complimenting you! 

### Results:

##### OpenCV
[!OpenCV object detection](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2013%20-%20YOLO%20V2V3V4/YoloOpenCV/savedImage.jpg)
