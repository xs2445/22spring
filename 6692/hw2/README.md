# Lab-JetsonNanoSetup-PretrainedDeployment

Assignment 2, completed assigned files which contains jetson nano setup and pretrained model deployment.

### Files finished
* Jupyter Notebook `JetsonNanoSetup.ipynb`
* Jupyter Notebook `PretrainedDeployment.ipynb`
* Utility file `utils/display.py`

### Functions
```python
# display faces in the query directory
display_faces(query, face_detector, num_images=3)
# display bounding boxes and landmarks of detected faces
display_detection_and_keypoints(query, face_detector, num_images=3)
# face inference of video
video_inference(video_path, face_detector, max_frames=30)
# face inference of webcam
webcam_inference(face_detector)
```

### Usage
```python
from facenet_pytorch import MTCNN
# instantiate a MTCNN model (keep_all shows all the detected faces)
mtcnn = MTCNN(select_largest=False, post_process=False, device=device, keep_all=True)
# structure of the model
print(mtcnn)

from utils.convolution1D import *
# use several downloaded images for inferencing and display detected faces
display_faces("soccer player", mtcnn, 2)
# use several downloaded images for inferencing and show bounding boxes and landmarks of detected faces
display_detection_and_keypoints("soccer player", mtcnn, 3)
# load video and inference the video
video_path = os.path.join(DATA_PATH, 'video_name.mp4')
detected_video_path = video_inference(video_path, mtcnn, max_frames=30)
# open camera and inference the stream from webcam
webcam_inference(mtcnn)
```

### Docker management on Jetson nano
We've already done the docker configuration. Only commonly used commands are listed here.
```bash
# mount docker with --rm, the container will be deleted upon exiting
sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-ml:r32.6.1-py3
# mount the docker with camera connected
sudo docker run -it --rm --runtime nvidia --network host --device /dev/video0 nvcr.io/nvidia/l4t-ml:r32.6.1-py3
```

### Organization of the repo
```

```
