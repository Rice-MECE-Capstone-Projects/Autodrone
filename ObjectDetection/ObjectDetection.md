# Object Detection

## 1 Dataset 

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is a large-scale benchmark created by the AISKYEYE team at the Lab of Machine Learning and Data Mining, Tianjin University, China. It contains carefully annotated ground truth data for various computer vision tasks related to drone-based image and video analysis.

VisDrone is composed of 288 video clips with 261,908 frames and 10,209 static images, captured by various drone-mounted cameras. The dataset covers a wide range of aspects, including location (14 different cities across China), environment (urban and rural), objects (pedestrians, vehicles, bicycles, etc.), and density (sparse and crowded scenes). The dataset was collected using various drone platforms under different scenarios and weather and lighting conditions. These frames are manually annotated with over 2.6 million bounding boxes of targets such as pedestrians, cars, bicycles, and tricycles. Attributes like scene visibility, object class, and occlusion are also provided for better data utilization.

![visdrone](images/dataset.png)

### The VisDrone dataset is organized into five main subsets, each focusing on a specific task:

- **Task 1:** Object detection in images
- **Task 2:** Object detection in videos
- **Task 3:** Single-object tracking
- **Task 4:** Multi-object tracking
- **Task 5:** Crowd counting

### Applications

The VisDrone dataset is widely used for training and evaluating deep learning models in drone-based computer vision tasks such as object detection, object tracking, and crowd counting. The dataset's diverse set of sensor data, object annotations, and attributes make it a valuable resource for researchers and practitioners in the field of drone-based computer vision.
The dataset can be downloaded here.

### Usage
To train a YOLOv8n model on the VisDrone dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training page](https://docs.ultralytics.com/modes/train/).

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=640)
```


- **Focus 1:** [Object detection](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/edit/main/ObjectDetection/ObjectDetection.md)
- **Focus 2:** 3d reconstruction
- **Focus 3:** Single-object tracking
- **Focus 4:** Multi-object tracking




