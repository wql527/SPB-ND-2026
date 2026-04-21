Scripts description:

1、BSV_collect.py: Handles coordinate transformation (WGS84 → Baidu BD-09) and Baidu Street View image downloading.

2、curriculum_learning.py: Acts as a bridge between the training pipeline and data augmentation strategies. Located under `./ultralytics/engine`.

3、custom_dataset.py: Customized dataset loading and preprocessing module, incorporating epoch-aware logic and curriculum-based scheduling. Located under `./ultralytics/data`.

4、data_config.yaml: Configuration file for YOLO training dataset.

5、label_converter.py: Converts polygon annotations from X-AnyLabeling JSON format to YOLO format.

6、Parking_Generated.py: Contains model inference and visualization examples.

7、Train_Test.ipynb: Notebook for training and testing the model.

8、YOLOv8l_CA.yaml: Modified YOLOv8 architecture integrated with Coordinate Attention (CA) mechanism.
