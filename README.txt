BSV_collect.py: Handles coordinate transformation (WGS84 → Baidu BD-09) and Baidu Street View image downloading.

curriculum_learning.py: Acts as a bridge between the training pipeline and data augmentation strategies. Located under `./ultralytics/engine`.

custom_dataset.py: Customized dataset loading and preprocessing module, incorporating epoch-aware logic and curriculum-based scheduling. Located under `./ultralytics/data`.

data_config.yaml: Configuration file for YOLO training dataset.

label_converter.py: Converts polygon annotations from X-AnyLabeling JSON format to YOLO format.

Parking_Generated.py: Contains model inference and visualization examples.

Train_Test.ipynb: Notebook for training and testing the model.

YOLOv8l_CA.yaml: Modified YOLOv8 architecture integrated with Coordinate Attention (CA) mechanism.