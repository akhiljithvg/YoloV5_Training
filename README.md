# **YOLOv5n Training on Custom Dataset**

This repository contains a Colab notebook to train a YOLOv5n model on a custom dataset. YOLOv5n is a lightweight variant of the YOLOv5 object detection model, ideal for edge devices and scenarios where efficiency and speed are critical.

---

## **Features**
- Train the YOLOv5n model on your own dataset.
- Use transfer learning by leveraging pre-trained weights.
- Support for custom object classes and annotations.
- Lightweight and optimized model for real-time object detection.
- Integration with popular annotation tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com/).

---

## **Table of Contents**
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Colab Workflow](#colab-workflow)
- [Training Process](#training-process)
- [Results and Evaluation](#results-and-evaluation)
- [Next Steps](#next-steps)
- [Acknowledgments](#acknowledgments)

---

## **Prerequisites**
Before you begin, ensure you have the following:
1. **Google Colab Account**: You can run the training process for free on Colab.
2. **Python 3.8+**: Used for running YOLOv5 scripts and dependencies.
3. **Dataset**: Your dataset should be annotated in YOLO format. Use tools like:
   - [LabelImg](https://github.com/heartexlabs/labelImg) for manual annotation.
   - [Roboflow](https://roboflow.com/) for annotation and dataset preprocessing.
4. **Git**: To clone the YOLOv5 repository.

---

## **Dataset Preparation**
To train your model:
1. **Annotate Images**: Use [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com/) to annotate your dataset.
2. **Format Dataset**: Ensure your dataset follows the structure below:
