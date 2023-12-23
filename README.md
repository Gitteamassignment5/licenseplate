# Car Number Plate Detection and Recognition using YOLOv3 and Pytesseract

I conducted this research project for my bachelor's thesis, implementing an Automatic Number Plate Detection and Recognition system using YOLOv3 and Pytesseract.

The custom YOLOv3 model was trained specifically for car number plates and utilized as a detection model to identify the location of number plates on cars.

Once the number plate is detected, the image is cropped, and various image processing steps are performed using OpenCV.

The processed image is then passed through Pytesseract OCR to extract the text from the number plate.

## Overview of the project

#![whole_project_overview_](https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/6cf97cc1-efce-4a0e-9bf7-79c424b6b495)


## Technology used

| Development Tools  | Description                                               |
|---------------------|-----------------------------------------------------------|
| Yolov3              | Deep learning algorithm for real-time object detection.   |
| OpenCV              | Computer vision library for image and video processing.   |
| Python              | Programming language used for the development of algorithms. |
| Flask               | Web framework for creating web applications with Python.  |
| Pytesseract OCR    | Optical Character Recognition (OCR) tool for text.         |
| Visual Studio Code | An IDE for coding.                                         |
| Google Colab        | Cloud-based platform for collaborative coding in Python.  |


##  License Plate Detection

The images below showcase successful license plate detection on vehicles.

### Single Car
Figure 1: Single Car

<img width="312" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/4214bafc-f636-4b70-8fc8-4753dbcfd888">

### Multiple Cars

Figure 2: Multiple Cars

<img width="385" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/f40e5e11-4849-4ec5-9308-90396fb57d84">

These images demonstrate successful detection in scenarios with both single and multiple cars, showcasing the versatility of the detection system.

## ROI Extraction and Post-Processing


After license plate detection, a set of preprocessing steps is performed on the extracted license plate to enhance recognition accuracy. The key techniques include:

1. Grayscale Conversion: Simplifying subsequent processing steps and reducing computational complexity by converting the cropped RGB image to grayscale.

2. Gaussian Blur: Applying a 7x7 filter size for Gaussian blur to suppress noise and enhance license plate features.

3. Color Inversion: Enhancing the contrast between the license plate and background through color inversion using bitwise not operation.

4. Binarization: Deriving a binary image from the inverted grayscale image using thresholding (threshold value of 100), accentuating license plate features, and suppressing background noise.

5. Morphological Dilation: Utilizing a 3x3 rectangular kernel in the dilation process to enhance license plate boundaries. This improves contour identification, contributing to a more robust model in various scenarios, including varying angles, lighting conditions, and image noise.

The figure below illustrates the sequence of image processing steps.

<img width="378" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/5c57b3be-de3d-4482-8ed5-998de0b34c10">


## Detection results from various environmental challenges.

Model Performance in different environmental conditions.

<img width="377" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/4afc073d-27f4-49d5-8e75-7718ee3b54a9">

## License Plate Recognition using OCR.

In the OCR phase, Pytesseract is used on preprocessed images to extract alphanumeric characters from the cropped License Plates as shown in Figure below. This is achieved by using the ‘image_to_string()’ function, which retrieves the text. 

For Korean characters, the language parameter is set to ‘Hangul’ .
Post-OCR, the text undergoes post-processing to correct any mistakes, which includes spellchecking and validation against known license plate formats, as well as handling unique characters or symbols found in Korean plates. The final step is to output the recognized text for further use or storage. 
The effectiveness of Pytesseract in recognizing Korean license plates can be significantly influenced by the image's quality, the preprocessing methods, and the OCR engine's training. 

Figure: Text extraction using OCR

<img width="378" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/5277c253-096f-44cc-a946-4b3847841ffd">


https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/e4177189-e3c6-4ab4-a26a-661183b520b9


## Real-time Licence Plate Detection From Video 

