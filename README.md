Scene-Based Image Segmentation

<b>Task Description :</b>
Given a sample video, your task is to segment the images from the video based on the following categories:

1. Indoor scenes
2. Outdoor scenes
3. Scenes with people present
4. Scenes without people present

The segments can be individual clips or combined based on the four categories. The clips don’t need to be mutually exclusive for
the 4 categories.

Name: Atharva Virkar
Contact : 9879009197
Email: atharvavirkar2003@gmail.com
Submission: 10th Jan 2025 (Monday)

## Explanation

The code has been developed by breaking the task into three steps :

- Part-1 Video import and frames creation.
- Part-2 Image Processing and Classification (person & not_a_person).
- Part-3 Classifying indoor & outdoor images using places365.

### 1. Part-1 Video import and frames creation.

This code uses **OpenCV** to extract frames from a video file stored on **Google Drive** and saves them as individual images in a specified folder.

- **Google Drive Mount:** Enables access to the video file and output storage.
- **Directory Creation:** Ensures the output folder exists using `os.makedirs()`.
- **Frame Extraction:** Reads the video frame by frame with `cv2.VideoCapture`.
- **Frame Saving:** Saves each frame as an image file using `cv2.imwrite()` with a sequential naming format (`frame_0001.jpg`, `frame_0002.jpg`, etc.).
- **Completion:** Displays the total number of extracted frames after processing is done.

This automated approach efficiently segments video content for further processing or analysis.

### 2. Part-2 Image Processing and Classification using YOLOv5 (person & not_a_person).

This code leverages the <b>YOLOv5</b> object detection model to classify images into two categories: <b>person</b> and <b>not a person</b>. The detected images are then automatically moved to corresponding folders in <b>Google Drive</b>.

#### **Why YOLOv5 Was Used**

- **Pre-trained on COCO Dataset:** YOLOv5 comes pre-trained on the **COCO (Common Objects in Context)** dataset, which includes 80 classes, such as "person," "car," and "dog."
- **Speed and Efficiency:** YOLO (You Only Look Once) is optimized for real-time object detection, making it suitable for tasks that require fast inference.
- **High Accuracy:** YOLOv5 maintains a balance between detection accuracy and computational efficiency.
- **Flexible Output:** Provides detailed bounding box coordinates and class predictions for detected objects.

#### **Key Steps for Person Detection using YOLOv5**

1. **Environment Setup:**
   - Cloned the YOLOv5 GitHub repository and installed dependencies.
2. **Model Loading:**

   - Loaded the pre-trained `yolov5s` model from PyTorch Hub.

3. **Drive Integration:**

   - Mounted Google Drive to access input images and store the classified results.

4. **Directory Creation:**

   - Created "person" and "not_a_person" folders to organize the output images.

5. **Image Processing:**

   - Iterated through all images in the specified folder.
   - Used YOLOv5 to detect objects in each image and extract class labels.

6. **Classification and Storage:**

   - If the class **'person'** was detected, the image was moved to the "person" folder.
   - If no person was detected, the image was moved to the "not_a_person" folder.

### 3. Part-3 Classifying indoor and outdoor images using places365.

This code classifies images into **indoor** and **outdoor** categories using the **Places365 ResNet18** pretrained model. The images are then saved into respective folders based on their classification.

#### **Why Places365 Was Used**

- **Extensive Scene Categories:** The Places365 model is trained on over **365 scene categories** (e.g., airport, bedroom, desert, park), making it ideal for environmental classification.
- **Scene Context Understanding:** Unlike models focusing on object detection, Places365 specializes in understanding contextual environments, enabling accurate indoor/outdoor classification.
- **Pretrained Model Availability:** Provides a ready-to-use ResNet18 architecture with scene-centric weights, reducing the need for additional training.

#### **Key Steps for Indoor-Outdoor Image Classification Using Places365**

1. **Environment Setup:**

   - Installed `torch` and `torchvision`.
   - Imported essential libraries for image processing.

2. **Model Loading:**

   - Loaded **ResNet18 Places365 pretrained model** and updated the fully connected layer for 365 scene categories.

3. **Mapping Files:**

   - Downloaded indoor-outdoor (`IO_places365.txt`) and scene category (`categories_places365.txt`) mapping files.

4. **Folder Creation:**

   - Created target directories for **indoor** and **outdoor** classified images.

5. **Image Processing:**

   - Applied transformations such as resizing and normalization to input images.

6. **Classification:**

   - Used the model to predict scene categories and mapped predictions to **indoor** or **outdoor** labels.

7. **Storage:**
   - Saved images to respective folders based on classification results.

This setup efficiently classifies and organizes images based on environmental context.

---

## **Challenges Faced During Development**

1. **Video Segmentation:**  
   Initially, I lacked knowledge about video segmentation. I had to invest time researching how to divide a video into individual frames and save those frames as images using OpenCV for later classification.

2. **Selecting the Right YOLO Version:**  
   There were multiple versions of the YOLO model available, which created confusion. I needed to carefully choose the most suitable version for my task and ensure its compatibility with my project requirements.

3. **Folder Creation for Image Classification:**  
   Setting up proper folder structures for organizing classified images posed a challenge. Specifically, I needed to create folders named "person" and "not_a_person" and manage the process of copying images to their respective folders based on classification.

4. **Dataset Availability:**  
   The unavailability of specific datasets for indoor and outdoor images made it difficult to build a custom dataset. As a result, I decided to use a pretrained model (Places365) to achieve the indoor/outdoor classification instead of developing a project-specific dataset from scratch.

5. **Integration of the Project Components:**  
   One of the main challenges was integrating the various sections of the project (video frame extraction, object detection using YOLOv5, and classification with Places365). Ensuring seamless interaction between all components was crucial for smooth execution.

6. **Inexperience with streamlit**

   I've not worked on Streamlit yet so i tried to get my hands on it. But it was getting difficult to integrate the whole code with streamlit.

---

## **Potential Improvements**

1. **Leveraging YOLO for Classification:**  
   YOLO is an excellent model for object detection, and I aim to enhance my skills with YOLO. Instead of using two separate models for classification, I plan to explore how to classify "person," "not_a_person," "indoor," and "outdoor" all using a single YOLO model. This would simplify the process and improve the model's efficiency.

2. **Training a Custom Model with a Dedicated Dataset:**  
   If I can access a dataset that includes a sufficient number of indoor and outdoor images, I would like to train my own model. This could potentially lead to improved results and better performance in classifying scenes specific to my needs.

3. **Transitioning to Local Systems:**  
   While the current implementation works on Google Colab, I would like to optimize the code to run on any local system. This would provide more flexibility in executing the project without relying on cloud-based platforms like Colab.

4. **Adding Streamlit**
   I would like to add streamlit and create an app as per suggested where the video can be uploaded and later on other processed would continue. I would really like to connect the whole code with Streamlit as my proficiency level gets better to make it more interactive and to improve user experience.
   <br><br><br>

##### Note:

Please be aware that this PDF file was generated from a Markdown (.md) file. As a result, some formatting irregularities may be present in the final PDF version. The content, however, remains accurate and is intended to provide a comprehensive overview of the project.
