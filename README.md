# Insight AI

Introduction 
------------
Insight AI is a website created by My friends Raghauv Saravanan [@sraghauv](https://www.github.com/sraghauv), Antony Sajesh [@ant-saj123](https://www.github.com/ant-saj123), Chris Abraham and I. This website was created for a national technology competition known as TSA. Our product is a medical detection AI that is able to read brain and chest scans to produce accurate diagnoses in a fraction of the time it takes a typical radiologist to interpret and give a diagnosis. This product placed 5th in the state of Texas for the software development event.

Conditions that are diagnosed
----------------------------
 - Alzimers severity: The Alzimers AI model takes in a brain MRI scan and then classifys it into 4 severity's those being  

Technical Overview
------------------
The project is split between a front end and an AI backend

  - Front-end: The base of the front end and the webpages for each of the medical detectors were created on a Python package known as streamlit and were stylized and integrated with CSS and Json animations and themes. 
  
  - Back-end: The AI predictions were created using a Python deep learning library known as PyTorch. The models were based on the state-of-the-art Vision Transformer Neural Network (ViT) architecture. This model architecture, coupled with transfer learning from the ImageNet1K dataset, allows all three predictive models to efficiently produce accurate diagnoses. 


Repositiory Breakdown
---------------------
- Homepage.py: Is the intoduction page to the website that gives a general introduction to the product and provides intructions on how to use the website.
- Pages folder: This folder contains the webpages that correspond to each type of predictive model. It allows the user to upload medical scans and then returns the result with a percent confidence graph and a short paragraph explaining the condition, along with possible treatments and professionals they should consult for their condition
- Assets folder: This folder contains the CSS themes, Json animations and images, to stylize the website and make the end user experience better than just the base templates provided by the streamlit package. 

- Training Models folder: contains the AI python scripts used to generate and train the ViT model on the medical scan data, these scripts then returns model.pth files which are then used to initalize the models created on the on-premise python AI scripts   
- Model folder: This folder contains the on-premise AI python scripts that perform inference on the uploaded medical scans, the scripts takes in model.pth files and then generates a neural network built on the weigths and biases of the model.pth file. Once the neural network is intialized it then performs inference on the medical scans and returns the percent confidence of the diagnosis and the time it took for the scan to be analyzed and interpreted. 







