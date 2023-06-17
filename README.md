# Insight AI

Introduction 
------------
Insight AI is a website created by My friends Raghauv Saravanan [@sraghauv](https://www.github.com/sraghauv), Antony Sajesh [@ant-saj123](https://www.github.com/ant-saj123), Chris Abraham and I. This website was created for a national technology competition known as TSA. Our product is a medical detection AI that is able to read brain and chest scans to produce accurate diagnoses in a fraction of the time it takes a typical radiologist to interpret and give a diagnosis. This product placed 5th in the state of Texas for the software development event.

Conditions that are diagnosed
----------------------------
 - Alzheimer's severity classifier: The alzheimer's AI model takes in a brain MRI scan and then classifys it into 4 severity's those being <br /> [Non_demented, Very-mild_Demted, Mild_demented, Moderate_demented] 
 - Brain tumor detector: Brain tumor detector takes in a brain MRI scan and then determines what type tumor if any is present <br /> [No_tumor, Glimoma_tumor, Meningioma_tumor, Pituitary_tumor]
 - Pneumonia detector: The Pneumonia detector takes in chest CT scans and then determines if the patient has pneumonia or not <br /> [No_pneumonia, Pneumonia]

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

Model Statistics
----------------
 - Alzheimer's severity model: test_accuracy: 99.93% @ 100 scans/secound
 - Brain tumor detection model: test_accuracy: 99.95% @ 93.2 scans/secound
 - Pnuemonia detection model: test_accuracy: 99.99% @ 302 scans/secound <br />
 - *Note scans/secound may vary due to hardware differences.* 

Insight AI's Impact
------------------
The purpose and main function of this website is to save precious time that doctors spend when analyzing different scans in order to save someone's life as well as eliminate human error. Medical Diagnoses are a complex process. There are a lot of abnormalities with a variety of symptoms and sometimes even overlaps in symptoms of many medical conditions. This makes it really difficult to completely understand the nature of the different medical conditions, such as Brain Tumors and Alzheimer’s. Also, a Specialist, like a Neurosurgeon or Radiologist, is required for MRI and X-Ray analysis. Oftentimes in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’s. So an automated system on the cloud such as Insight AI would be helpful to doctors all around the world.
 







