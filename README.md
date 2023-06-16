# Insight AI

Introduction 
------------
Insight AI is a website created by My friends Raghauv Saravanan [@sraghauv](https://www.github.com/sraghauv), Antony Sajesh [@ant-saj123](https://www.github.com/ant-saj123), Chris Abraham and I. This website was created for a national technology competition known as TSA. Our product is a medical detection AI that is able to read brain and chest scans to produce accurate diagnoses in a fraction of the time it takes a typical radiologist to interpret and give a diagnosis. This product placed 5th in the state of Texas for the software development event.

Technical Overview
------------------
The project is split between a front end and an AI backend

  - Front-end: The base of the front end and the webpages for each of the medical detectors were created on a Python package known as streamlit and were stylized and integrated with CSS and Json animations and themes. 
  
  - Back-end: The AI predictions were created using a Python deep learning library known as PyTorch. The models were based on the state-of-the-art Vision Transformer Neural Network (ViT) architecture. This model architecture, coupled with transfer learning from the ImageNet1K dataset, allows all three predictive models to efficiently produce accurate diagnoses. 


Repositiory Breakdown
---------------------
- Homepage.py: Is the intoduction page to the website that gives a general introduction to the product and provides intructions on how to use the website.
- Pages folder: Contains the webpages that corespond to each type of preditve model 




