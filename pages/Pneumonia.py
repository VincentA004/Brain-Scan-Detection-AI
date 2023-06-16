import streamlit as st
from PIL import Image
import numpy as np
import json
from streamlit_lottie import st_lottie
import pandas as pd
from matplotlib import pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from models import chest_xray_server_side as xray



def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie1 = load_lottiefile("/Users/vincentallam/Desktop/code/TSA_2/assets/1708-success (2).json")

col1, col2 = st.columns([4,1])


with col1:
    st.title("Pneumonia Detection")
    

st.write("Chest X-ray imaging is a commonly used diagnostic tool to detect pneumonia, an infection of the lungs. A chest X-ray produces an image of the chest, including the lungs, heart, blood vessels, and bones. In the case of pneumonia, the lungs may appear cloudy or hazy, indicating fluid or inflammation in the air spaces within the lungs. This fluid accumulation can make it difficult for oxygen to pass from the air sacs of the lungs into the bloodstream, leading to symptoms such as shortness of breath, cough, and chest pain. The datasets collected chest X-ray images of Normal and Pneumonia affected patients and every image was verified. It has two outputs labeled as Normal and Pneumonia. ")

add_vertical_space(2)
st.subheader("Sample Input:")
st.image("/Users/vincentallam/Desktop/code/TSA_2/assets/BACTERIA-92115-0001.jpeg", width = 600)
add_vertical_space(2)

x_ray_scan = st.file_uploader("Upload Chest CT", type =["png","jpg","jpeg"])
 
if x_ray_scan is not None:
    x_ray_scan = Image.open(x_ray_scan).convert('RGB')
    
    with col2:
        st_lottie(
            lottie1,
            quality = "High",
            width = 100,
            height = 100)
        
 
    
    
        
    fig = plt.figure()
    data, pred_time = xray.predict(x_ray_scan)
    names = list(data.keys())
    values = list(data.values())
    
    
    successIndicator= f"Uploaded Successfully! Finished in {pred_time} seconds."
    st.success(successIndicator, icon="✅")
    add_vertical_space(3)
    
    plt.style.use('/Users/vincentallam/Desktop/code/TSA_2/assets/graph.mplstyle')
    plt.barh(range(len(data)), values, tick_label=names)
    
    col3, col4 = st.columns([1,2])  
    
    with col3: 
        st.image(x_ray_scan)
        
    with col4:
        st.write(fig)
    
    
      
    max_key = max(data, key=data.get)
    max_value = data[max_key]
  
    if max_key == "normal":
        st.header("Results: Normal")
        st.write(f"CONGRATULATIONS! The Pneumonia detection AI does not see any significant signs of Pneumonia. The Pneumonia detection AI is {round(max_value, 2)}% confident  in the diagnosis of the patient not having Pneumonia. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
        
        
    if max_key == "pneumonia":
        st.header("Results: Pneumonia")
        st.write(f"The Pneumonia detection AI is {round(max_value, 2)}% confident  in the diagnosis of Pneumonia being present. Pneumonia is a type of lung infection that causes inflammation of the air sacs in one or both lungs. It can be caused by various organisms such as bacteria, viruses, fungi, or parasites. Pneumonia can range from mild to severe, and symptoms may include cough, fever, chills, shortness of breath, chest pain, fatigue, and muscle aches. In some cases, people may also experience sweating, rapid breathing, and confusion. Treatment for pneumonia depends on the cause of the infection, and can include antibiotics, antiviral drugs, or supportive care such as oxygen therapy. Vaccination can also help prevent pneumonia. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
       
        
    


st.markdown("""
<style>
.css-h5rgaw.egzxvld1
{
    visibility: hidden
}
</style>           
            
""", unsafe_allow_html=True) #Removes "Made With Streamlit"

        
with open("/Users/vincentallam/Desktop/code/TSA_2/assets/designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)


st.markdown("""
<style>
.c
{
    visibility: hidden
}
</style>           
            
""", unsafe_allow_html=True) #Removes "Made With Streamlit"