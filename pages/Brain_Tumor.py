import streamlit as st
from PIL import Image
import json
from streamlit_lottie import st_lottie
from matplotlib import pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from models import brain_tumor_server_side as tumor


def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie1 = load_lottiefile("/Users/vincentallam/Desktop/code/TSA_2/assets/1708-success (2).json")

col1, col2 = st.columns([4,1])


with col1:
    st.title("Brain Tumor Detection")
    
st.write("Magnetic Resonance Imaging (MRI) is a commonly used imaging tool to detect brain tumors. These images can show the location, size, and shape of a brain tumor, as well as its relationship to nearby brain structures. MRI can also be used to differentiate between benign and malignant tumors, and to monitor the progression of a brain tumor over time.  The data is hand collected from various websites with each and every label verified. The data has four classes of images both in training as well as a testing set: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, No Tumor.")

add_vertical_space(2)
st.subheader("Sample Input:")
st.image("/Users/vincentallam/Desktop/code/TSA_2/assets/gg (19).jpg", width = 400)
add_vertical_space(2)

brain_scan = st.file_uploader("Upload Brain MRI", type =["png","jpg"])

if brain_scan is not None:
    with col2:
        st_lottie(
            lottie1,
            quality = "High",
            width = 100,
            height = 100)
        

    
    
    brain_scan = Image.open(brain_scan).convert('RGB')
    
        
    fig = plt.figure()
    data, pred_time = tumor.predict(brain_scan)
    names = list(data.keys())
    values = list(data.values())
    successIndicator= f"Uploaded Successfully! Finished in {pred_time} seconds."
    st.success(successIndicator, icon="✅")
    add_vertical_space(3)
    plt.style.use('/Users/vincentallam/Desktop/code/TSA_2/assets/graph.mplstyle')
    plt.barh(range(len(data)), values, tick_label=names)
    col3, col4 = st.columns([1,2])  
   
    with col3: 
        st.image(brain_scan)
    
    with col4:
        st.write(fig)
    

      
    max_key = max(data, key=data.get)
    max_value = data[max_key]
    
    if max_key == "glioma_tumor":
        st.header("Results: Glimoa Tumor")
        st.write(f"The brain tumor detection AI is {round(max_value, 2)}% confident  in the diagnosis of a Glioma Tumor being present. Glioma is a type of brain tumor that originates from glial cells, which are supportive cells in the central nervous system. Gliomas can be classified into various grades based on their malignancy, with high-grade gliomas being more aggressive and having a poorer prognosis compared to low-grade gliomas. Symptoms of gliomas may include headache, seizures, changes in mood or behavior, and weakness on one side of the body. Treatment options for gliomas can include surgery, radiation therapy, and chemotherapy. DISCLAIMER : Further clinical evaluation is need to verify patient's status.")
        
    if max_key == "meningioma_tumor":
        st.header("Results: Meningioma Tumor")
        st.write(f"The brain tumor detection AI is {round(max_value, 2)}% confident  in the diagnosis of a Meningioma Tumor being present. A meningioma is a type of brain tumor that arises from the meninges, the protective membranes that surround the brain and spinal cord. Meningiomas are typically slow-growing and benign (non-cancerous), but can cause symptoms by pressing on surrounding brain tissue or by compressing vital structures. Symptoms can include headache, seizure, vision changes, and weakness or numbness in the arms or legs. Treatment options for meningiomas can include surgical removal, radiation therapy, and sometimes observation if the tumor is not causing symptoms. DISCLAIMER : Further clinical evaluation is need to verify patient's status.")
        
    if max_key == "no_tumor":
        st.header("Results: No Tumor")
        st.write(f"CONGRATULATIONS! The Brain Tumor detection AI does not see any significant signs of a tumor.The Brain Tumor detection AI is {round(max_value, 2)}% confident  in the diagnosis of no tumor being present. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
    
    if max_key == "pituitary_tumor":
        st.header("Results: Pituitary Tumor")
        st.write(f"The brain tumor detection AI is {round(max_value, 2)}% confident  in the diagnosis of a Pituitary Tumor being present. A pituitary tumor is a type of brain tumor that develops in the pituitary gland which is a small endocrine gland located at the base of the brain. Th pituitary gland regulates the production and release of hormones that control various bodily functions, including growth, metabolism, and reproductive processes. Pituitary tumors can either be functional or non-functional, depending on whether they produce excess hormones or not. Symptoms of a putuitary tuomr may include headaches, vision changes, hormonal imbalances, and decreased ability to produce certain hormones. Treatment options fr pituitary tumors can include surgical removal, radiation therapy, and medication to regulate hormone levels. DISCLAIMER : Further clinical evaluation is need to verify patient's status.")



with open("/Users/vincentallam/Desktop/code/TSA_2/assets/designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
            
st.markdown("""
<style>
.css-h5rgaw.egzxvld1
{
    visibility: hidden
}
</style>           
            
""", unsafe_allow_html=True) #Removes "Made With Streamlit"