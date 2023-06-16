import streamlit as st
from PIL import Image
import json
from streamlit_lottie import st_lottie
from matplotlib import pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from models import alzheimers_server_side as al



def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie1 = load_lottiefile("/Users/vincentallam/Desktop/code/TSA_2/assets/1708-success (2).json")

col1, col2 = st.columns([4,1])


with col1:
    st.title("Alzheimer's Severity")
    
    
st.write("Magnetic Resonance Imaging (MRI) is a non-invasive imaging technique that can be used to detect Alzheimer's disease by capturing images of the brain. MRI scans can show the shrinkage of the brain that occurs in Alzheimer's, as well as changes in the structure of specific regions of the brain that are commonly affected by the disease. These changes can indicate the presence of Alzheimer's even before symptoms appear.  The data is hand collected from various websites with each and every label verified. The data consists of MRI images. The data has four classes of images both in training as well as a testing set: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented.")

add_vertical_space(2)
st.subheader("Sample Input:")
st.image("/Users/vincentallam/Desktop/code/TSA_2/assets/mildDem1.jpg", width = 400)
add_vertical_space(2)

brain_scan = st.file_uploader("Upload Brain MRI", type =["png","jpg","jpeg"])
  
if brain_scan is not None:
    with col2:
        st_lottie(
            lottie1,
            quality = "High",
            width = 100,
            height = 100)
    brain_scan = Image.open(brain_scan).convert('RGB')
    
    
    
    fig = plt.figure()
    plt.style.use('/Users/vincentallam/Desktop/code/TSA_2/assets/graph.mplstyle')
    data, pred_time = al.predict(brain_scan)
    names = list(data.keys())
    values = list(data.values())
    successIndicator= f"Uploaded Successfully! Finished in {pred_time} seconds."
    st.success(successIndicator, icon="✅")
    plt.barh(range(len(data)), (values), tick_label=names, )
    add_vertical_space(3)
    col3, col4 = st.columns([1,2])  
    
    
   
    with col3: 
        st.image(brain_scan)
        
    with col4:
        st.write(fig)
        
    max_key = max(data, key=data.get)
    max_value = data[max_key]
    
    if max_key == "MildDemented":
        st.header("Results: Mild Demented")
        st.write(f"The  Alzheimer's detection AI is {round(max_value, 2)}% confident  in the diagnosis of the patient suffering from mild dementia. Dementia is a term used to describe a decline in cognitive function, including memory, reasoning, judgment, and communication skills, to the point where it interferes with daily life. It is a progressive brain disorder that can occur as a result of Alzheimer's disease. Symptoms of dementia can vary depending on the underlying cause and the stage of the condition, but may include memory loss, difficulty with language, disorientation, changes in mood and behavior, and problems with activities of daily living. The term \"mildly demented\" is often used to describe patients who has a mild form of dementia. Mild dementia typically refers to the early stage of the condition, when symptoms are just starting to appear and the person is still able to function independently. However, as the condition progresses, symptoms become more severe and can interfere with daily life. It is important to note that dementia is a complex and progressive condition, and the level of severity and progression can vary greatly from person to person.There is currently no cure for dementia, but treatment options may include medications, lifestyle changes, and support services to help manage symptoms and improve quality of life. DISCLAIMER: Further clinical evaluation is need to verify patient’s status.")
        
    if max_key == "ModerateDemented":
        st.header("Results: Moderate Demented")
        st.write(f"The  Alzheimer's detection AI is {round(max_value, 2)}% confident  in the diagnosis of the patient suffering from moderate dementia. Dementia is a term used to describe a decline in cognitive function, including memory, reasoning, judgment, and communication skills, to the point where it interferes with daily life. It is a progressive brain disorder that can occur as a result of Alzheimer's disease. \"Moderately demented\" is a term used to describe a patient with a moderate stage of dementia, a progressive brain disorder characterized by a decline in cognitive function. At this stage, the person may have significant difficulty with memory recall and communication, and may require assistance with some activities of daily living. They may also experience confusion, disorientation, changes in mood and behavior, and difficulty with problem solving and decision making. In the moderate stage of dementia, the person may need more support and assistance, but is still able to live independently in some cases. It is important to note that the progression of dementia varies greatly from person to person, and seeking early treatment can help delay its progression and improve quality of life. There is currently no cure for dementia, but treatment options may include medications, lifestyle changes, and support services. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
        
    if max_key == "NonDemented":
        st.header("Results: Non-Demented")
        st.write(f"CONGRATULATIONS! The Alzheimer's detection AI does not see any significant signs of dementia. The  Alzheimer's detection AI is {round(max_value,2)}% confident  in the diagnosis of the patient is not suffering from dementia. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
    
    if max_key == "VeryMildDemented":
        st.header("Results: Very Mild Demented")
        st.write(f"The  Alzheimer's detection AI is {round(max_value, 2)}% confident  in the diagnosis of the patient suffering from very mild dementia. Dementia is a term used to describe a decline in cognitive function, including memory, reasoning, judgment, and communication skills, to the point where it interferes with daily life. It is a progressive brain disorder that can occur as a result of Alzheimer's disease. Very mildly demented is a term used to describe a patient who has the early stages of dementia, characterized by minor declines in cognitive function. In this stage, the person may have difficulty with memory recall, experience mild confusion, and have trouble with language. Despite these symptoms, they are still able to live independently and perform most daily activities without significant difficulty. As the condition progresses, symptoms may become more pronounced and interfere with daily life. It is important to note that the progression of dementia varies greatly from person to person, and early detection and treatment can help delay its progression and improve quality of life. There is currently no cure for dementia, but treatment options may include medications, lifestyle changes, and support services. DISCLAIMER: Further clinical evaluation is need to verify patient's status.")
    
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