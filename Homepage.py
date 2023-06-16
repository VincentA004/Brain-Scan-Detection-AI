
import streamlit as st


with open("/assets/designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([6,1], gap="small")


with col1:
    st.title("Insight AI")
    
with col2: 
    st.image("assets/image 2.png")


st.subheader("Creators: Vincent Allam, Chris Abraham, Antony Sajesh, and Raghauv Saravanan")
st.markdown("""---""")

st.write("This project is meant to save precious time that doctors spend when analyzing different scans in order to save someone's life as well as eliminate human error. Medical Diagnoses are a complex process. There are a lot of abnormalities with a variety of symptoms and sometimes even overlaps in symptoms of many medical conditions . This makes it really difficult to completely understand the nature of the different medical conditions, such as Brain Tumors and Alzheimer’s. Also, a Specialist, like a Neurosurgeon or Radiologist, is required for MRI and X-Ray analysis. Oftentimes in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’s. So an automated system on Cloud can solve this problem. Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms such as Vision Transformer neural networks (ViT) aided with Transfer Learning, would be helpful to doctors all around the world. Users are able to upload their own scans and get predictions on those scans of what illness or issue the patient has.")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://e1.pxfuel.com/desktop-wallpaper/761/68/desktop-wallpaper-plain-backgrounds-for-websites-plain-studio-background.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#add_bg_from_url()

st.markdown("""
<style>
.css-h5rgaw.egzxvld1
{
    visibility: hidden
}
</style>           
            
""", unsafe_allow_html=True) #Removes "Made With Streamlit"





