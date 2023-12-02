import interfaces
import streamlit as st
from PIL import Image
import io
import os


#define a seimple streamlit page where I can upload a photo and get a description, than the app uses the description to generate a new image

#load the model

llm = interfaces.construct_openai_mm_llm()
images=st.file_uploader("Upload Images",type=['png','jpeg','jpg'])
if images is not None:
    image=Image.open(images)
    st.image(image,caption='Uploaded Image')
    #create the directory to save the image
    
    if not os.path.exists('./input_images/streamlittemp'):
        os.makedirs('./input_images/streamlittemp')
    #save the image
    image.save('./input_images/streamlittemp/image.jpg')


#images=interfaces.read_all_images('./input_images')
descriptions={}
#Show a button to start the inference
if st.button("Start Inference"):
    #st.write(image_byte_array)
    images = interfaces.read_all_images('./input_images/streamlittemp')
    descriptions = interfaces.inference_on_images(llm, images)
    st.text_area("Description", descriptions)
    
    # Extract only the description inside the "" quotes
    for image_path in descriptions.keys():
        desc = descriptions[image_path]#.split('"')[1]
        print("\n\n***DALLE PROMPT***\n\n" + desc)
        img_url = interfaces.generate_image(desc)
        print(img_url)
        
        # Download the image from the URL and show it
        st.image(img_url)




#show a button to start the image generation



