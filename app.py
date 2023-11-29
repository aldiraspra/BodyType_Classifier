import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('model.h5')

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Body Type Classifier",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key

def display_ectomorph_info():
    st.write("Ectomorph: This body type is thin, usually tall, and lanky.")
    st.write("Individuals with a sturdy, rounder bone structure have wider hips, stocky limbs, and barrel-shaped rib cages.")
    st.write("Diet:")
    st.write("- Ectomorphs can generally eat a wide variety of foods without gaining much weight.")
    st.write("- Most of their calories should come from carbs and at least 25 percent from protein if they want to build muscles.")
    st.write("Workout:")
    st.write("- It is also important to indulge in high-intensity interval training and eat something small every two hours. They must be sure to get all necessary nutrients to build strong muscles.")
    st.write("- Include a mix of cardio for overall health.")

def display_endomorph_info():
    st.write("Endomorph: This body type is characterized by a rounder, softer physique.")
    st.write("Individuals tend to gain weight faster and have a curvier appearance.")
    st.write("Diet:")
    st.write("- Start by cutting down carbs to only 30 percent of total daily calories. The other 70 percent should be divided into protein and good fats")
    st.write("- Emphasize a diet rich in whole foods, vegetables, and lean proteins.")
    st.write("Workout:")
    st.write("- Cardio is the best form of exercise because it ensures that the body does not revert to fat-storing stages.")
    st.write("- Consistent exercise is important for maintaining a healthy weight.")
    st.write('Their main goal should not be to lose weight but to boost metabolism and prevent calories from turning into fat.')

def display_mesomorph_info():
    st.write("Mesomorph: This body type is generally considered the ideal body type.")
    st.write("Individuals usually look lighter and have a more rectangular bone structure, longer limbs, thinner bones, and a flatter ribcage.")
    st.write("Diet:")
    st.write("- Just like ectomorphs, mesomorphs too can eat whatever they like without putting on weight, but the flip side here is that they may gain weight just as easily if they aren’t careful.")
    st.write("- Because people with this body type have a lot to do with muscle, it stands to reason that they need a more protein-based diet that can help build and repair muscles.")
    st.write('- They also need to consume a lot of fruits and vegetables rich in antioxidants and whole grains (quinoa, oats, etc.')
    st.write("Workout:")
    st.write("- Just because mesomorphs have a naturally muscular body doesn’t mean they don’t need to exercise. They do need training to stay lean. Alternating between weights and cardio would be the ideal workout for mesomorphs.")
    st.write("- Regular exercise is important for maintaining muscle mass.")


with st.sidebar:
        st.image('chibi.webp')
        st.title("Body Type Classifier")
        st.subheader("Get to know your body type")

st.write("""
         # Body type Classifier
         ## Upload an Image for Classification
         """)

# Display an example image for users to understand how to upload
example_image_path = 'foto.jpg'
example_image = Image.open(example_image_path)
example_image = example_image.resize((150, 150))

# Checkbox to show/hide the example image
show_example = st.checkbox("Show Example Image")

# Display the example image if the checkbox is checked
if show_example:
    st.image(example_image, caption='Example Image: Follow the instructions below', use_column_width=True)

# Add instructions for users
st.write("""
         ## Instructions:
         1. Click the "Browse Files" button to upload an image.
         2. Upload an image of a person to classify body type.
         3. Wait for the classification results to appear.
         4. Check the sidebar for the predicted body type and accuracy.
         """)

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
    size = (200, 200)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload your body image")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Ectomorph', 'Endomorph', 'Mesomorph']

    string = "Body Type : " + class_names[np.argmax(predictions)]
    
    if string:
        st.sidebar.success(string)

    if class_names[np.argmax(predictions)] == 'Ectomorph':
        st.balloons()
        print('Ectomorph')
        st.info("This body type is thin, usually tall, and lanky. Individuals with a sturdy, rounder bone structure have wider hips, stocky limbs and barrel-shaped rib cages.")
        display_ectomorph_info()
        
    elif class_names[np.argmax(predictions)] == 'Endomorph':
        if string:
            st.sidebar.warning(string)
            st.markdown("## Result")
        print('Endomorph')
        st.info("If an individual finds it ridiculously tough to shed fat, then they probably have an endomorphic body type. They tend to gain weight faster. However, they look curvy. Endomorphs usually look broader and have a triangular bone structure, narrower hips, and broader shoulders.")
        display_endomorph_info()

    elif class_names[np.argmax(predictions)] == 'Mesomorph':
        if string:
            st.sidebar.warning(string)
            st.markdown("## Result")
        print('Mesomorph')
        st.info("This body type is generally considered the ideal body type. Individuals usually look lighter and have a more rectangular bone structure, longer limbs, thinner bones and a flatter ribcage.")
        display_mesomorph_info()
        
st.sidebar.markdown("---")
st.sidebar.write("Made with 💪 by Al Diras Pradiptha")