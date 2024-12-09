import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

# Initialize the BLIP model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the state dict (weights) from the saved file
state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# If you are using a GPU, move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to load and process the image, and generate a caption
def generate_caption_from_image(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate the caption
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=70, num_beams=1, temperature=1.5, top_k=50, top_p=0.9, do_sample=True)

    # Decode the generated caption
    generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_caption


# Streamlit app
st.title("Radiology Report Generation")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    image_path = image  # Replace with your image path
    generated_caption = generate_caption_from_image(image_path)
    print("Generated Caption:", generated_caption)

    st.title('Patient Report:')
    st.write(generated_caption.upper())                       