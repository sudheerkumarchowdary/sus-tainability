import streamlit as st
from PIL import Image
import tempfile
import os
import google.generativeai as genai
from detection import run_inference

# Configure the Generative AI with your API key
genai.configure(api_key="AIzaSyAsdivNEUd6GB9yuQEKCgCo-PhWi-slY2Y")  # Replace with your actual API key

# Streamlit App title
st.title("SuS - Tainability")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_image_path = temp_file.name

    # Run inference directly using the run_inference function
    st.write("Running inference...")

    try:
        # Call the run_inference function and pass the temporary image path
        detected_objects = run_inference(temp_image_path)

        # Format the output to display only unique object names
        unique_objects = set(detected_objects)  # Use a set to avoid duplicates
        formatted_output = ', '.join(unique_objects)  # Join the unique objects

        if formatted_output:  # Check if any objects were detected
            st.write(f'Reusable objects in the image: {formatted_output}')  # Removed brackets
            
            # Create a prompt for the Generative AI model
            prompt = f"Give a brief on how to reuse these objects: {formatted_output}"

            # Generate content using the Gemini model
            model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the model
            response = model.generate_content(prompt)

            # Extract and format the response
            suggestions = response.text.strip()

            # Split the suggestions into lines and display them as bullet points
            if suggestions:
                lines = suggestions.split('\n')
                bullet_points = [line.strip() for line in lines if line.strip()]
                
                # Display bullet points
                st.write("Suggestions on how to reuse these objects:")
                for point in bullet_points:
                    st.write(f"- {point}")

        else:
            st.write("No reusable objects detected.")

    except Exception as e:
        st.write("Error running inference:", str(e))

    # Clean up the temporary file
    os.remove(temp_image_path)

# Adding a chat feature using Generative AI
st.write("## Chat with the Model")

# Create a text input for the user to ask questions
user_input = st.text_input("Ask anything related to sustainability, waste management, or reuse:")

if user_input:
    try:
        # Create a prompt based on the user's question
        prompt = f"Answer this sustainability-related question: {user_input}"

        # Generate content using the Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Extract the response text
        raw_response = response.text

        # Remove any markdown-like headers (e.g., ## What is a Sustainable Life?)
        cleaned_response = '\n'.join([line for line in raw_response.split('\n') if not line.startswith('#')])

        # Display the cleaned response to the user
        st.write(cleaned_response)

    except Exception as e:
        st.write("Error during chat:", str(e))
