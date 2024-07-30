import base64
from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os
import textwrap
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
import io
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import requests
from PIL import Image
from io import BytesIO
from transformers import pipeline
import streamlit.components.v1 as components
# Load environment variables
load_dotenv()

# Configure the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}


def to_markdown(text):
    text = text.replace('•', ' *')
    return st.markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to load OpenAI model and get responses
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Function for translation using LLM
def translate_text(text, source_language, target_language):
    prompt = f"Translate the following text from {source_language} to {target_language}: {text}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to summarize text using LLM
def summarize_text(text):
    prompt = f"Summarize the following text: {text}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to generate code completion using Google's API
def generate_code(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to detect errors in code using Google's API
def detect_errors_in_code(code):
    prompt = f"Analyze the following code for potential errors and suggest improvements:\n\n{code}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to optimize code for performance using Google's API
def optimize_code_for_performance(code):
    prompt = f"Optimize the following code for performance and suggest improvements:\n\n{code}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to generate a table from raw data
def generate_table(raw_data):
    try:
        # Convert raw data into a DataFrame
        df = pd.DataFrame([x.split(',') for x in raw_data.split('\n')])

        # Display the DataFrame as a table
        st.write(df)
    except Exception as e:
        st.write(f"Error in generating table: {e}")

# Function to generate a policy document based on customer prompt
def generate_policy_document(prompt):
    gen_prompt = f"Generate a policy document based on the following requirements:\n\n{prompt}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(gen_prompt)
    return response.text

# Function to generate different types of graphs from raw data
def generate_graph(graph_type, raw_data):
    try:
        # Convert raw data into a DataFrame
        df = pd.DataFrame([x.split(',') for x in raw_data.split('\n')])
        df.columns = df.iloc[0]
        df = df[1:]

        if graph_type == 'Bar':
            fig = px.bar(df, x=df.columns[0], y=df.columns[1:])
        elif graph_type == 'Line':
            fig = px.line(df, x=df.columns[0], y=df.columns[1:])
        elif graph_type == 'Pie':
            fig = px.pie(df, names=df.columns[0], values=df.columns[1])
        elif graph_type == 'Scatter':
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1:])

        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Error in generating {graph_type} chart: {e}")

# Function to generate an image from text using Hugging Face API
def generate_image(prompt):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt}
    )
    image = Image.open(BytesIO(response.content))
    return image

# Set the page configuration
st.set_page_config(page_title="Generatia Official Website", page_icon="🧠")

# HTML and CSS for zig-zag text animation
html_code = """
<style>
@keyframes zigzag {
  0% {
    top: 0;
  }
  25% {
    top: 25%;
  }
  0% {
    down: 0;
  }
  25% {
    down: 25%;
  }
}

.zigzag-text {
  position: absolute;
  animation: zigzag 4s infinite;
  font-size: 40px;
  font-weight: bold;
}
</style>
<div>
</div>
<div>
<br>
</div>
<div class="zigzag-text">Welcome to Generatia Official Website</div>
"""

# Embed the HTML and CSS code into the Streamlit app
components.html(html_code, height=100)

# Add CSS to style the header and footer
st.markdown(
    """
    <style>
    .stApp {
        background-color: Orange;
    }
    .generated-content {
        background-color: Turquoise;
        border: 2px solid black;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add CSS to set the background image
background_image_url = "https://th.bing.com/th/id/R.eef97857d24e8bb90a5c55c732e4328f?rik=3J%2bvcNb6o3EeJA&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fZzQWCHl.jpg&ehk=wOxNkJkTIRTocZ1DiQ7tvIMhWP%2f5IhyPzg2qNzq6XO8%3d&risl=&pid=ImgRaw&r=0"  # Replace with the actual image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({background_image_url});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Initialize the Streamlit app
st.markdown('<h3 style="text-align:center; padding: 10px;">Gen-Ai Tech Solutions</h3>', unsafe_allow_html=True)

# Add the custom HTML element after the title
# # Add the image after the title
# image_url = "https://zealousys.com.au/wp-content/uploads/2024/01/AI-is-Reshaping-Australian-eCommerce-Industry.webp"  # Replace with the actual image URL
# st.image(image_url, caption="Generatia Gen-Ai")

# Add CSS to center the title
st.markdown(
    """
    <style>
    .css-1l02zno {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Dropdown menu for selecting feature
option = st.selectbox("Select an option", ["AskAi","Translator", "Text Summarizer",
                                          "Code Auto-Completion", "Error Detection in Code",
                                          "Optimize Code for Performance", "Generate Table",
                                          "Generate Graph",
                                          "Generate Images"])

# Language options
languages = ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Korean",  "Arabic", "Portuguese"]

if option == "Translator":
    input_text = st.text_input("Input Text: ", key="input_text")
    source_language = st.selectbox("Source Language: ", languages, index=0, key="source_language")
    target_language = st.selectbox("Target Language: ", languages, index=1, key="target_language")
    translate_button = st.button("Translate")

    if translate_button:
        translated_text = translate_text(input_text, source_language, target_language)
        st.markdown("Below is your translated text:")
        st.markdown(f"<div class='generated-content'>{translated_text}</div>", unsafe_allow_html=True)

elif option == "AskAi":
    input_question = st.text_input("Write a prompt here: ", key="input_question")
    ask_button = st.button("Submit")

    if ask_button:
        response = get_gemini_response(input_question)
        st.markdown("Below is the response related to your search:")
        st.markdown(f"<div class='generated-content'>{response}</div>", unsafe_allow_html=True)


elif option == "Text Summarizer":
    input_text = st.text_area("Input Text for Summarization: ", key="input_text_summarizer")
    summarize_button = st.button("Summarize")

    if summarize_button:
        summarized_text = summarize_text(input_text)
        st.markdown("Below is your summarized text:")
        st.markdown(f"<div class='generated-content'>{summarized_text}</div>", unsafe_allow_html=True)

elif option == "Code Auto-Completion":
    code_prompt = st.text_area("Describe what code you want to generate: ", key="code_prompt")
    generate_code_button = st.button("Generate Code")

    if generate_code_button:
        generated_code = generate_code(code_prompt)
        st.markdown("Below is the generated code:")
        st.code(f"<div class='generated-content'>{generated_code}</div>", language='python')

elif option == "Error Detection in Code":
    code_input = st.text_area("Paste your code here for error detection: ", key="code_input")
    detect_errors_button = st.button("Detect Errors")

    if detect_errors_button:
        error_analysis = detect_errors_in_code(code_input)
        st.markdown("Below is the error analysis of your code:")
        st.markdown(f"<div class='generated-content'>{error_analysis}</div>", unsafe_allow_html=True)

elif option == "Optimize Code for Performance":
    code_input = st.text_area("Paste your code here for optimization: ", key="code_input")
    optimize_code_button = st.button("Optimize Code")

    if optimize_code_button:
        optimized_code = optimize_code_for_performance(code_input)
        st.markdown("Below is the optimized code:")
        st.code(f"<div class='generated-content'>{optimized_code}</div>", language='python')

elif option == "Generate Table":
    raw_data = st.text_area("Paste your raw data here (comma-separated values):", key="raw_data")
    generate_table_button = st.button("Generate Table")

    if generate_table_button:
        generate_table(raw_data)

elif option == "Generate Policy Document":
    policy_prompt = st.text_area("Describe the requirements for the policy document:", key="policy_prompt")
    generate_policy_button = st.button("Generate Policy Document")

    if generate_policy_button:
        policy_document = generate_policy_document(policy_prompt)
        st.markdown("Below is your generated policy document:")
        st.markdown(f"<div class='generated-content'>{policy_document}</div>", unsafe_allow_html=True)

elif option == "Generate Graph":
    graph_type = st.selectbox("Select graph type", ["Bar", "Line", "Pie", "Scatter"])
    raw_data = st.text_area("Paste your raw data here (comma-separated values):", key="graph_raw_data")
    generate_graph_button = st.button("Generate Graph")

    if generate_graph_button:
        st.markdown("Below is the generated graph as per your data:")
        generate_graph(graph_type, raw_data)

elif option == "Generate Images":
    image_prompt = st.text_area("Enter your text prompt for the image:", key="image_prompt")
    generate_image_button = st.button("Generate Image")

    if generate_image_button:
        image = generate_image(image_prompt)

        # Convert image to bytes for download link
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Generate download link
        st.markdown("Below is the generated image based on your prompt:")
        st.image(image, use_column_width=True)
        download_link = f"<a href='data:image/png;base64, {img_str}' download='image.png'>Download Image</a>"
        st.markdown(download_link, unsafe_allow_html=True)



# from langchain.llms import OpenAI
# from dotenv import load_dotenv
# import streamlit as st
# import os
# import pathlib
# import textwrap
# import google.generativeai as genai
# from IPython.display import display, Markdown

# def to_markdown(text):
#     text = text.replace('•', '  *')
#     return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# load_dotenv()  # take environment variables from .env

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)

# # Function to load OpenAI model and get responses
# def get_gemini_response(question):
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content(question)
#     return response.text

# # Placeholder function for translation
# def translate_text(text, source_language, target_language):
#     prompt = f"Translate the following text from {source_language} to {target_language}: {text}"
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content(prompt)
#     return response.text

# # Initialize our Streamlit app
# st.set_page_config(page_title="Play With Generative AI")

# st.header("Gemini Application")

# # Dropdown menu for selecting feature
# option = st.selectbox("Select an option", ["Translator", "Ask Query"])

# # Language options
# languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Hindi", "Arabic", "Portuguese"]

# if option == "Translator":
#     input_text = st.text_input("Input Text: ", key="input_text")
#     source_language = st.selectbox("Source Language: ", languages, index=0, key="source_language")
#     target_language = st.selectbox("Target Language: ", languages, index=1, key="target_language")
#     translate_button = st.button("Translate")

#     if translate_button:
#         translated_text = translate_text(input_text, source_language, target_language)
#         st.subheader("Translated Text")
#         st.write(translated_text)

# elif option == "Ask Query":
#     input_question = st.text_input("Input Question: ", key="input_question")
#     ask_button = st.button("Ask the question")

#     if ask_button:
#         response = get_gemini_response(input_question)
#         st.subheader("The Response is")
#         st.write(response)
