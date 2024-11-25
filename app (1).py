import streamlit as st
import pandas as pd
import os
from utils import generate_visualizations, get_gemini_response_multiple, parse_gemini_response_multiple, get_gemini_response, parse_gemini_response

# Set up Google Gemini API key
GENAI_API_KEY = os.environ["GOOGLE_API_KEY"]  # Replace with your actual Gemini API key

# Streamlit UI components
st.set_page_config(page_title="AI-Vizz : Data Visualization and Insights", layout="wide")

# Title
# st.title("Welcome to **AI-Vizz**")
st.title("AI-Vizz : Data Visualization and Insights")
st.markdown("""
    **AI-Vizz** is a powerful web application that enables you to easily upload datasets and generate visualizations
    powered by AI. Simply upload your dataset, and let VizAI suggest the most insightful visualizations for your data.
    You can also ask natural language questions to gain deeper insights into the dataset.
""")

# File upload for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Load and display the uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head(), width=1000)  # Adjusted the size for a larger display

    # Natural language query input
    st.write("### Ask a question about the dataset:")
    query = st.text_input("Type your question here:")
    
    if query:
        # Get response from Gemini model
        response = get_gemini_response(query, df, GENAI_API_KEY)
        st.subheader("Gemini's Answer")
        st.write(response)

        # Parse the response to get chart details
        chart_type, chart_code, x_label, y_label = parse_gemini_response(response)
        
        # Generate and display the visualizations
        generate_visualizations(df, chart_type, chart_code, x_label, y_label)

    # Generate 5 initial visualizations based on the dataset
    st.write("### Important visualizations...")

    # Get 5 visualizations in one go from the model
    response = get_gemini_response_multiple(df, GENAI_API_KEY)
    
    # Parse the response to get chart details
    visualizations = parse_gemini_response_multiple(response)

    for chart_type, chart_code, x_label, y_label in visualizations:
        # Generate and display each visualization
        generate_visualizations(df, chart_type, chart_code, x_label, y_label)
