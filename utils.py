import google.generativeai as genai
import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

def get_gemini_response(query, df, api_key):
    """
    Get response from the Gemini model with the dataset information and the query.
    """
    try:
        # Setup your Gemini API with the provided API key
        genai.configure(api_key=api_key)
        
        df_info = df.info()
        df_sample = df.head()

        # Create a prompt based on the dataset and the query
        prompt = f"""
            Given the following dataset:

            Data types of columns:
            {df_info}

            Sample data (first 5 rows):
            {df_sample}

            Query: {query}

            !!!The response should not contain any numbers!!!. Please provide the following in this exact format and in single line, There should be no  mention of line description (eg Visualization Type) in the response. Just give the output there is no need to rewrite line descriptions:
            Refer to the example response below:
            1. Visulaization Type: (e.g., scatter plot, bar chart, box plot, etc.)$ Python Code: Provide the Python code to generate the specified visualization using libraries like seaborn or matplotlib. The code should only include the code for the plot generation.$ X Axis Label: Provide the label for the x-axis of the plot. Do not include any code here, just the label text.$ Y Axis Label: Provide the label for the y-axis of the plot. Do not include any code here, just the label text.
            2. Python code: Provide the Python code to generate the specified visualization using libraries like seaborn or matplotlib. The code should only include the code for the plot generation.
            3. X Axis Label: Provide the label for the x-axis of the plot. Do not include any code here, just the label text.
            4. Y Axis Label: Provide the label for the y-axis of the plot. Do not include any code here, just the label text.
            For example response : 'Scatter Plot$import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='column1', y='column2', data=df)$Age$bmi'
            Respond with these 4 things seperated by a "$" symbol without any leading or trailing "$" symbols.
            """


        # Send the request to Gemini
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        #   gemini-1.5-flash
        return response.text.strip()
    
    except Exception as e:
        return f"Error while getting Gemini response: {str(e)}"

def parse_gemini_response(response: str):
    """
    Parse the Gemini response into the chart type, chart code, and axis labels.
    """
    try:
        # Split the response by "$"
        response_parts = response.split('$')

        # Check if the split was successful
        if len(response_parts) != 4:
            raise ValueError("The response format is incorrect. Expected 4 parts.")

        # Extract the components from the split response
        visualization_type = response_parts[0].strip()  # Visualization type (e.g., "Scatter Plot")
        chart_code = response_parts[1].strip()  # Python code for generating the chart
        x_label = response_parts[2].strip()  # X-axis label (e.g., "Age")
        y_label = response_parts[3].strip()  # Y-axis label (e.g., "BMI")
        # chart_code = chart_code.replace(" sns", "\nsns").replace(" plt", "\nplt")  # Add line breaks between code

        print("Visualization Type:", visualization_type)
        print("Chart Code:", chart_code)
        print("X-axis Label:", x_label)
        print("Y-axis Label:", y_label)

        return visualization_type, chart_code, x_label, y_label

    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Gemini response:", response)
        raise

def generate_visualizations(df: pd.DataFrame, chart_type: str, chart_code: str, x_label: str, y_label: str):
    """
    Generates the chart using the provided code and labels.
    """
    try:
        # Ensure 'data' is defined as the dataframe `df` in the global context
        globals()['df'] = df

        # Execute the code in the current namespace to generate the plot
        exec(chart_code, globals())

        # After executing the code, we manually label the axes
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Show the plot in Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.write("Error generating the chart:", e)

def get_gemini_response_multiple(df: pd.DataFrame, api_key: str) -> str:
    """
    Get response from the Gemini model with the dataset information and request 5 visualizations in one go.
    """
    try:
        # Setup your Gemini API with the provided API key
        genai.configure(api_key=api_key)
        
        df_info = df.info()
        df_sample = df.head()

        # Create a prompt to generate 5 visualizations
        prompt = f"""
            Given the following dataset:

            Data types of columns:
            {df_info}

            Sample data (first 5 rows):
            {df_sample}

            Please generate 5 important visualizations for the dataset, separated by '&'. 
            Each output should be in the format:
            Visualization Type$Python Code$X Axis Label$Y Axis Label
            Example Responce : 'Scatter Plot$import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='column1', y='column2', data=df)$Age$bmi'&...
            Respond with these 5 outputs, each separated by an '&' symbol. 
        """

        # Send the request to Gemini
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)

        return response.text.strip()
    
    except Exception as e:
        return f"Error while getting Gemini response: {str(e)}"


def parse_gemini_response_multiple(response: str):
    """
    Parse the Gemini response into the chart type, chart code, and axis labels for multiple visualizations.
    """
    try:
        # Split the response by '&' to separate multiple visualizations
        response_parts = response.split('&')

        # Check if the split was successful
        visualizations = []

        for part in response_parts:
            # Split each visualization by '$'
            print(part)
            response_details = part.split('$')
            if len(response_details) == 4:
                chart_type = response_details[0].strip()
                chart_code = response_details[1].strip()
                x_label = response_details[2].strip()
                y_label = response_details[3].strip()

                visualizations.append((chart_type, chart_code, x_label, y_label))
            # else:
                # raise ValueError("The response format is incorrect. Expected 4 parts per visualization.")

        return visualizations

    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Gemini response:", response)
        raise
