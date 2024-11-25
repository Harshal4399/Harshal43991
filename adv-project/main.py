import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import plotly.graph_objects as go

# Initialize Gemini API
genai.configure(api_key="AIzaSyAu6s-gdY4it5PpgYSbSeD-3ZYN6VdNYeQ")
model = genai.GenerativeModel("gemini-1.5-flash")


# Function to determine key columns and appropriate visualizations
def analyze_dataset(df):
    response = model.generate_content(
        """You are a data visualization expert. Analyze the following dataset columns and their datatypes, then suggest possible visualizations.
        
        Dataset columns and types:
        """ + str(df.dtypes.to_string()) + """

        Return ONLY a JSON array of visualization suggestions in this exact format:
        [
            {
                "type": "bar",
                "x": "column1",
                "y": "column2"
            }
        ]

        Rules:
        1. Response must be valid JSON
        2. Use double quotes for strings
        3. Only include supported chart types: bar, pie, box, scatter, line, histogram, heatmap, violin
        4. Ensure suggested columns exist in the dataset
        5. Return ONLY the JSON array, no other text
        """
    )

    try:
        # Clean the response text to ensure it's valid JSON
        cleaned_response = response.text.strip()
        if not cleaned_response.startswith('['):
            # Extract JSON array if it's wrapped in other text
            start = cleaned_response.find('[')
            end = cleaned_response.rfind(']') + 1
            if start != -1 and end != 0:
                cleaned_response = cleaned_response[start:end]

        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        st.error(
            f"Failed to parse visualization suggestions. Using default visualizations.")
        # Return a default visualization if parsing fails
        return [{"type": "bar", "x": df.columns[0], "y": df.columns[1]}]


# Function to generate visualizations
def generate_visualizations(df, visualizations):
    charts = []
    for viz in visualizations:
        try:
            if viz["type"] == "pie":
                values_col = viz["y"]
                names_col = viz["x"]

                if df[values_col].dtype in ['int64', 'float64']:
                    agg_df = df.groupby(names_col)[
                        values_col].sum().reset_index()
                    fig = px.pie(agg_df, values=values_col, names=names_col)
                else:
                    counts = df[names_col].value_counts().reset_index()
                    counts.columns = [names_col, 'count']
                    fig = px.pie(counts, values='count', names=names_col)
            else:
                if viz["type"] == "bar":
                    fig = px.bar(df, x=viz["x"], y=viz["y"])
                elif viz["type"] == "box":
                    fig = px.box(df, x=viz["x"], y=viz["y"])
                elif viz["type"] == "scatter":
                    fig = px.scatter(df, x=viz["x"], y=viz["y"])
                elif viz["type"] == "line":
                    fig = px.line(df, x=viz["x"], y=viz["y"])
                elif viz["type"] == "histogram":
                    fig = px.histogram(df, x=viz["x"])
                elif viz["type"] == "heatmap":
                    fig = px.imshow(df.pivot_table(
                        index=viz["x"], columns=viz["y"], aggfunc='size').fillna(0))
                elif viz["type"] == "violin":
                    fig = px.violin(df, x=viz["x"], y=viz["y"])
            charts.append(fig)
        except Exception as e:
            st.warning(f"Could not generate {viz['type']} chart: {str(e)}")
            continue

    return charts


# Streamlit app
st.title("Automated Data Visualization System")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            # First try comma separator
            df = pd.read_csv(uploaded_file)

            # If the dataframe has only one column, try semicolon separator
            if len(df.columns) == 1:
                # Reset the file pointer to the beginning
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')

                # If still one column, try space separator
                if len(df.columns) == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep='\s+')
        else:
            df = pd.read_excel(uploaded_file)

        # Display dataset info
        st.info(f"Dataset shape: {df.shape} (rows × columns)")
        st.write("Dataset Preview:")
        st.write(df.head())

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Analyze dataset and generate visualizations

    visualizations = analyze_dataset(df)
    charts = generate_visualizations(df, visualizations)

    st.write("Generated Visualizations:")
    for ind, chart in enumerate(charts):
        st.plotly_chart(chart, key=ind)

    # Natural language query
    query = st.text_input("Ask a question about the dataset:")
    if query:
        # First, get the response from Gemini
        prompt = f"""
        Analyze this dataset with columns and their data types:
        {str(df.dtypes.to_string())}

        User query: {query}

        If the query requires a visualization, respond in this exact JSON format:
        {{
            "response": "your text response here",
            "visualization": {{
                "type": "chart_type",
                "x": "column_name",
                "y": "column_name"
            }}
        }}

        If no visualization is needed, respond in this format:
        {{
            "response": "your text response here"
        }}

        Use only these chart types: bar, pie, box, scatter, line, histogram, heatmap, violin
        Ensure all column names exist in the dataset.
        """

        response = model.generate_content(prompt)

        try:
            # Parse the response as JSON
            result = json.loads(response.text)

            # Display the text response
            st.write(result["response"])

            # If visualization is requested, create and display the chart
            if "visualization" in result:
                viz = result["visualization"]
                try:
                    if viz["type"] == "pie":
                        values_col = viz["y"]
                        names_col = viz["x"]
                        if df[values_col].dtype in ['int64', 'float64']:
                            agg_df = df.groupby(names_col)[
                                values_col].sum().reset_index()
                            fig = px.pie(
                                agg_df, values=values_col, names=names_col)
                        else:
                            counts = df[names_col].value_counts().reset_index()
                            counts.columns = [names_col, 'count']
                            fig = px.pie(counts, values='count',
                                         names=names_col)
                    else:
                        if viz["type"] == "bar":
                            fig = px.bar(df, x=viz["x"], y=viz["y"])
                        elif viz["type"] == "box":
                            fig = px.box(df, x=viz["x"], y=viz["y"])
                        elif viz["type"] == "scatter":
                            fig = px.scatter(df, x=viz["x"], y=viz["y"])
                        elif viz["type"] == "line":
                            fig = px.line(df, x=viz["x"], y=viz["y"])
                        elif viz["type"] == "histogram":
                            fig = px.histogram(df, x=viz["x"])
                        elif viz["type"] == "heatmap":
                            fig = px.imshow(df.pivot_table(
                                index=viz["x"], columns=viz["y"], aggfunc='size').fillna(0))
                        elif viz["type"] == "violin":
                            fig = px.violin(df, x=viz["x"], y=viz["y"])

                    print(fig)
                    st.plotly_chart(fig)

                except Exception as e:
                    st.warning(
                        f"Could not generate the requested visualization: {str(e)}")

        except json.JSONDecodeError:
            # If response is not JSON, just display the text
            st.write(response.text)
