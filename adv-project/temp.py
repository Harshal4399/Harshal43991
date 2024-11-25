import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import plotly.graph_objects as go

# Initialize Gemini API
genai.configure(api_key="AIzaSyCzr1UTv9-7zlP5GaAmmglur1TMYCJHsxA")
model = genai.GenerativeModel("gemini-pro")


# Function to determine key columns and appropriate visualizations
def analyze_dataset(df):
    response = model.generate_content(
        """You are a data visualization expert. Analyze the following dataset columns and their datatypes, then suggest appropriate visualizations.
        
        Dataset columns and types:
        """ + str(df.dtypes.to_string()) + """

        Respond ONLY with a JSON array of visualization suggestions. Each suggestion should be an object with exactly these fields:
        - "type": visualization type (bar, pie, scatter, line, box, histogram, heatmap, or violin)
        - "x": column name for x-axis
        - "y": column name for y-axis (optional for histogram)

        Example format:
        [
            {
                "type": "bar",
                "x": "column1",
                "y": "column2"
            }
        ]

        Ensure the response is valid JSON with proper quotes around strings. Do not include any explanations or additional text.
        """
    )

    try:
        # Clean the response text to ensure it's valid JSON
        cleaned_response = response.text.strip()
        # If response starts with ``` and ends with ```, remove them
        if cleaned_response.startswith('```') and cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[3:-3]
        # Remove any "json" or other language identifiers
        if cleaned_response.startswith('json'):
            cleaned_response = cleaned_response[4:]

        cleaned_response = cleaned_response.strip()

        # Parse the JSON
        res = json.loads(cleaned_response)
        return res
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse visualization suggestions: {str(e)}")
        # Return a default visualization as fallback
        return [
            {
                "type": "bar",
                "x": df.columns[0],
                "y": df.columns[1] if len(df.columns) > 1 else df.columns[0]
            }
        ]


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

            # If the dataframe has only one column, it might be semicolon separated
            if len(df.columns) == 1:
                # Reset the file pointer to the beginning
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Dataset:")
        st.write(df.head())

        # Initial Visualizations Section
        st.header("Initial Dataset Visualizations")
        visualizations = analyze_dataset(df)
        charts = generate_visualizations(df, visualizations)

        for ind, chart in enumerate(charts):
            st.plotly_chart(chart, key=f"initial_chart_{ind}")

        # Query Section with separate container
        st.header("Query Analysis")
        query = st.text_input("Ask a question about the dataset:")
        if query:
            # Create a new container for query results
            query_container = st.container()

            with query_container:
                viz_check_response = model.generate_content(
                    f"""Given this user query: "{query}"
                    Determine if it requires a data visualization.
                    Return ONLY "yes" or "no" without any additional text or explanation."""
                )

                needs_viz = viz_check_response.text.strip().lower() == "yes"

                if needs_viz:
                    # Get visualization specifications
                    viz_response = model.generate_content(
                        f"""Based on this dataset with columns and their types:
                        {str(df.dtypes.to_string())}

                        And this user query: "{query}"

                        Respond ONLY with a JSON object containing visualization specifications with these exact fields:
                        - "type": visualization type (bar, pie, scatter, line, box, histogram, heatmap, or violin)
                        - "x": column name for x-axis
                        - "y": column name for y-axis (optional for histogram)
                        - "title": chart title

                        Example format:
                        {{"type": "bar", "x": "column1",
                            "y": "column2", "title": "Chart Title"}}
                        """
                    )

                    try:
                        # Clean and parse the JSON response
                        cleaned_response = viz_response.text.strip()
                        if cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[3:-3]
                        if cleaned_response.startswith('json'):
                            cleaned_response = cleaned_response[4:]
                        cleaned_response = cleaned_response.strip()

                        viz_spec = json.loads(cleaned_response)

                        try:
                            if viz_spec["type"] == "pie":
                                values_col = viz_spec["y"]
                                names_col = viz_spec["x"]

                                if df[values_col].dtype in ['int64', 'float64']:
                                    agg_df = df.groupby(names_col)[
                                        values_col].sum().reset_index()
                                    fig = px.pie(
                                        agg_df, values=values_col, names=names_col, title=viz_spec["title"])
                                else:
                                    counts = df[names_col].value_counts(
                                    ).reset_index()
                                    counts.columns = [names_col, 'count']
                                    fig = px.pie(
                                        counts, values='count', names=names_col, title=viz_spec["title"])
                            elif viz_spec["type"] == "bar":
                                fig = px.bar(
                                    df, x=viz_spec["x"], y=viz_spec["y"], title=viz_spec["title"])
                            elif viz_spec["type"] == "scatter":
                                fig = px.scatter(
                                    df, x=viz_spec["x"], y=viz_spec["y"], title=viz_spec["title"])
                            elif viz_spec["type"] == "line":
                                fig = px.line(
                                    df, x=viz_spec["x"], y=viz_spec["y"], title=viz_spec["title"])
                            elif viz_spec["type"] == "box":
                                fig = px.box(
                                    df, x=viz_spec["x"], y=viz_spec["y"], title=viz_spec["title"])
                            elif viz_spec["type"] == "histogram":
                                fig = px.histogram(
                                    df, x=viz_spec["x"], title=viz_spec["title"])
                            elif viz_spec["type"] == "violin":
                                fig = px.violin(
                                    df, x=viz_spec["x"], y=viz_spec["y"], title=viz_spec["title"])

                            st.plotly_chart(
                                fig, key=f"query_viz_{hash(query)}")

                        except Exception as e:
                            st.error(
                                f"Error generating visualization: {str(e)}")

                    except json.JSONDecodeError as e:
                        st.error(
                            f"Error parsing visualization specification: {str(e)}")

                # Text response
                text_response = model.generate_content(
                    f"""Based on this dataset with columns:
                    {str(df.dtypes.to_string())}

                    Answer this user query: "{query}"
                    """
                )
                st.write(text_response.text)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
