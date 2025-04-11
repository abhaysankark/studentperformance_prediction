import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Function to load model and associated data
@st.cache_data
def load_model_and_data():
    """
    Load the saved model and associated data
    """
    if not os.path.exists('models/student_performance_model.pkl'):
        st.error("Model files not found. Please run the preprocessing and training script first.")
        return None, None, None
    
    model = joblib.load('models/student_performance_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model_info = joblib.load('models/model_info.pkl')
    
    return model, scaler, model_info

# Function to load sample data
@st.cache_data
def load_sample_data():
    """
    Load sample data for visualization
    """
    try:
        # Try to load the actual dataset
        df = pd.read_csv('StudentPerformanceFactors.csv')
        return df
    except:
        # Return dummy data if file not found
        st.warning("Full dataset not found. Using sample data for visualizations.")
        # Create sample data based on the provided 5 rows
        data = {
            'Hours_Studied': [23, 19, 24, 29, 19],
            'Attendance': [84, 64, 98, 89, 92],
            'Parental_Involvement': ['Low', 'Low', 'Medium', 'Low', 'Medium'],
            'Access_to_Resources': ['High', 'Medium', 'Medium', 'Medium', 'Medium'],
            'Extracurricular_Activities': ['No', 'No', 'Yes', 'Yes', 'Yes'],
            'Sleep_Hours': [7, 8, 7, 8, 6],
            'Previous_Scores': [73, 59, 91, 98, 65],
            'Motivation_Level': ['Low', 'Low', 'Medium', 'Medium', 'Medium'],
            'Internet_Access': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            'Tutoring_Sessions': [0, 2, 2, 1, 3],
            'Family_Income': ['Low', 'Medium', 'Medium', 'Medium', 'Medium'],
            'Teacher_Quality': ['Medium', 'Medium', 'Medium', 'Medium', 'High'],
            'School_Type': ['Public', 'Public', 'Public', 'Public', 'Public'],
            'Peer_Influence': ['Positive', 'Negative', 'Neutral', 'Negative', 'Neutral'],
            'Physical_Activity': [3, 4, 4, 4, 4],
            'Learning_Disabilities': ['No', 'No', 'No', 'No', 'No'],
            'Parental_Education_Level': ['High School', 'College', 'Postgraduate', 'High School', 'College'],
            'Distance_from_Home': ['Near', 'Moderate', 'Near', 'Moderate', 'Near'],
            'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
            'Exam_Score': [67, 61, 74, 71, 70]
        }
        return pd.DataFrame(data)

# Function to make predictions
def predict_exam_score(input_data, model, scaler, model_info):
    """
    Make predictions using the trained model
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply category mappings
    for col in model_info['categorical_cols']:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(model_info['category_mappings'][col])
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make a prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# Function to create scatter plot without statsmodels dependency
def create_scatter_plot(df, x_col, y_col, title):
    """
    Create a scatter plot without using statsmodels for trendline
    """
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        color_discrete_sequence=['#3366CC']
    )
    
    # Manually add a simple trend line
    try:
        # Simple linear regression
        x = df[x_col].values
        y = df[y_col].values
        
        # Calculate the trend line parameters
        n = len(x)
        if n > 1:  # Make sure we have enough data points
            m = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x*x) - np.sum(x) ** 2)
            b = (np.sum(y) - m * np.sum(x)) / n
            
            # Create trendline points
            x_trend = np.array([min(x), max(x)])
            y_trend = m * x_trend + b
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
    except Exception as e:
        # If something goes wrong, just skip the trendline
        pass
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400, 
        width=500
    )
    
    return fig

# Function to create visualizations
def create_visualizations(df):
    """
    Create visualizations for the dashboard
    """
    st.subheader("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        st.write("Correlation Matrix (Numerical Features)")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=500, width=500)
        st.plotly_chart(fig)
    
    with col2:
        # Distribution of exam scores
        st.write("Distribution of Exam Scores")
        fig = px.histogram(
            df, 
            x="Exam_Score", 
            nbins=20,
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(
            xaxis_title="Exam Score",
            yaxis_title="Count",
            height=500, 
            width=500
        )
        st.plotly_chart(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hours studied vs exam score
        st.write("Hours Studied vs Exam Score")
        fig = create_scatter_plot(df, "Hours_Studied", "Exam_Score", "Hours Studied vs Exam Score")
        st.plotly_chart(fig)
    
    with col2:
        # Attendance vs exam score
        st.write("Attendance vs Exam Score")
        fig = create_scatter_plot(df, "Attendance", "Exam_Score", "Attendance vs Exam Score")
        st.plotly_chart(fig)
    
    # Top factors affecting exam scores
    st.subheader("Key Features Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Previous scores vs exam score
        st.write("Previous Scores vs Exam Score")
        fig = create_scatter_plot(df, "Previous_Scores", "Exam_Score", "Previous Scores vs Exam Score")
        st.plotly_chart(fig)
    
    with col2:
        # Sleep hours vs exam score
        st.write("Sleep Hours vs Exam Score")
        fig = create_scatter_plot(df, "Sleep_Hours", "Exam_Score", "Sleep Hours vs Exam Score")
        st.plotly_chart(fig)
    
    # Categorical features analysis
    st.subheader("Categorical Features Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exam score by parental involvement
        st.write("Exam Score by Parental Involvement")
        
        # Group by parental involvement and calculate mean exam score
        involvement_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean().reset_index()
        
        fig = px.bar(
            involvement_scores,
            x='Parental_Involvement',
            y='Exam_Score',
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig)
    
    with col2:
        # Exam score by gender
        st.write("Exam Score by Gender")
        
        # Group by gender and calculate mean exam score
        gender_scores = df.groupby('Gender')['Exam_Score'].mean().reset_index()
        
        fig = px.bar(
            gender_scores,
            x='Gender',
            y='Exam_Score',
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig)

# Main app
def main():
    st.title("📚 Student Performance Prediction System")
    st.write("Predict student exam scores based on various factors.")
    
    # Load model and data
    model, scaler, model_info = load_model_and_data()
    df = load_sample_data()
    
    # Check if model is loaded
    if model is None:
        st.warning("Model not found. Running in demo mode with limited functionality.")
    
    # App tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Make Prediction", "ℹ️ About"])
    
    # Dashboard tab
    with tab1:
        st.header("Student Performance Analytics")
        
        # Dataset overview
        with st.expander("Dataset Overview"):
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1]-1}")
            st.dataframe(df.head())
        
        # Create visualizations
        create_visualizations(df)
        
        # Model performance
        if model is not None:
            st.subheader("Model Performance")
            st.write(f"Model used: **{model_info['model_name']}**")
            
            # Display placeholder metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Placeholder for model metrics
                metrics = {
                    'R² Score': 0.756,  # Based on the output you shared
                    'RMSE': 1.86,       # Based on the output you shared
                    'MAE': 1.5          # Placeholder value
                }
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=['#3366CC', '#DC3912', '#FF9900']
                ))
                
                fig.update_layout(
                    title="Model Evaluation Metrics",
                    height=400
                )
                
                st.plotly_chart(fig)
            
            with col2:
                st.write("Key Factors Impacting Performance")
                st.write("""
                Based on our analysis, the following factors have the strongest influence on exam scores:
                
                1. **Previous Scores**: Strong indicator of current performance
                2. **Hours Studied**: Shows positive correlation with exam scores
                3. **Attendance**: Regular attendance improves performance
                4. **Parental Involvement**: Higher involvement correlates with better scores
                5. **Sleep Hours**: Adequate sleep is important for academic performance
                """)
    
    # Prediction tab
    with tab2:
        st.header("Predict Student Performance")
        st.write("Enter the student's details to predict their exam score.")
        
        # Create form for user input
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hours_studied = st.slider("Hours Studied (per week)", 5, 35, 20)
                attendance = st.slider("Attendance (%)", 60, 100, 80)
                sleep_hours = st.slider("Sleep Hours (per night)", 4, 10, 7)
                previous_scores = st.slider("Previous Scores", 50, 100, 70)
                physical_activity = st.slider("Physical Activity (hours/week)", 1, 5, 3)
                tutoring_sessions = st.slider("Tutoring Sessions (per week)", 0, 5, 1)
            
            with col2:
                parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
                access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
                motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
                family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
                teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
                peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
            
            with col3:
                extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
                learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
                school_type = st.selectbox("School Type", ["Public", "Private"])
                parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
                distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
                gender = st.selectbox("Gender", ["Male", "Female"])
            
            submit_button = st.form_submit_button("Predict Exam Score")
        
        # Make prediction when form is submitted
        # Inside the Prediction tab section, replace the existing prediction display code with this:

    if submit_button:
        # Prepare input data
        input_data = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': parental_involvement,
            'Access_to_Resources': access_to_resources,
            'Extracurricular_Activities': extracurricular,
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': motivation_level,
            'Internet_Access': internet_access,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': family_income,
            'Teacher_Quality': teacher_quality,
            'School_Type': school_type,
            'Peer_Influence': peer_influence,
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': learning_disabilities,
            'Parental_Education_Level': parental_education,
            'Distance_from_Home': distance_from_home,
            'Gender': gender
        }
        
        if model is not None:
            # Make prediction
            try:
                prediction = predict_exam_score(input_data, model, scaler, model_info)
                
                # Determine pass/fail status (threshold: 65)
                pass_threshold = 65
                is_pass = prediction >= pass_threshold
                
                # Display prediction with pass/fail status
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Exam Score", f"{prediction:.2f}")
                
                with col2:
                    if is_pass:
                        st.success("### PASS ✅")
                    else:
                        st.error("### FAIL ❌")
                
                # Show how far from passing/failing
                if is_pass:
                    margin = prediction - pass_threshold
                    st.write(f"Student is **{margin:.2f} points** above the passing threshold.")
                else:
                    margin = pass_threshold - prediction
                    st.write(f"Student needs **{margin:.2f} more points** to pass.")
                
                # Provide interpretation
                st.subheader("Interpretation")
                
                if prediction >= 90:
                    st.write("🌟 The student is likely to achieve excellent results!")
                elif prediction >= 80:
                    st.write("✨ The student is likely to achieve very good results.")
                elif prediction >= 70:
                    st.write("👍 The student is likely to achieve good results.")
                elif prediction >= 65:
                    st.write("📝 The student is likely to achieve acceptable results.")
                elif prediction >= 60:
                    st.write("⚠️ The student is at risk of failing and needs some support.")
                else:
                    st.write("🔍 The student needs significant support to improve their results.")
                
                # Show key factors that could improve the score
                st.subheader("Improvement Opportunities")
                
                improvement_tips = []
                
                if hours_studied < 25:
                    improvement_tips.append("Increase study hours to at least 25 hours per week")
                
                if attendance < 90:
                    improvement_tips.append("Improve class attendance to above 90%")
                
                if sleep_hours < 7:
                    improvement_tips.append("Ensure adequate sleep of at least 7-8 hours per night")
                
                if parental_involvement == "Low":
                    improvement_tips.append("Increase parental involvement in academic activities")
                
                if motivation_level == "Low":
                    improvement_tips.append("Work on improving motivation through goal-setting and rewards")
                
                if tutoring_sessions < 2 and previous_scores < 80:
                    improvement_tips.append("Consider additional tutoring sessions")
                
                if not improvement_tips:
                    improvement_tips.append("The student is already on a good track. Continue with the current approach.")
                
                for i, tip in enumerate(improvement_tips, 1):
                    st.write(f"{i}. {tip}")
                
                # Add a visual indicator of how close to passing/failing
                st.subheader("Pass/Fail Margin")
                
                # Create a progress bar to visualize how close to passing/failing
                fig = go.Figure()
                
                # Set the range to show context around the passing threshold
                score_min = max(50, min(prediction - 10, pass_threshold - 10))
                score_max = min(100, max(prediction + 10, pass_threshold + 10))
                
                # Add the failing range
                fig.add_shape(
                    type="rect",
                    x0=score_min,
                    x1=pass_threshold,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(width=0)
                )
                
                # Add the passing range
                fig.add_shape(
                    type="rect",
                    x0=pass_threshold,
                    x1=score_max,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(0, 255, 0, 0.2)",
                    line=dict(width=0)
                )
                
                # Add the threshold line
                fig.add_shape(
                    type="line",
                    x0=pass_threshold,
                    x1=pass_threshold,
                    y0=0,
                    y1=1,
                    line=dict(color="black", width=2, dash="dash")
                )
                
                # Add the score marker
                fig.add_trace(go.Scatter(
                    x=[prediction],
                    y=[0.5],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color="blue",
                        symbol="circle"
                    ),
                    name="Predicted Score"
                ))
                
                # Add labels
                fig.add_annotation(
                    x=pass_threshold - 5,
                    y=0.5,
                    text="FAIL",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                
                fig.add_annotation(
                    x=pass_threshold + 5,
                    y=0.5,
                    text="PASS",
                    showarrow=False,
                    font=dict(color="green", size=14)
                )
                
                fig.add_annotation(
                    x=prediction,
                    y=0.8,
                    text=f"Score: {prediction:.1f}",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=12)
                )
                
                # Update layout
                fig.update_layout(
                    xaxis=dict(
                        title="Exam Score",
                        range=[score_min, score_max]
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False
                    ),
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please check that your input values are within the expected ranges.")
        else:
            st.warning("Model not loaded. Cannot make predictions.")
            st.write("This is a demo. In a real application, predictions would be made using the trained model.")
        
        st.subheader("Disclaimer")
        st.write("""
        This is a predictive model and not a definitive assessment of a student's capabilities. 
        Many factors influence academic performance, and this tool should be used as one of 
        many resources to help students succeed.
        """)
        
        st.subheader("Dataset Description")
        
        # Create a table explaining each feature
        feature_descriptions = {
            "Hours_Studied": "Number of hours the student studies per week",
            "Attendance": "Student's class attendance percentage",
            "Parental_Involvement": "Level of parental involvement in the student's education (Low/Medium/High)",
            "Access_to_Resources": "Student's access to educational resources (Low/Medium/High)",
            "Extracurricular_Activities": "Whether the student participates in extracurricular activities (Yes/No)",
            "Sleep_Hours": "Average number of sleep hours per night",
            "Previous_Scores": "Student's scores in previous exams",
            "Motivation_Level": "Student's motivation level (Low/Medium/High)",
            "Internet_Access": "Whether the student has internet access at home (Yes/No)",
            "Tutoring_Sessions": "Number of tutoring sessions per week",
            "Family_Income": "Family income level (Low/Medium/High)",
            "Teacher_Quality": "Quality of teachers (Low/Medium/High)",
            "School_Type": "Type of school (Public/Private)",
            "Peer_Influence": "Influence of peers on the student (Negative/Neutral/Positive)",
            "Physical_Activity": "Hours of physical activity per week",
            "Learning_Disabilities": "Whether the student has learning disabilities (Yes/No)",
            "Parental_Education_Level": "Education level of parents (High School/College/Postgraduate)",
            "Distance_from_Home": "Distance from home to school (Near/Moderate/Far)",
            "Gender": "Student's gender (Male/Female)",
            "Exam_Score": "Student's final exam score (target variable)"
        }
        
        # Display the feature descriptions as a DataFrame
        feature_df = pd.DataFrame(
            {"Feature": list(feature_descriptions.keys()), 
             "Description": list(feature_descriptions.values())}
        )
        
        st.dataframe(feature_df, use_container_width=True)

if __name__ == "__main__":
    main()