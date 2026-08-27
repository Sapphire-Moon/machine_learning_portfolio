
# Importing libraries
import gradio as gd
import pandas as pd 
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model and importance data
try:
    with open("student_rf_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
    with open("feature_importance.pkl", "rb") as f:
        imp_data = pickle.load(f)
except FileNotFoundError:
    print("Error: Required files not found! Run your updated rf_train.py first.")

# Helper function to create chart
def get_importance_plot():
    if imp_data is None:
        return None
    
    # Picking top 10 most influential features
    indices = np.argsort(imp_data['scores'])[-10:]
    names = [imp_data['names'][i] for i in indices]
    scores = [imp_data['scores'][i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, scores, color='#4A90E2')
    ax.set_title("Top 10 Factors Influencing Prediction")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    return fig

# Prediction Function
def predict_and_explain(gender, age, address, famsize, Pstatus, M_Edu, F_Edu, M_Job, F_Job, relationship, smoker, tuition_fee, time_friends, ssc_result):
    
    # Preparing input data like training set
    data = {
        'gender': gender,
        'age': float(age),
        'address': address,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'M_Edu': float(M_Edu),
        'F_Edu': float(F_Edu),
        'M_Job': M_Job,
        'F_Job': F_Job,
        'relationship': relationship,
        'smoker': smoker,
        'tuition_fee': np.log1p(float(tuition_fee)), # Match training log-transform
        'time_friends': float(time_friends),
        'ssc_result': float(ssc_result),
        'Total_Parent_Edu': float(M_Edu) + float(F_Edu) # Match engineered feature
    }
    
    input_df = pd.DataFrame([data])
    
    
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype(object)
    
    # Column order
    column_order = [
        'gender', 'age', 'address', 'famsize', 'Pstatus', 'M_Edu', 'F_Edu',
        'M_Job', 'F_Job', 'relationship', 'smoker', 'tuition_fee',
        'time_friends', 'ssc_result', 'Total_Parent_Edu'
    ]
    input_df = input_df[column_order]
    
    # Predict
    prediction = model.predict(input_df)[0]
    final_gpa = max(0, min(5, prediction))
    
    result_text = f"Predicted HSC Result: {final_gpa:.2f} GPA"
    
    # Return both text result and plot
    return result_text, get_importance_plot()

# Gradio UI Layout
inputs = [
    gd.Radio(["M", "F"], label="Gender"),
    gd.Number(label="Age (Years)", value=18),
    gd.Radio(["Urban", "Rural"], label="Address Type"),
    gd.Radio(["GT3", "LE3"], label="Family size (Greater than 3 / Less than 3)"),
    gd.Radio(["Together", "Apart"], label="Parents' Status"),
    gd.Slider(0, 4, step=1, label="Mother's Education (0-4)"),
    gd.Slider(0, 4, step=1, label="Father's Education (0-4)"),
    gd.Dropdown(["At_home", "Health", "Other", "Services", "Teacher"], label="Mother's Job"),
    gd.Dropdown(["Teacher", "Other", "Services", "Health", "Business", "Farmer"], label="Father's Job"),
    gd.Radio(["Yes", "No"], label="In a Relationship?"),
    gd.Radio(["Yes", "No"], label="Smoker?"),
    gd.Number(label="Annual Tuition Fee (BDT)", value=50000),
    gd.Slider(1, 5, step=1, label="Time with Friends (1-5)"),
    gd.Number(label="SSC Result (GPA)", value=5.0)
]

# Launching
app = gd.Interface(
    fn=predict_and_explain,
    inputs=inputs,
    outputs=[
        gd.Textbox(label="Prediction Result"),
        gd.Plot(label="Model Explanation (XAI)")
    ],
    title="XGBoost: Student Performance Predictor",
    description="This system uses Explainable AI (XAI) to predict and visualize academic outcomes.",
    theme="soft"
)

if __name__ == "__main__":
    app.launch(share=True)