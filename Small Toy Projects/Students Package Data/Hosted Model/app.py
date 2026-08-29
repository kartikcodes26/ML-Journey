from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "student_package_model_200k.pkl"
)

model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    hours_college = float(request.form["hours_college"])
    hours_skills = float(request.form["hours_skills"])

    college_tier = request.form["college_tier"]

    mhtcet = float(request.form["mhtcet"])
    jee = float(request.form["jee"])

    gender = request.form["gender"]

    # Create DataFrame with EXACT same column names
    input_data = pd.DataFrame({
        "Hours_College_Syllabus": [hours_college],
        "Hours_Skill_Development": [hours_skills],
        "College_Tier": [college_tier],
        "MHTCET_Percentile": [mhtcet],
        "JEE_Percentile": [jee],
        "Gender": [gender]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
