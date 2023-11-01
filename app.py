from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained machine learning model
with open("model.pickle", "rb") as model_file:
    model = pickle.load(model_file)


# Define a function to map user-friendly options to numerical values
def map_to_numeric(value, mapping):
    return mapping.get(value, 0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Map user-friendly options to numerical values
        data["gender"] = map_to_numeric(data["gender"], {"male": 1, "female": 0})
        data["hypertension"] = map_to_numeric(data["hypertension"], {"Yes": 1, "No": 0})
        data["heart_disease"] = map_to_numeric(
            data["heart_disease"], {"Yes": 1, "No": 0}
        )
        data["ever_married"] = map_to_numeric(data["ever_married"], {"Yes": 1, "No": 0})
        data["work_type"] = map_to_numeric(
            data["work_type"],
            {
                "Govt job": 4,
                "Never worked": 0,
                "Private": 3,
                "Self-employed": 2,
                "children": 1,
            },
        )
        data["Residence_type"] = map_to_numeric(
            data["Residence_type"], {"urban": 0, "rural": 1}
        )
        data["smoking_status"] = map_to_numeric(
            data["smoking_status"],
            {"formerly smoked": 2, "never smoked": 1, "smokes": 3, "unknown": 0},
        )

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])

        # Make a prediction using the pre-trained model
        prediction = model.predict(input_data)
        prediction_probability = model.predict_proba(input_data)[:, 1]

        # Determine the prediction result
        result = "Stroke Risk" if prediction[0] == 1 else "No Stroke Risk"

        response = {"result": result, "probability": prediction_probability[0]}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
