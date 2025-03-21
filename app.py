from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None  # Handle the case where the model file is missing

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Transaction type mapping
        transaction_mapping = {
            "CASH_OUT": 1,
            "PAYMENT": 2,
            "CASH_IN": 3,
            "TRANSFER": 4,
            "DEBIT": 5
        }

        # Get form inputs safely
        transaction_type_str = request.form.get("type")
        amount = request.form.get("amount")
        old_balance = request.form.get("oldbalanceOrg")
        new_balance = request.form.get("newbalanceOrig")

        # Validate transaction type
        if transaction_type_str not in transaction_mapping:
            return jsonify({"error": "Invalid transaction type"}), 400

        transaction_type = transaction_mapping[transaction_type_str]

        # Validate and convert numeric inputs
        try:
            amount = float(amount)
            old_balance = float(old_balance)
            new_balance = float(new_balance)

            # Check for negative values
            if amount < 0 or old_balance < 0 or new_balance < 0:
                return jsonify({"error": "Invalid input! Please enter valid values."}), 400

        except ValueError:
            return jsonify({"error": "Invalid numeric values! Please enter valid numbers."}), 400

        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model file not found"}), 500

        # Make prediction
        features = np.array([[transaction_type, amount, old_balance, new_balance]])
        prediction = model.predict(features)[0]

        # Convert prediction result
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
