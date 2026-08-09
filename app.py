from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("finance_model.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


def calculate_health_score(income, savings, debt_ratio, credit, emergency, surplus):
    """
    Calculates a 0-100 Financial Health Score from raw form inputs.
    Weights: Credit 30%, Savings Ratio 25%, Debt Ratio 20%,
             Emergency Fund 15%, Monthly Surplus 10%
    """
    monthly_income = income / 12 if income else 1

    # Credit score: dataset range is 350-900
    credit_component = min(max((credit - 350) / (900 - 350) * 100, 0), 100)

    # Savings ratio: savings relative to annual income, capped at 2x
    savings_ratio = savings / income if income else 0
    savings_component = min(savings_ratio / 2 * 100, 100)

    # Debt-to-income ratio: lower is better, capped at 5.0
    debt_component = max(0, 100 - (min(debt_ratio, 5) / 5 * 100))

    # Emergency fund: 6 months is considered fully healthy
    emergency_component = min(emergency / 6 * 100, 100)

    # Monthly surplus relative to monthly income
    surplus_ratio = surplus / monthly_income if monthly_income else 0
    surplus_component = min(max(surplus_ratio, 0) * 100, 100)

    score = (
        credit_component * 0.30
        + savings_component * 0.25
        + debt_component * 0.20
        + emergency_component * 0.15
        + surplus_component * 0.10
    )

    return round(min(max(score, 0), 100))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    income = float(request.form["income"])
    expenses = float(request.form["expenses"])
    savings = float(request.form["savings"])
    loan = float(request.form["loan"])
    credit = int(request.form["credit"])
    dependents = int(request.form["dependents"])
    emergency = float(request.form["emergency"])
    surplus = float(request.form["surplus"])
    debt = float(request.form["debt"])
    existing_loan = request.form["existing_loan"]
    experience = request.form["experience"]
    risk = request.form["risk"]
    employment = request.form["employment"]
    goal = request.form["goal"]

    existing_loan = encoders["existing_loan"].transform([existing_loan])[0]
    experience = encoders["investment_experience"].transform([experience])[0]
    risk = encoders["risk_tolerance"].transform([risk])[0]
    employment = encoders["employment_type"].transform([employment])[0]
    goal = encoders["investment_goal"].transform([goal])[0]

    # ↓↓↓ REPLACE THE OLD "data = pd.DataFrame(...)" BLOCK WITH THIS ↓↓↓
    data = pd.DataFrame([[
        age,
        income,
        expenses,
        savings,
        existing_loan,
        loan,
        credit,
        experience,
        risk,
        dependents,
        employment,
        emergency,
        goal,
        surplus,
        debt
    ]], columns=[
        "age",
        "annual_income",
        "monthly_expenses",
        "savings",
        "existing_loan",
        "loan_amount",
        "credit_score",
        "investment_experience",
        "risk_tolerance",
        "dependents",
        "employment_type",
        "emergency_fund_months",
        "investment_goal",
        "monthly_surplus",
        "debt_to_income_ratio"
    ])
    # ↑↑↑ END REPLACEMENT ↑↑↑

    prediction = model.predict(data)
    result = target_encoder.inverse_transform(prediction)[0]
    probability = model.predict_proba(data).max() * 100

    health_score = calculate_health_score(
        income=income,
        savings=savings,
        debt_ratio=debt,
        credit=credit,
        emergency=emergency,
        surplus=surplus
    )

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2),
        health_score=health_score
    )

if __name__ == "__main__":
    app.run(debug=True)