import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ================= LOAD DATA =================
laptop_df = pd.read_csv("Cleaned_laptop_data.csv")
car_df = pd.read_csv("Cleaned_Car_data.csv")

# REMOVE EXTRA COLUMN
if 'Unnamed: 0' in laptop_df.columns:
    laptop_df.drop(columns=['Unnamed: 0'], inplace=True)

if 'Unnamed: 0' in car_df.columns:
    car_df.drop(columns=['Unnamed: 0'], inplace=True)

# ================= LOAD MODELS =================
laptop_model = pickle.load(open("laptop_model.pkl", "rb"))
car_model = pickle.load(open("car_model.pkl", "rb"))

# ================= HOME PAGE =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= CAR PAGE =================
@app.route("/car")
def car_page():
    companies = sorted(car_df["company"].unique())
    years = sorted(car_df["year"].unique(), reverse=True)
    fuels = car_df["fuel_type"].unique()
    return render_template("car.html", companies=companies, years=years, fuels=fuels)

# AJAX: Get models based on company
@app.route("/get_models", methods=["POST"])
def get_models():
    company = request.form.get("company")
    models = car_df[car_df["company"] == company]["name"].unique().tolist()
    return jsonify(models)

# Car Prediction
@app.route("/predict_car", methods=["POST"])
def predict_car():
    data = {
        "name": request.form["name"],
        "company": request.form["company"],
        "year": int(request.form["year"]),
        "kms_driven": int(request.form["kms_driven"]),
        "fuel_type": request.form["fuel_type"]
    }

    df = pd.DataFrame([data])
    prediction = car_model.predict(df)[0]
    return render_template("result.html", prediction=round(prediction, 2))

# ================= LAPTOP PAGE =================
@app.route("/laptop")
def laptop_page():
    return render_template("laptop.html",
        companies=laptop_df["Company"].unique(),
        types=laptop_df["TypeName"].unique(),
        cpu=laptop_df["Cpu_brand"].unique(),
        gpu=laptop_df["Gpu_brand"].unique(),
        os=laptop_df["OS"].unique()
    )

# Laptop Prediction
@app.route("/predict_laptop", methods=["POST"])
def predict_laptop():

    input_dict = {
        "Company": request.form["Company"],
        "TypeName": request.form["TypeName"],
        "Ram": int(request.form["Ram"]),
        "Weight": float(request.form["Weight"]),
        "Touchscreen": int(request.form["Touchscreen"]),
        "Ips": int(request.form["Ips"]),
        "PPI": float(request.form["PPI"]),
        "Cpu_brand": request.form["Cpu_brand"],
        "HDD": int(request.form["HDD"]),
        "SSD": int(request.form["SSD"]),
        "Gpu_brand": request.form["Gpu_brand"],
        "OS": request.form["OS"]
    }

    df = pd.DataFrame([input_dict])

    # 🔥 THIS LINE FIXES YOUR ERROR
    df["Unnamed: 0"] = 0

    # Arrange columns same as model training
    df = df[laptop_model.feature_names_in_]

    prediction = laptop_model.predict(df)[0]

    return render_template("result.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
