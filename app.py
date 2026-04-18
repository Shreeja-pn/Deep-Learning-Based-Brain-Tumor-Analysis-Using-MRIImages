from flask import Flask, render_template, request
import os

from inference import load_model, predict
from utils.classification import classify_tumor
from utils.severity import calculate_severity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
MODEL_PATH = "model/brats_3d_unet_full.pth"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load segmentation model once
model = load_model(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():

    try:

        # -----------------------------------
        # Get patient details from form
        # -----------------------------------
        patient_name = request.form.get("patient_name")
        age = request.form.get("age")
        gender = request.form.get("gender")

        # -----------------------------------
        # Get uploaded folder files
        # -----------------------------------
        files = request.files.getlist("folder")

        if len(files) == 0:
            return "❌ Please upload a BraTS patient folder."

        t1_path = None
        t1ce_path = None
        t2_path = None
        flair_path = None

        # -----------------------------------
        # Save files and detect MRI modalities
        # -----------------------------------
        for file in files:

            filename = file.filename.lower()

            # Ignore segmentation ground truth
            if "seg" in filename:
                continue

            save_path = os.path.join(UPLOAD_FOLDER, filename)

            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            file.save(save_path)

            # Detect MRI types
            if "t1n" in filename:
                t1_path = save_path

            elif "t1c" in filename:
                t1ce_path = save_path

            elif "t2w" in filename:
                t2_path = save_path

            elif "t2f" in filename:
                flair_path = save_path

        # -----------------------------------
        # Validate required MRI files
        # -----------------------------------
        if not all([t1_path, t1ce_path, t2_path, flair_path]):
            return "❌ Could not detect all MRI modalities (t1n, t1c, t2w, t2f)."

        # -----------------------------------
        # Run segmentation model
        # -----------------------------------
        overlay_path, volumes = predict(
            model,
            t1_path,
            t1ce_path,
            t2_path,
            flair_path,
            OUTPUT_FOLDER
        )

        overlay_filename = os.path.basename(overlay_path)

        # -----------------------------------
        # Tumor classification
        # -----------------------------------
        crop_path = os.path.join("static", "outputs", "tumor_crop.png")

        if os.path.exists(crop_path):
            tumor_type = classify_tumor(crop_path)
        else:
            tumor_type = "Tumor classification unavailable"

        # -----------------------------------
        # Tumor severity estimation
        # -----------------------------------
        severity = calculate_severity(volumes)

        # -----------------------------------
        # Render result page
        # -----------------------------------
        return render_template(
            "result.html",
            patient_name=patient_name,
            age=age,
            gender=gender,
            tumor_type=tumor_type,
            severity=severity,
            image=overlay_filename
        )

    except Exception as e:
        return f"❌ Error during processing: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)