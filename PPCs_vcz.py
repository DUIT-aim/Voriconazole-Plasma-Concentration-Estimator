import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 0. Page configuration
# ===============================
st.set_page_config(
    page_title="Voriconazole Concentration Estimator",
    layout="wide"
)

st.title("Voriconazole (VCZ) Plasma Concentration Estimator")
st.caption(
    "This tool estimates voriconazole exposure by combining a clearance prediction model "
    "with a pharmacokinetic back-calculation approach."
)

# ===============================
# 1. Load models
# ===============================
@st.cache_resource
def load_assets():
    model_cl = joblib.load("xgb_cl.pkl")          # CL/F prediction model
    calibrator = joblib.load("calibrator_DV.pkl") # concentration calibrator
    return model_cl, calibrator

model_cl, calibrator = load_assets()

# ===============================
# 2. Define CL/F model features
#    (must exactly match training order)
# ===============================
features_cl = ['CRP', 'ALB', 'GenotypingValue', 'Age', 'Sex', 'TBIL', 'Weight']

# ===============================
# 3. Sidebar: patient information
# ===============================
st.sidebar.header("Patient information")

# ---- Dose and timing (not used for CL/F prediction) ----
daydose = st.sidebar.number_input(
    "Daily voriconazole dose (mg/day)",
    min_value=10.0, max_value=800.0,
    value=400.0, step=10.0
)

time_val = st.sidebar.number_input(
    "Time since initiation of therapy (days)",
    min_value=0.0, max_value=60.0,
    value=7.0, step=0.5
)

# ---- CL/F covariates ----
age = st.sidebar.number_input(
    "Age (years)",
    min_value=0.0, max_value=120.0,
    value=50.0, step=1.0
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=1.0, max_value=200.0,
    value=60.0, step=1.0
)

alb = st.sidebar.number_input(
    "Albumin (ALB, g/L)",
    min_value=5.0, max_value=60.0,
    value=32.0, step=0.5
)

crp = st.sidebar.number_input(
    "C-reactive protein (CRP, mg/L)",
    min_value=0.0, max_value=300.0,
    value=30.0, step=1.0
)

tbil = st.sidebar.number_input(
    "Total bilirubin (TBIL, µmol/L)",
    min_value=0.0, max_value=500.0,
    value=12.0, step=1.0
)

sex_input = st.sidebar.selectbox(
    "Sex",
    options=["Male", "Female"]
)
sex = 1 if sex_input == "Male" else 2  # must match training encoding

# ---- CYP2C19 metabolizer status ----
geno_label = st.sidebar.selectbox(
    "CYP2C19 metabolizer status",
    options=[
        "NM (Normal metabolizer)",
        "IM (Intermediate metabolizer)",
        "PM (Poor metabolizer)"
    ],
    index=0
)

geno_map = {
    "NM (Normal metabolizer)": 1,
    "IM (Intermediate metabolizer)": 2,
    "PM (Poor metabolizer)": 3
}
GenotypingValue = geno_map[geno_label]

st.sidebar.caption(
    "Metabolizer status is encoded according to the scheme used during model development. "
    "Genotype-based status may not fully reflect actual metabolic capacity under inflammatory conditions."
)

# ===============================
# 4. Pharmacokinetic calculation
# ===============================
def calculate_theoretical_conc(pred_cl, dose_mg_per_day):
    pred_cl_safe = max(float(pred_cl), 0.1)
    return float(dose_mg_per_day) / (24.0 * pred_cl_safe)

# Assemble input strictly following feature order
input_row = pd.DataFrame([{
    "GenotypingValue": GenotypingValue,
    "Weight": weight,
    "Sex": sex,
    "CRP": crp,
    "ALB": alb,
    "Age": age,
    "TBIL": tbil
}], columns=features_cl)

# ===============================
# 5. Estimation
# ===============================
if st.button("Estimate voriconazole concentration"):

    # Step 1: estimate clearance
    pred_cl = model_cl.predict(input_row)[0]

    # Step 2: PK back-calculation
    theory_conc = calculate_theoretical_conc(pred_cl, daydose)

    # Step 3: calibration
    pred_conc = calibrator.predict(
        np.array(theory_conc).reshape(-1, 1)
    )[0]
    pred_conc = max(float(pred_conc), 0.1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estimated clearance")
        st.metric("Estimated CL/F (L/h)", f"{pred_cl:.3f}")
        st.caption(
            "Clearance is estimated from genotype and clinical covariates. "
            "Dose is not included in this step."
        )

    with col2:
        st.subheader("Estimated plasma concentration")
        st.metric(
            "Voriconazole concentration (mg/L)",
            f"{pred_conc:.3f}",
            help=(
                "An empirical variability of approximately ±30% may be expected "
                "under routine clinical conditions."
            )
        )

    st.markdown("---")
    st.subheader("Model interpretation")
    st.write(f"Daily dose: **{daydose:.1f} mg/day**")
    st.write(
        f"Theoretical concentration (dose / [24 × CL]): **{theory_conc:.3f} mg/L**"
    )
    st.write(f"Time since initiation of therapy: **{time_val:.1f} days**")

    if time_val < 7:
        st.warning(
            "This estimate is based on a sample collected before steady-state "
            "conditions are typically achieved. Interpretation should be made with caution."
        )
    else:
        st.success(
            "The sampling time is consistent with near steady-state conditions "
            "used during model calibration."
        )

# ===============================
# 6. Notes
# ===============================
st.markdown("### Notes")
st.markdown("""
- This tool is intended to support research and clinical interpretation of voriconazole exposure.
- Results should be interpreted in conjunction with therapeutic drug monitoring and clinical judgment.
- The model has been developed and evaluated under specific assumptions, including near steady-state conditions.
""")