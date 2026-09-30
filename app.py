import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append("src")

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────
VIOLET = "#7C5CFF"
SKY = "#3BB6FF"
MINT = "#22C7A9"
SUN = "#FFB72B"
CORAL = "#FF5C7A"
INK = "#2B2350"
SOFT = "#6F6A8F"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Nunito', sans-serif; color: {INK}; }}
.stApp {{
    background:
        radial-gradient(circle at 0% 0%, #EAE4FF 0, transparent 35%),
        radial-gradient(circle at 100% 10%, #DDF3FF 0, transparent 35%),
        #FBFAFF;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Hero */
.hero {{
    background: linear-gradient(120deg, {VIOLET} 0%, {SKY} 60%, {MINT} 100%);
    border-radius: 28px; padding: 2.25rem 2.5rem; margin-bottom: 1.5rem;
    color: #fff; position: relative; overflow: hidden;
    box-shadow: 0 14px 30px rgba(124, 92, 255, .25);
}}
.hero h1 {{ color: #fff; font-size: 2.3rem; font-weight: 900; margin: 0 0 .4rem 0; padding: 0; }}
.hero p {{ color: #F2F7FF; font-size: 1.08rem; margin: 0; max-width: 58ch; font-weight: 600; }}
.hero .bubble {{
    position: absolute; border-radius: 50%; background: rgba(255,255,255,.16);
}}
.hero .b1 {{ width: 180px; height: 180px; right: -40px; top: -60px; }}
.hero .b2 {{ width: 90px; height: 90px; right: 150px; bottom: -35px; }}
.hero .emoji {{ position: absolute; right: 3rem; top: 50%; transform: translateY(-50%); font-size: 4.2rem; }}

/* Section headers */
.sec {{
    display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .8rem 0;
}}
.sec .ico {{
    width: 40px; height: 40px; border-radius: 12px; display: flex;
    align-items: center; justify-content: center; font-size: 1.3rem;
    background: var(--bg);
}}
.sec .t {{ font-weight: 800; font-size: 1.15rem; margin: 0; line-height: 1.1; }}
.sec .s {{ color: {SOFT}; font-size: .86rem; margin: 0; font-weight: 600; }}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #fff; border: 2px solid #EFEAFF !important;
    border-radius: 22px !important; box-shadow: 0 6px 18px rgba(124, 92, 255, .07);
}}

/* Inputs */
[data-baseweb="select"] > div, .stNumberInput input {{ border-radius: 12px !important; }}
label p {{ font-weight: 700 !important; color: {INK}; }}

/* Buttons */
.stButton > button {{
    border-radius: 14px; font-weight: 800; border: 2px solid #E4DCFF;
    background: #fff; color: {VIOLET}; padding: .55rem 1rem; transition: all .15s ease;
}}
.stButton > button:hover {{ background: #F3EFFF; border-color: {VIOLET}; color: {VIOLET}; }}
.stFormSubmitButton > button {{
    background: linear-gradient(120deg, {VIOLET}, {SKY}); color: #fff; border: none;
    border-radius: 16px; font-weight: 900; font-size: 1.1rem; padding: .9rem 1rem;
    box-shadow: 0 8px 18px rgba(124, 92, 255, .3);
}}
.stFormSubmitButton > button:hover {{ filter: brightness(1.06); color: #fff; }}
:focus-visible {{ outline: 3px solid {SUN}; outline-offset: 2px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: #fff; border-right: 2px solid #EFEAFF; }}
.tile {{
    border-radius: 16px; padding: .7rem 1rem; margin-bottom: .55rem;
    background: var(--bg); display: flex; justify-content: space-between; align-items: center;
}}
.tile span:first-child {{ font-weight: 700; color: {INK}; font-size: .92rem; }}
.tile span:last-child {{ font-weight: 900; color: var(--c); font-size: 1.1rem; }}

/* Verdict */
.verdict {{
    text-align: center; border-radius: 20px; padding: 1.3rem 1rem 1.1rem;
    background: var(--bg); margin-bottom: .5rem;
}}
.verdict .face {{ font-size: 3.6rem; line-height: 1; }}
.verdict .label {{ font-size: 1.5rem; font-weight: 900; color: var(--c); margin: .3rem 0 0 0; }}
.verdict .desc {{ color: {INK}; font-weight: 600; margin: .1rem 0 0 0; }}

/* Chips */
.chip {{
    display: inline-block; padding: .35rem .85rem; margin: 0 .4rem .45rem 0;
    border-radius: 999px; font-size: .86rem; font-weight: 800;
    background: var(--bg); color: var(--c);
}}

/* Tips */
.tip {{
    display: flex; gap: .7rem; align-items: flex-start; padding: .65rem .85rem;
    border-radius: 14px; background: #F7F5FF; margin-bottom: .45rem; font-weight: 600;
}}

.empty {{ text-align: center; padding: 2rem 1rem; color: {SOFT}; font-weight: 600; }}
.empty .big {{ font-size: 3.4rem; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Model loading & preprocessing (unchanged logic)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model_files = list(Path("models").glob("best_model_logistic_regression.pkl"))
        if not model_files:
            st.error("No trained model found. Train the model first, then reload this page.")
            return None, None
        model = joblib.load(model_files[0])
        preprocessor = joblib.load(Path("models/preprocessor.pkl"))
        return model, preprocessor
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        return None, None


def preprocess_input(input_df, preprocessor):
    df = input_df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if df["TotalCharges"].isnull().sum() > 0:
        df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df["TotalCharges"].median()

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 73], labels=[0, 1, 2, 3], include_lowest=True
    )
    df["tenure_group"] = df["tenure_group"].astype(float).astype(int)

    df["avg_monthly_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["avg_monthly_per_tenure"] = df["avg_monthly_per_tenure"].replace([np.inf, -np.inf], 0)

    service_cols = ["PhoneService", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["num_services"] = 0
    for col in service_cols:
        if col in df.columns:
            df["num_services"] += (df[col] == "Yes").astype(int)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = pd.Categorical(df[col]).codes

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_per_tenure"]
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    df[numerical_cols] = preprocessor.scaler.transform(df[numerical_cols])
    return df


# ──────────────────────────────────────────────────────────────
# Profiles
# ──────────────────────────────────────────────────────────────
DEFAULT = {
    "gender": "Male", "senior": "No", "partner": "No", "dependents": "No",
    "tenure": 12, "phone": "Yes", "lines": "No", "internet": "DSL",
    "security": "No", "backup": "No", "protection": "No", "support": "No",
    "tv": "No", "movies": "No", "contract": "Month-to-month",
    "paperless": "Yes", "payment": "Electronic check",
    "monthly": 70.0, "total": 840.0,
}
HIGH_RISK = {**DEFAULT, "gender": "Female", "tenure": 3, "internet": "Fiber optic",
             "tv": "Yes", "movies": "Yes", "monthly": 85.0, "total": 255.0}
LOW_RISK = {**DEFAULT, "partner": "Yes", "dependents": "Yes", "tenure": 48, "lines": "Yes",
            "internet": "Fiber optic", "security": "Yes", "backup": "Yes",
            "protection": "Yes", "support": "Yes", "tv": "Yes", "movies": "Yes",
            "contract": "Two year", "paperless": "No", "payment": "Credit card (automatic)",
            "monthly": 105.0, "total": 5040.0}
PROFILES = {"default": DEFAULT, "high": HIGH_RISK, "low": LOW_RISK}


def load_profile(name):
    for k, v in PROFILES[name].items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def risk_level(p):
    if p >= 50:
        return "High risk", "😟", CORAL, "#FFE8ED", "This customer is likely to leave."
    if p >= 30:
        return "Keep an eye on them", "🤔", "#D98A00", "#FFF3D6", "A quick check-in could help."
    return "Looking good", "😊", "#0E9F87", "#DDF8F2", "This customer is likely to stay."


def chips(items, colour, bg):
    return "".join(f'<span class="chip" style="--c:{colour};--bg:{bg}">{i}</span>' for i in items)


def section(icon, title, sub, bg):
    st.markdown(
        f'<div class="sec"><div class="ico" style="--bg:{bg}">{icon}</div>'
        f'<div><p class="t">{title}</p><p class="s">{sub}</p></div></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 About the model")
        st.markdown(
            f"""
            <div class="tile" style="--bg:#EFEAFF;--c:{VIOLET}"><span>Type</span><span>Logistic Regression</span></div>
            <div class="tile" style="--bg:#DFF3FF;--c:#1A8FD6"><span>ROC-AUC</span><span>0.8458</span></div>
            <div class="tile" style="--bg:#DDF8F2;--c:#0E9F87"><span>Accuracy</span><span>80.41%</span></div>
            <div class="tile" style="--bg:#FFF3D6;--c:#D98A00"><span>F1-score</span><span>0.5929</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ✨ Try an example")
        st.caption("Fills the form for you.")
        st.button("🆕 New, month-to-month", on_click=load_profile, args=("high",),
                  use_container_width=True)
        st.button("💎 Loyal, two-year", on_click=load_profile, args=("low",),
                  use_container_width=True)
        st.button("🧹 Reset form", on_click=load_profile, args=("default",),
                  use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Form
# ──────────────────────────────────────────────────────────────
def render_form():
    yn = ["No", "Yes"]
    yn_net = ["No", "Yes", "No internet service"]

    with st.form("customer_form", border=False):
        with st.container(border=True):
            section("👤", "About the customer", "Who are we talking about?", "#EFEAFF")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox("Gender", ["Male", "Female"], key="gender")
                st.selectbox("Senior citizen", yn, key="senior")
            with c2:
                st.selectbox("Has partner", yn, key="partner")
                st.selectbox("Has dependents", yn, key="dependents")
            with c3:
                st.slider("Months as customer", 0, 72, key="tenure")

        with st.container(border=True):
            section("💳", "Account & billing", "How do they pay?", "#DFF3FF")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
                st.selectbox("Paperless billing", yn, key="paperless")
            with c2:
                st.selectbox("Payment method",
                             ["Electronic check", "Mailed check",
                              "Bank transfer (automatic)", "Credit card (automatic)"],
                             key="payment")
            with c3:
                st.number_input("Monthly charges ($)", 0.0, 200.0, step=5.0, key="monthly")
                st.number_input("Total charges ($)", 0.0, 10000.0, step=50.0, key="total")

        with st.container(border=True):
            section("📡", "Services", "What do they use?", "#DDF8F2")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox("Phone service", yn, key="phone")
                st.selectbox("Multiple lines", ["No", "Yes", "No phone service"], key="lines")
                st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="internet")
            with c2:
                st.selectbox("Online security", yn_net, key="security")
                st.selectbox("Online backup", yn_net, key="backup")
                st.selectbox("Device protection", yn_net, key="protection")
            with c3:
                st.selectbox("Tech support", yn_net, key="support")
                st.selectbox("Streaming TV", yn_net, key="tv")
                st.selectbox("Streaming movies", yn_net, key="movies")

        return st.form_submit_button("🔮  Predict churn risk", use_container_width=True)


def build_input():
    s = st.session_state
    return pd.DataFrame({
        "customerID": ["PRED-001"],
        "gender": [s.gender],
        "SeniorCitizen": [1 if s.senior == "Yes" else 0],
        "Partner": [s.partner],
        "Dependents": [s.dependents],
        "tenure": [s.tenure],
        "PhoneService": [s.phone],
        "MultipleLines": [s.lines],
        "InternetService": [s.internet],
        "OnlineSecurity": [s.security],
        "OnlineBackup": [s.backup],
        "DeviceProtection": [s.protection],
        "TechSupport": [s.support],
        "StreamingTV": [s.tv],
        "StreamingMovies": [s.movies],
        "Contract": [s.contract],
        "PaperlessBilling": [s.paperless],
        "PaymentMethod": [s.payment],
        "MonthlyCharges": [s.monthly],
        "TotalCharges": [s.total],
    })


def factors():
    s = st.session_state
    risks, good = [], []
    if s.contract == "Month-to-month":
        risks.append("Month-to-month contract")
    else:
        good.append(f"{s.contract} contract")
    if s.tenure < 12:
        risks.append("New customer (< 1 year)")
    elif s.tenure >= 36:
        good.append("Long-time customer")
    if s.monthly > 80:
        risks.append("High monthly charges")
    if s.security == "No":
        risks.append("No online security")
    elif s.security == "Yes":
        good.append("Has online security")
    if s.support == "No":
        risks.append("No tech support")
    elif s.support == "Yes":
        good.append("Has tech support")
    if s.payment == "Electronic check":
        risks.append("Pays by electronic check")
    if s.partner == "No":
        risks.append("No partner")
    elif s.partner == "Yes":
        good.append("Has a partner")
    return risks, good


# ──────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────
def gauge(churn_p, colour):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_p,
        number={"suffix": "%", "valueformat": ".1f",
                "font": {"size": 42, "color": INK, "family": "Nunito"}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 30, 50, 100],
                     "tickfont": {"size": 11, "color": SOFT}, "tickwidth": 0},
            "bar": {"color": colour, "thickness": 0.3},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#CFF3EA"},
                {"range": [30, 50], "color": "#FFE7B0"},
                {"range": [50, 100], "color": "#FFC9D3"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=15, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Nunito"})
    return fig


def render_results():
    with st.container(border=True):
        section("🎯", "Your result", "Updates when you predict", "#FFF3D6")
        res = st.session_state.get("result")

        if res is None:
            st.markdown(
                """
                <div class="empty">
                    <div class="big">🔮</div>
                    <p><b>Ready when you are!</b></p>
                    <p>Fill in the customer details and press
                    <b>Predict churn risk</b>, or load an example from the sidebar.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if "error" in res:
            st.error(f"Prediction failed: {res['error']}")
            with st.expander("Technical details"):
                st.code(res["trace"])
            return

        churn_p = res["churn_p"]
        label, face, colour, bg, desc = risk_level(churn_p)

        st.markdown(
            f'<div class="verdict" style="--c:{colour};--bg:{bg}">'
            f'<div class="face">{face}</div><p class="label">{label}</p>'
            f'<p class="desc">{desc}</p></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(churn_p, colour), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(f"Chance of churning: {churn_p:.1f}% · Chance of staying: {100 - churn_p:.1f}%")

        risks, good = factors()
        if risks:
            st.markdown("**⚠️ What raises the risk**")
            st.markdown(chips(risks, "#D63A5A", "#FFE8ED"), unsafe_allow_html=True)
        if good:
            st.markdown("**💚 What keeps them**")
            st.markdown(chips(good, "#0E9F87", "#DDF8F2"), unsafe_allow_html=True)

        st.markdown("**💡 What to do next**")
        if churn_p >= 50:
            tips = [("🎁", "Offer a discount or loyalty promotion"),
                    ("📝", "Propose a longer contract with perks"),
                    ("📞", "Have support reach out personally"),
                    ("📋", "Send a short satisfaction survey")]
        elif churn_p >= 30:
            tips = [("💬", "Check in on how they're doing"),
                    ("🛡️", "Suggest security or tech support add-ons"),
                    ("📆", "Highlight the benefits of a longer contract")]
        else:
            tips = [("✅", "Keep service quality steady"),
                    ("⭐", "Invite them to the loyalty program"),
                    ("🙂", "Run regular satisfaction checks")]
        st.markdown("".join(f'<div class="tip"><span>{i}</span><span>{t}</span></div>'
                            for i, t in tips), unsafe_allow_html=True)

        with st.expander("Technical details"):
            st.write("Processed shape:", res["shape"])
            st.dataframe(res["processed"], use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div class="hero">
            <div class="bubble b1"></div><div class="bubble b2"></div>
            <div class="emoji">🔮</div>
            <h1>Will this customer stay?</h1>
            <p>Tell us about a customer and get a friendly, instant read on their
            churn risk, plus ideas for keeping them happy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, preprocessor = load_models()
    if model is None or preprocessor is None:
        st.stop()

    render_sidebar()

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        submitted = render_form()

    if submitted:
        try:
            processed = preprocess_input(build_input(), preprocessor)
            proba = model.predict_proba(processed)[0]
            st.session_state.result = {
                "churn_p": float(proba[1] * 100),
                "shape": processed.shape,
                "processed": processed,
            }
        except Exception as e:
            import traceback
            st.session_state.result = {"error": str(e), "trace": traceback.format_exc()}

    with col_result:
        render_results()


if __name__ == "__main__":
    main()