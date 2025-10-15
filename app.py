import streamlit as st
import json, os, secrets, base64, hashlib
import pandas as pd
from pathlib import Path
import re

def _uniq_clean_list(values):
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return sorted(out)

def _options_from_students_df(df: pd.DataFrame):
    major_opts = _uniq_clean_list(df.get("major_intent", pd.Series(dtype=str)).tolist())
    degree_opts = _uniq_clean_list(df.get("degree_goal", pd.Series(dtype=str)).tolist())
    raw_interests = df.get("interests", pd.Series(dtype=str)).dropna().astype(str).tolist()
    split_items = []
    for s in raw_interests:
        parts = re.split(r"[;,|/]+", s)
        split_items.extend([p.strip() for p in parts if p.strip()])
    interest_opts = _uniq_clean_list(split_items)
    return major_opts, degree_opts, interest_opts

def ensure_students_options_in_state():
    if "students_opts" in st.session_state:
        return
    st.session_state["students_opts"] = {
        "major": [],
        "degree": [],
        "interest": [],
    }

USERS_DB = "users.json"

def hash_pw(pw, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return base64.b64encode(salt).decode(), base64.b64encode(dk).decode()

def verify_pw(pw, salt_b64, hash_b64):
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(hash_b64)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return secrets.compare_digest(dk, expected)

def load_users():
    if os.path.exists(USERS_DB):
        return json.load(open(USERS_DB, "r", encoding="utf-8"))
    return {}

def save_users(data):
    json.dump(data, open(USERS_DB, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

st.set_page_config(page_title="StudyMatch", layout="wide")
st.markdown("<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)
st.title("Explore degrees by leading Australian Universities")

st.session_state.setdefault("auth_open", False)
col1, col2 = st.columns([6, 1])
with col2:
    if st.session_state.get("user"):
        st.write(f"Hello, **{st.session_state['user']}**")
        if st.button("Sign out", key="btn_signout"):
            st.session_state["user"] = None
            st.rerun()
    else:
        if st.button("Login / Register", key="btn_login_toggle"):
            st.session_state["auth_open"] = not st.session_state["auth_open"]

if st.session_state.get("auth_open"):
    with st.sidebar:
        st.markdown("### Sign in or create account")
        tabs = st.tabs(["Sign in", "Register"])
        users = load_users()

        with tabs[0]:
            email = st.text_input("Email", key="auth_signin_email")
            pw = st.text_input("Password", type="password", key="auth_signin_pw")
            if st.button("Sign in now", key="btn_signin_now"):
                u = users.get(email)
                if u and verify_pw(pw, u["salt"], u["hash"]):
                    st.session_state["user"] = u.get("name") or email
                    st.session_state["auth_open"] = False
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
        with tabs[1]:
            name = st.text_input("Your name", key="auth_reg_name")
            email2 = st.text_input("Email", key="auth_reg_email")
            pw1 = st.text_input("Password (min 8 chars)", type="password", key="auth_reg_pw1")
            pw2 = st.text_input("Confirm password", type="password", key="auth_reg_pw2")
            if st.button("Create account", key="btn_create_account"):
                if len(pw1) < 8:
                    st.error("Password must be at least 8 characters.")
                elif pw1 != pw2:
                    st.error("Passwords do not match.")
                elif email2 in users:
                    st.error("This email is already registered.")
                else:
                    salt, hashed = hash_pw(pw1)
                    users[email2] = {"name": name, "salt": salt, "hash": hashed}
                    save_users(users)
                    st.success("Account created! You can now log in.")

MAJOR_FIELDS = [
    "Accounting, Commerce & Economics",
    "Agriculture, Animal & Veterinary Science",
    "Allied Health",
    "Architecture & Design",
    "Arts, Humanities & Social Sciences",
    "Aviation",
    "Business, Marketing & Management",
    "Computer Science & Information Technology",
    "Creative, Media & Communication",
    "Engineering",
    "Health & Biomedical Sciences",
    "Law & Justice",
    "Mathematics & Data Science",
    "Medicine, Dentistry & Oral Health",
    "Music",
    "Nursing & Midwifery",
    "Nutrition & Food Science",
    "Property, Construction & Real Estate",
    "Psychology & Social Work",
    "Science, Environment & Sustainability",
    "Teaching & Education",
    "Tourism, Sport & Events",
]

with st.sidebar:
    st.markdown("### Filters")
    ensure_students_options_in_state()
    opts = st.session_state.get("Students opts", {"major": [], "degree": [], "interest": []})
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Name", "")
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=22)
    with c2:
        country_origin = st.selectbox(
            "Country origin",
            options=[
                "Australia", "China", "India", "Vietnam", "Malaysia", "Singapore", "Indonesia",
                "Philippines", "Thailand", "South Korea", "Japan", "United States", "United Kingdom",
                "Canada", "New Zealand", "Nepal", "Bangladesh", "Pakistan", "Sri Lanka", "Other"
            ],
            index=1,
            key="country_origin_select"
        )
        gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"])

    major_intent = st.selectbox("Major intent", options=(opts["major"] or MAJOR_FIELDS), index=0)
    default_idx = 0
    if "Master" in opts["degree"]:
        default_idx = opts["degree"].index("Master")
    degree_goal = st.selectbox("Degree goal", options=(opts["degree"] or ["Bachelor", "Master", "PhD", "Mixed"]), index=default_idx)

    gpa_default = st.session_state.get("gpa", 3.0)
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.1, value=gpa_default, key="gpa_input")
    st.session_state["gpa"] = gpa

    study_purpose = st.selectbox(
        "Study purpose",
        options=[
            "Career development", "Research pathway", "Migration pathway", "PR pathway",
            "Further study", "Skill improvement", "Other"
        ],
        index=0,
        key="study_purpose_select"
    )

    selected_interests = st.multiselect(
        "Interests (choose up to 4)",
        options=MAJOR_FIELDS,
        default=MAJOR_FIELDS[:4],
        key="interests"
    )
    if len(selected_interests) > 4:
        st.warning("You can select up to 4 interests only.")
        selected_interests = selected_interests[:4]

    languages = st.multiselect(
        "Languages",
        ["English", "Mandarin", "Cantonese", "Hindi", "Spanish", "Arabic", "Vietnamese", "Korean", "Japanese"],
        default=["English", "Mandarin"]
    )
    english_test_type = st.selectbox("English test type", ["IELTS", "PTE", "TOEFL", "None"], index=0)
    english_score_overall = st.number_input("English score overall", min_value=0.0, max_value=9.0, step=0.5, value=6.5)
    has_pr = st.radio("Do you have PR (Permanent Residency)?", options=["Yes", "No"], horizontal=True, key="has_pr")

    if st.button("Save Profile"):
        st.session_state["PROFILE"] = {
            "Name": name,
            "Age": age,
            "Country origin": country_origin,
            "Gender": gender,
            "Interests": selected_interests,
            "Languages": languages,
            "English test type": english_test_type,
            "English score overall": english_score_overall,
            "has_pr": has_pr,
            "degreeGoal": degree_goal,
            "interests_str": ", ".join(selected_interests),
            "major_intent": major_intent,
            "degree_goal": degree_goal,
            "gpa": gpa,
        }
        st.success("Profile saved.")

search_q = st.text_input("Search for courses and subjects", "", placeholder="Search for courses and subjects")
tab1, tab2, tab3 = st.tabs(["Degrees", "Subjects", "Short courses"])
with tab1:
    st.info("Degrees tab (demo placeholder)")
with tab2:
    st.info("Subjects tab (coming soon).")
with tab3:
    st.info("Short courses tab (coming soon).")
