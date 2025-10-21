import streamlit as st
import json, os, secrets, base64, hashlib

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

st.set_page_config(page_title="StudyMatch", layout="wide")
st.markdown(
    "<style>#MainMenu {visibility:hidden;} footer {visibility:hidden;}</style>",
    unsafe_allow_html=True,
)
st.title("Explore degrees by leading Australian Universities")

MAJOR_FIELDS = [
    "Accounting", "Business", "Computer Science", "Data Science", "Information Technology",
    "Cybersecurity", "Artificial Intelligence", "Engineering", "Mechanical Engineering",
    "Electrical Engineering", "Civil Engineering", "Software Engineering", "Biomedical Science",
    "Health Sciences", "Nursing", "Psychology", "Education", "Law", "Architecture",
    "Communication", "Creative Arts", "Economics", "Finance", "Marketing", "International Relations"
]

DEGREE_GOALS = ["Bachelor", "Master", "PhD", "Diploma"]

with st.sidebar:
    st.markdown("### Profile")
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

    major_intent = st.selectbox("Major intent", options=MAJOR_FIELDS, index=0)
    degree_goal = st.selectbox("Degree goal", options=DEGREE_GOALS, index=0)
    gpa_default = st.session_state.get("gpa", 3.0)
    gpa = st.number_input("GPA (max 4.0)", min_value=0.0, max_value=4.0, step=0.1, value=gpa_default, key="gpa_input")
    st.session_state["gpa"] = gpa

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

    if english_test_type == "IELTS":
        english_score_overall = st.number_input("English score overall", min_value=0.0, max_value=9.0, step=0.5, value=6.5)
    elif english_test_type == "TOEFL":
        english_score_overall = st.number_input("English score overall", min_value=0.0, max_value=120.0, step=1.0, value=90.0)
    elif english_test_type == "PTE":
        english_score_overall = st.number_input("English score overall", min_value=0.0, max_value=90.0, step=1.0, value=65.0)
    else:
        english_score_overall = 0.0

    has_pr = st.radio("Do you want to get PR from studying", options=["Yes", "No"], horizontal=True, key="has_pr")

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
        st.success("Profile saved!")

search_q = st.text_input("Search for courses and subjects", "", placeholder="Search for courses and subjects")
tab1, tab2, tab3 = st.tabs(["Degrees", "Subjects", "Tutors"])

with tab1:
    st.info("Degrees tab (coming soon).")

with tab2:
    st.info("Subjects tab (coming soon).")

with tab3:
    st.info("Tutors tab (coming soon).")
