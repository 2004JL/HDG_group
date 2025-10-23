import streamlit as st
import io
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retrieval import Retrieval

# ----------------- Login / Register Button -----------------
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
        st.write(f"👋 Hello, **{st.session_state['user']}**")
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

        import json, os, secrets, base64, hashlib
        USERS_DB = "users.json"
        def load_users():
            if os.path.exists(USERS_DB):
                return json.load(open(USERS_DB, "r", encoding="utf-8"))
            return {}
        def save_users(data):
            json.dump(data, open(USERS_DB, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
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

# ================== PAGE SETUP ==================
st.set_page_config(page_title="OUA-Style Degrees", layout="wide")
import streamlit as st

st.set_page_config(page_title="StudyMatch", layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Explore degrees by leading Australian Universities")

# ================== OUA LOOK & FEEL ==================
# 1) CSS


st.markdown("""
<style>
.profile2-card{
  background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;
  padding:18px; box-shadow:0 8px 24px rgba(15,23,42,.06); margin-bottom:16px;
}
.profile2-title{font-weight:800;font-size:1.05rem;color:#0f172a;margin:0 0 2px}
.profile2-sub{font-size:.85rem;color:#64748b;margin:0 0 14px}

#profile2-panel input, #profile2-panel select, #profile2-panel textarea{
  border:1px solid #e5e7eb !important; border-radius:10px !important; background:#fff !important;
  box-shadow:0 2px 8px rgba(2,6,23,.03);
}
#profile2-panel input:focus, #profile2-panel select:focus, #profile2-panel textarea:focus{
  outline:none !important; border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,.15) !important;
}
#profile2-panel .stTextInput, #profile2-panel .stNumberInput, #profile2-panel .stSelectbox, #profile2-panel .stTextArea{
  margin-bottom:10px !important;
}

#profile2-panel [role="radiogroup"] > label{
  border:1px solid #d1d5db; border-radius:999px; padding:6px 14px; margin-right:10px;
  background:#fff; cursor:pointer; transition:all .2s ease;
}
#profile2-panel [role="radiogroup"] > label:hover{ box-shadow:0 2px 8px rgba(0,0,0,.08); background:#f9fafb; }
input[type="radio"]{ accent-color:#2563eb; }
</style>
""", unsafe_allow_html=True)

def oua_colgrid(items, per_row=2):
    cols = st.columns(per_row)
    for i, item in enumerate(items):
        yield cols[i % per_row], item

def heart_toggle(key, default=False):
    st.session_state.setdefault(key, default)
    clicked = st.button("♡" if not st.session_state[key] else "♥", key=f"heart_{key}")
    if clicked: st.session_state[key] = not st.session_state[key]

def render_oua_degree_card(p: dict, idx: int):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ribbon"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="logo-chip">{p.get("institution","")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="heart">', unsafe_allow_html=True)
    heart_toggle(f"deg_{idx}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="body">', unsafe_allow_html=True)
    st.markdown(f'<div class="title"><a href="{p.get("url","#")}" target="_blank">{p.get("title","")}</a></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{p.get("level","Undergraduate")} | {p.get("code","")}</div>', unsafe_allow_html=True)

    if p.get("blurb"):
        st.markdown(f'<div class="desc">{p["blurb"]}</div>', unsafe_allow_html=True)
        with st.expander("View more"):
            st.write(p.get("long", p["blurb"]))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    lines = [
        ("100% online" if p.get("online", True) else "Online & on-campus"),
        (p.get("duration","3 years full time or part time equivalent")),
        (p.get("entry","No ATAR required. Start with a subject.")),
    ]
    st.markdown('<div class="meta">', unsafe_allow_html=True)
    for text in lines:
        st.markdown(f'<span>{text}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    cta = st.columns([1,1,1])
    with cta[0]:
        st.markdown(f'<a class="btn" href="{p.get("url","#")}" target="_blank">Explore details</a>', unsafe_allow_html=True)
    with cta[1]:
        if p.get("majors"):
            st.markdown(f'<span class="badge">Available majors</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  
    st.markdown('</div>', unsafe_allow_html=True) 

if "DEMO_DEGREES" not in st.session_state:
    st.session_state.DEMO_DEGREES = [
        {
            "title":"Undergraduate Certificate in Health Sciences",
            "institution":"Curtin University",
            "level":"Undergraduate",
            "code":"CUR-CHS-CTF",
            "online":True,
            "duration":"6 months full time or part time equivalent",
            "entry":"No ATAR required. Start with a subject.",
            "blurb":"Take your first step towards a career in health—including nursing",
            "url":"https://example.org/chs",
        },
        {
            "title":"Undergraduate Certificate in Psychology",
            "institution":"Curtin University",
            "level":"Undergraduate",
            "code":"CUR-HPS-CTF",
            "online":True,
            "duration":"6 months full time or part time equivalent",
            "entry":"No ATAR required. Start with a subject.",
            "blurb":"Launch your future in psychology",
            "url":"https://example.org/psy",
        },
        {
            "title":"Bachelor of Business",
            "institution":"Griffith University",
            "level":"Undergraduate",
            "code":"GRF-BUS-DEG",
            "online":True,
            "duration":"3 years full time or part time equivalent",
            "entry":"No ATAR required. Start with a subject.",
            "blurb":"Get the professional skills employers are looking for",
            "url":"https://example.org/bus",
            "majors":["Marketing","Management","International Business"],
        },
        {
            "title":"Bachelor of Psychology and Counselling",
            "institution":"Edith Cowan University",
            "level":"Undergraduate",
            "code":"ECU-PSC-DEG",
            "online":True,
            "duration":"3 years full time or part time equivalent",
            "entry":"No ATAR required. Start with a subject.",
            "blurb":"Analyse, listen, and respond to people with empathy",
            "url":"https://example.org/pc",
            "majors":["Psychology","Counselling"],
        },
    ]

# student input
with st.sidebar:
    st.markdown('<div class="profile2-card"><div class="profile2-title">Student profile & export</div><div class="profile2-sub">Fill in your student profile for exporting JSON</div></div>', unsafe_allow_html=True)
    st.markdown('<div id="profile2-panel">', unsafe_allow_html=True)

    is_ug = st.session_state.get("Study level-Undergraduate", True) if "Study level-Undergraduate" in st.session_state else False
    is_pg = st.session_state.get("Study level-Postgraduate", False) if "Study level-Postgraduate" in st.session_state else False
    selected_areas = st.session_state.get("Interest area-Area", ["Business"]) if "Interest area-Area" in st.session_state else ["Business"]

    if is_pg:
        auto_degree = "master"
    elif is_ug:
        auto_degree = "bachelor"
    else:
        auto_degree = "bachelor"

    auto_interests = ";".join([x.strip().lower() for x in selected_areas if str(x).strip()])
    auto_major_intent = selected_areas[0].strip().lower() if selected_areas else "general"

    c1, c2 = st.columns(2, vertical_alignment="center")
    with c1:
        sid = st.text_input("Student ID", value="S000123", key="student_id_input")
    with c2:
        major = st.text_input("Major intent", value=auto_major_intent, key="major_intent_input")

    c3, c4 = st.columns(2, vertical_alignment="center")
    with c3:
        degree = st.selectbox("Degree goal", ["bachelor", "master", "phd"],
                              index=["bachelor","master","phd"].index(auto_degree), key="degree_goal_input")
    with c4:
        eng_type = st.selectbox("English test type", ["IELTS", "TOEFL", "PTE"], index=0, key="english_test_type_input")

    c5, c6 = st.columns(2, vertical_alignment="center")
    with c5:
        eng_score = st.number_input("English overall", min_value=0.0, max_value=120.0, step=0.5, value=7.0,
                                    help="IELTS 0–9, TOEFL 0–120, PTE 0–90", key="english_overall_input")
    with c6:
        gpa = st.number_input("GPA (0–4 scale)", min_value=0.0, max_value=4.0, step=0.1, value=3.4, key="gpa_input_sidebar")

    interests = st.text_area("Interests (; separated)", value=auto_interests or "ai;data science;ml", key="interests_input")

    st.divider()

    st.markdown("**Do you want to get PR from studying**")
    st.radio(" ", options=["Yes", "No"], horizontal=True, key="has_pr")

    st.markdown('</div>', unsafe_allow_html=True)

    payload = {
        "student_id": sid.strip(),
        "major_intent": major.strip(),
        "degree_goal": degree.strip(),
        "english_test_type": eng_type.strip(),
        "english_score_overall": float(eng_score),
        "gpa_std_4": float(gpa),
        "interests": interests.strip(),
        "has_pr": st.session_state.get("has_pr", "No"),
    }

    if st.button("Save", use_container_width=True):
        ROOT = Path(__file__).resolve().parents[0]
        dest_dir = ROOT / "retrieval"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / "student.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        st.success(f"Saved → {out_path}")
        st.session_state["last_student_json_path"] = str(out_path)




json_path = st.session_state.get(
    "last_student_json_path",
    str(ROOT / "retrieval" / "student.json")
)

#programs recommendation
run = st.button("Find eligible programs")
if run:
    try:
        r = Retrieval()
        df = r.run(json_path)
        import math
        student_interest_str = st.session_state.get("interests_input") or ";".join(
            st.session_state.get("interests", [])
        )
        student_interests = {
            s.strip().lower() for s in (student_interest_str or "").replace(",", ";").split(";") if s.strip()
        }
        if not student_interests:
            student_interests = {"business"}  

        def interest_overlap(tags: str) -> float:
            if not isinstance(tags, str) or not tags.strip():
                return 0.0
            tags_set = {t.strip().lower() for t in tags.replace(",", ";").split(";") if t.strip()}
            if not tags_set:
                return 0.0
            return len(student_interests & tags_set) / max(1, len(student_interests))

        df["interest_score"] = df.get("field_tags", "").apply(interest_overlap)

        if "overall_ranking" in df.columns:
            df["overall_ranking"] = pd.to_numeric(df["overall_ranking"], errors="coerce")
            r_min = df["overall_ranking"].min(skipna=True)
            r_max = df["overall_ranking"].max(skipna=True)
            if pd.notna(r_min) and pd.notna(r_max) and r_max > r_min:
                df["rank_score"] = (r_max - df["overall_ranking"]) / (r_max - r_min)
            else:
                df["rank_score"] = 0.0
        else:
            df["rank_score"] = 0.0

        W_INTEREST = 0.6
        W_RANK = 0.4
        df["relevance_score"] = W_INTEREST * df["interest_score"] + W_RANK * df["rank_score"]

        df = df.sort_values(["relevance_score", "interest_score", "rank_score"], ascending=False).reset_index(drop=True)

        st.markdown("### 🎓 Recommended Programs")

        for i, row in df.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color:#ffffff;border:1px solid #e5e7eb;border-radius:14px;
                    padding:14px 16px;margin-bottom:12px;box-shadow:0 4px 12px rgba(15,23,42,0.06);">
                <p style="font-size:0.9rem;margin:0;">
                    <b>Program ID:</b> {row.get('program_id','')}<br>
                    <b>Institution ID:</b> {row.get('institution_id','')}<br>
                    <b>Field Tags:</b> {row.get('field_tags','')}<br>
                    <b>Overall Ranking:</b> {row.get('overall_ranking','')}<br>
                    <b>Interests:</b> {row.get('interests','')}<br>
                    <b>Match Score:</b> {row.get('relevance_score',0):.2f}
                </p>
                </div>
                """, unsafe_allow_html=True)


        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full CSV",
            data=csv,
            file_name="eligible_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error: {e}")