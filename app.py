import streamlit as st
import io
import sys
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retrieval_program import Retrieval
from retrieval.retrieval_mentor import RetrievalMentor
from retrieval.retrieval_core import RetrievalCore

# Login
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

st.title("Educational pathway")

# ================== OUA LOOK & FEEL ==================
# 1) CSS
st.markdown("""
<style>
:root{
  --br:#e5e7eb;--muted:#6b7280;--blue:#1f45ff;--ink:#111827;--bg:#fff;
}
/* container width a little wider feeling */
.block-container{padding-top:1rem;}

.oua-wrap{padding-top:4px}
.oua-hbar{display:flex;gap:10px;justify-content:space-between;align-items:center;margin:4px 0 14px}
.oua-title{font-weight:700;color:var(--ink)}
.oua-saved{color:var(--muted);font-size:.92rem}
.oua-compare{border:1px solid var(--br);background:#f8fafc;border-radius:999px;padding:8px 14px}

.card{
  border:1px solid var(--br); border-radius:14px; background:var(--bg); padding:0 0 14px; position:relative;
  box-shadow:0 2px 10px rgba(2,6,23,.04);
}
.ribbon{height:52px; border-radius:14px 14px 0 0;
  background: linear-gradient(135deg, #2339a2 50%, #7b4bd4 50%);
}
.logo-chip{position:absolute; top:32px; left:16px; background:#fff; border:1px solid var(--br);
  border-radius:10px; padding:4px 8px; font-weight:600; box-shadow:0 2px 6px rgba(2,6,23,.08);}
.heart{position:absolute; right:14px; top:38px; border:1px solid var(--br); width:38px; height:38px;
  border-radius:999px; display:flex; align-items:center; justify-content:center; background:#fff}
.body{padding:12px 16px 0}
.title a{font-weight:800; color:#183c8c; text-decoration:underline; font-size:1.02rem}
.subtitle{color:var(--muted); font-size:.92rem; margin:2px 0 8px}
.desc{margin:6px 0 10px; color:var(--ink)}
.hr{border-top:1px solid var(--br); margin:8px 0 10px}
.meta{display:flex; flex-direction:column; gap:8px; color:var(--muted);}
.meta span{display:flex; gap:8px; align-items:center}
.btn{display:inline-block; background:#1f45ff; color:#fff; border-radius:999px; padding:10px 18px; font-weight:700; text-decoration:none}
.badge{display:inline-flex; gap:6px; align-items:center; background:#f1f5f9; border:1px solid var(--br); padding:3px 8px; border-radius:999px; font-size:.82rem}
.viewmore{color:#183c8c; cursor:pointer; user-select:none}
.pag{display:flex; gap:8px; justify-content:center; margin-top:14px}
.pag button{border:1px solid var(--br); background:#fff; padding:6px 10px; border-radius:8px}
</style>
""", unsafe_allow_html=True)

# 2) 一些小工具
def oua_colgrid(items, per_row=2):
    cols = st.columns(per_row)
    for i, item in enumerate(items):
        yield cols[i % per_row], item

def heart_toggle(key, default=False):
    st.session_state.setdefault(key, default)
    clicked = st.button("♡" if not st.session_state[key] else "♥", key=f"heart_{key}")
    if clicked: st.session_state[key] = not st.session_state[key]

# 3) OUA 卡片（Degree）
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

# 4) Mock 数据
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

# 5) 左侧 Filters（折叠样式）
with st.sidebar:
    st.markdown("### Filters")
    with st.expander("Study level", expanded=True):
        st.checkbox("Undergraduate", True)
        st.checkbox("Postgraduate", False)
    with st.expander("Interest area", expanded=False):
        st.multiselect("Area", ["Business","IT & computer science","Education & teaching","Psychology"], ["Business"]) 
    with st.expander("University", expanded=False):
        st.multiselect("University", ["Curtin","Griffith","ECU","La Trobe"]) 
    with st.expander("Qualification", expanded=False):
        st.multiselect("Type", ["Degree","Undergraduate certificate","Diploma"], ["Degree"]) 
    with st.expander("Study method", expanded=False):
        st.checkbox("100% online", True)
        st.checkbox("On-campus", False)
    with st.expander("Entry options", expanded=False):
        st.checkbox("No ATAR required", True)
    with st.expander("Other", expanded=False):
        st.slider("Duration (years)", 0, 6, (0,3))
        st.markdown("#### PR (Permanent Residency)")
        has_pr = st.radio(
            "Do you have PR?",
            options=["Yes", "No"],
            horizontal=True,
            key="has_pr"
        )

INTEREST_OPTIONS = ["accounting", "architecture", "artificial intelligence", "banking", "business", "computer science", "cybersecurity",
    "data science", "design", "ecology", "education", "engineering", "environmental science", "film", "finance", "information technology",
    "international law", "law", "marketing", "nursing", "psychology", "public health", "renewable energy", "sustainable design"]

# Take student input
with st.sidebar:
    st.markdown("Student profile & export")
    is_ug = st.session_state.get("Study level-Undergraduate", True) if "Study level-Undergraduate" in st.session_state else False
    is_pg = st.session_state.get("Study level-Postgraduate", False) if "Study level-Postgraduate" in st.session_state else False
    selected_areas = st.session_state.get("Interest area-Area", ["Business"]) if "Interest area-Area" in st.session_state else ["Business"]

    if is_pg:
        auto_degree = "master"
    elif is_ug:
        auto_degree = "bachelor"
    else:
        auto_degree = "bachelor"

    auto_interests = [x.strip().lower() for x in selected_areas if str(x).strip()]

    auto_major_intent = selected_areas[0].strip().lower() if selected_areas else "general"

    sid = st.text_input("Student ID", value="S000123", key="student_id_input")
    major = st.text_input("Major intent", value=auto_major_intent, key="major_intent_input")
    degree = st.selectbox("Degree goal", ["bachelor", "master", "phd"], index=["bachelor","master","phd"].index(auto_degree))
    eng_type = st.selectbox("English test type", ["IELTS", "TOEFL", "PTE"], index=0)
    eng_score = st.number_input("English overall", min_value=0.0, max_value=120.0, step=0.5, value=7.0, help="IELTS 0-9, TOEFL 0-120, PTE 0-90")
    gpa = st.number_input("GPA (0-4 scale)", min_value=0.0, max_value=4.0, step=0.1, value=3.4)
    migration = st.radio("migration option", options=["Yes", "No"], horizontal=False)
    migration = (migration == "Yes")
    interests = st.multiselect("Interests (choose one or more)", options=INTEREST_OPTIONS, default=[], help="You can pick multiple interests")
    interests = ";".join([t.strip().lower() for t in interests])
    
    payload = {
        "student_id": sid.strip(),
        "major_intent": major.strip(),
        "degree_goal": degree.strip(),
        "english_test_type": eng_type.strip(),
        "english_score_overall": float(eng_score),
        "gpa_std_4": float(gpa),
        "migration_interest": migration,
        "interests": interests.strip(),
    }

    # Output json
    col, = st.columns(1)
    with col:
        if st.button("json", use_container_width=True):
            ROOT = Path(__file__).resolve().parents[0]
            dest_dir = ROOT / "retrieval"
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_path = dest_dir / "student.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

json_path = st.session_state.get(
    "last_student_json_path",
    str(ROOT / "retrieval" / "student.json")
)

def program_labelmatch(df):
    bundle = joblib.load(ROOT / "models" / "xgb_program_labelmatch_regressor.pkl")
    if isinstance(bundle, dict) and "model" in bundle:
        model   = bundle["model"]
        mlb_int = bundle["mlb_int"]
        mlb_tag = bundle["mlb_tag"]

    df["interests"]  = df["interests"].fillna("")
    df["field_tags"] = df["field_tags"].fillna("")

    def to_list(s):
        return [x.strip().lower() for x in str(s).split(";") if x.strip()]

    X_int = mlb_int.transform(df["interests"].map(to_list))
    X_tag = mlb_tag.transform(df["field_tags"].map(to_list))
    X = hstack([X_int, X_tag], format="csr")

    dX = xgb.DMatrix(X)
    pred = model.predict(dX)

    out = df.copy()
    out["pred_label_match"] = np.round(pred, 4)
    return out

col1, col2 = st.columns(2)
if "last_output" not in st.session_state:
    st.session_state["last_output"] = None
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None

# Programs recommendation
with col1:
    if st.button("Find eligible programs", use_container_width=True):
        if migration != True:
            r = Retrieval()
            df = r.run(json_path)

            out = program_labelmatch(df)
            out = out.sort_values("pred_label_match", ascending=False).reset_index(drop=True)
            out = out.head(3).reset_index(drop=True)
            out.to_csv( ROOT / "retrieval" / "student_program.csv", index=False)
        else:
            f = RetrievalCore()
            df = f.find(json_path)

            bundle = joblib.load(ROOT / "models" / "xgb_core_program_labelmatch_regressor.pkl")
            if isinstance(bundle, dict) and "model" in bundle:
                model   = bundle["model"]
                mlb_int = bundle["mlb_int"]
                mlb_tag = bundle["mlb_tag"]

            df["interests"] = df["interests"].fillna("")
            df["core_program"] = df["core_program"].fillna("")

            def to_list(s):
                return [x.strip().lower() for x in str(s).split(";") if x.strip()]
            
            X_int = mlb_int.transform(df["interests"].map(to_list))
            X_tag = mlb_tag.transform(df["core_program"].map(to_list))
            X = hstack([X_int, X_tag], format="csr")

            dX = xgb.DMatrix(X)
            pred = model.predict(dX)

            p = df.copy()
            p["pred_label_match"] = np.round(pred, 4)
            p = p.sort_values("pred_label_match", ascending=False).reset_index(drop=True)
            p.to_csv( ROOT / "retrieval" / "core_program.csv", index=False)
        
            dp = f.run(json_path, top_n=3)

            out = program_labelmatch(dp)
            out = out.sort_values("pred_label_match", ascending=False).reset_index(drop=True)
            out = out.head(3).reset_index(drop=True)
            out.to_csv( ROOT / "retrieval" / "student_program.csv", index=False)

        st.session_state["last_output"] = out
        st.session_state["last_action"] = "programs"

# Mentors recommendation
with col2:
    if st.button("Find mentors for programs", use_container_width=True):
        rm = RetrievalMentor()
        df_m = rm.run(top_n=3)

        bundle = joblib.load(ROOT / "models" / "xgb_mentor_labelmatch_regressor.pkl")
        model = bundle["model"]
        mlb_int = bundle["mlb_int"]
        mlb_tag = bundle["mlb_tag"]

        df_m["field_tags"] = df_m["field_tags"].fillna("")
        df_m["expertise_tags"] = df_m["expertise_tags"].fillna("")

        def to_list(s):
            return [x.strip().lower() for x in str(s).split(";") if x.strip()]

        X_f = mlb_int.transform(df_m["field_tags"].map(to_list))
        X_e = mlb_tag.transform(df_m["expertise_tags"].map(to_list))
        X = hstack([X_f, X_e], format="csr")

        dX = xgb.DMatrix(X)
        pred = model.predict(dX)

        scored = df_m.copy()
        scored["pred_label_match"] = np.round(pred, 4)

        scored = scored.sort_values(["program_id", "pred_label_match"], ascending=[True, False])
        out = scored.groupby("program_id", as_index=False).head(3).reset_index(drop=True)
        out.to_csv(ROOT / "retrieval" / "program_mentor_scored.csv", index=False)

        st.session_state["last_output"] = out
        st.session_state["last_action"] = "mentors"

if st.session_state["last_output"] is not None:
    st.dataframe(st.session_state["last_output"].head(10), use_container_width=True)
