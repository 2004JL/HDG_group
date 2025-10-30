import streamlit as st
import io
import sys
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retrieval_program import Retrieval
from retrieval.retrieval_mentor import RetrievalMentor
from retrieval.retrieval_core import RetrievalCore

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

sort_rank = st.checkbox("Sort by institution rank", value=False)

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
        if st.button("update", use_container_width=True):
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
            out.to_csv( ROOT / "retrieval" / "student_program.csv", index=False)

            out = out.head(3).reset_index(drop=True)
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
            out.to_csv( ROOT / "retrieval" / "student_program.csv", index=False)

            out = out.head(3).reset_index(drop=True)

        st.session_state["last_output"] = out
        st.session_state["last_action"] = "programs"

# Mentors recommendation
with col2:
    if st.button("Find mentors for programs", use_container_width=True):
        rm = RetrievalMentor()
        df = rm.run(top_n=3)

        bundle = joblib.load(ROOT / "models" / "xgb_mentor_labelmatch_regressor.pkl")
        model = bundle["model"]
        mlb_int = bundle["mlb_int"]
        mlb_tag = bundle["mlb_tag"]

        df["field_tags"] = df["field_tags"].fillna("")
        df["expertise_tags"] = df["expertise_tags"].fillna("")

        def to_list(s):
            return [x.strip().lower() for x in str(s).split(";") if x.strip()]

        X_f = mlb_int.transform(df["field_tags"].map(to_list))
        X_e = mlb_tag.transform(df["expertise_tags"].map(to_list))
        X = hstack([X_f, X_e], format="csr")

        dX = xgb.DMatrix(X)
        pred = model.predict(dX)

        scored = df.copy()
        scored["pred_label_match"] = np.round(pred, 4)

        scored = scored.sort_values(["program_id", "pred_label_match"], ascending=[True, False])
        scored.to_csv(ROOT / "retrieval" / "program_mentor.csv", index=False)

        out = scored.groupby("program_id", as_index=False).head(3).reset_index(drop=True)
        st.session_state["last_output"] = out
        st.session_state["last_action"] = "mentors"

def render_program_cards(df: pd.DataFrame, top_k: int = 3):
    small = df.head(top_k).reset_index(drop=True)

    def safe(row, key, default=""):
        return row.get(key, default)

    st.markdown("""
    <style>
    .prog-card{
        border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; margin:10px 0;
        box-shadow:0 2px 10px rgba(2,6,23,.05); background:#fff;
    }
    .prog-title{font-weight:700; font-size:1.05rem; margin-bottom:6px;}
    .prog-meta{color:#6b7280; font-size:.92rem; margin:2px 0;}
    .prog-link a{font-weight:600; text-decoration:underline;}
    </style>
    """, unsafe_allow_html=True)

    for i, row in small.iterrows():
        ins_name = safe(row, "institution_name")
        website = safe(row, "website")
        program_name = safe(row, "program_name")

        st.markdown(f"""
        <div class="prog-card">
          <div class="prog-title">{ins_name or 'Institution'}</div>
          <div class="prog-meta">program_name: <b>{program_name}</b></div>
          <div class="prog-link">website: {"<a href='"+website+"' target='_blank'>"+website+"</a>"}</div>
        </div>
        """, unsafe_allow_html=True)

def render_mentor_cards(df: pd.DataFrame, top_k_prog: int = 3, top_k_mentor: int = 3):
    cols = ["institution_name", "overall_ranking", "program_id", "program_name", "field_tags", "mentor_id", "mentor_name", "expertise_tags","languages", "years_experience", "pred_label_match"]
    df = df[cols].copy()

    prog_order = df["program_id"].drop_duplicates().head(top_k_prog).tolist()

    st.markdown("""
        <style>
            .prog-outer{
                border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; margin:14px 0; background:#fff;
            }
            .prog-title{font-weight:800; font-size:1.05rem; margin-bottom:8px;}
            .prog-sub{color:#6b7280; font-size:.92rem; margin-bottom:6px;}
            .mentor-card{
                border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; background:#fafafa;
            }
            .meta{color:#6b7280; font-size:.92rem; margin-bottom:8px;}
        </style>
    """, unsafe_allow_html=True)

    for pid in prog_order:
        sub = df[df["program_id"] == pid].head(top_k_mentor).reset_index(drop=True)
        program_name = sub.get("program_name").iloc[0]
        institution_name = sub.get("institution_name").iloc[0]

        st.markdown(f"""
            <div class="prog-outer">
                <div class="prog-title">Institution: {institution_name}</div>
                <div class="prog-sub">program: {program_name}</div>
                <div class="prog-sub">program_id: {pid}</div>
            </div>
        """, unsafe_allow_html=True)

        for r in sub.to_dict(orient="records"):
            mna = r.get("mentor_name", "")
            langs = r.get("languages", "")
            yrs = r.get("years_experience", "")

            st.markdown(f"""
                <div class="mentor-card">
                <div class="meta"><b>mentor_name: {mna} </b></div>
                <div class='meta'><b>languages: {langs} </b></div>
                <div class='meta'><b>years_experience: {yrs} </b></div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

if st.session_state["last_output"] is not None:
    df_out = st.session_state["last_output"]

    if sort_rank and "overall_ranking" in df_out.columns:
        df_out = df_out.sort_values("overall_ranking", ascending=True).reset_index(drop=True)

    if st.session_state.get("last_action") == "programs":
        render_program_cards(df_out, top_k=3)
    elif st.session_state.get("last_action") == "mentors":
        render_mentor_cards(df_out, top_k_prog=3, top_k_mentor=3)