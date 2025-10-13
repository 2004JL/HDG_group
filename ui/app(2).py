import streamlit as st

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

# 👇 隐藏顶部菜单、右上角的“Deploy ▾”、以及底部的“Made with Streamlit”
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
        ("🌐","100% online" if p.get("online", True) else "Online & on-campus"),
        ("⏱️", p.get("duration","3 years full time or part time equivalent")),
        ("✅", p.get("entry","No ATAR required. Start with a subject.")),
    ]
    st.markdown('<div class="meta">', unsafe_allow_html=True)
    for icon, text in lines:
        st.markdown(f'<span><span>{icon}</span><span>{text}</span></span>', unsafe_allow_html=True)
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

# ================== 结果区（OUA 布局） ==================
# 顶部搜索框 + Tabs
search_q = st.text_input("Search for courses and subjects", "", placeholder="Search for courses and subjects")
tab1, tab2, tab3 = st.tabs(["Degrees","Subjects","Short courses"])

with tab1:
    st.markdown('<div class="oua-wrap">', unsafe_allow_html=True)
    # 头部标题行
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown('<div class="oua-hbar"><div class="oua-title">Course results for Degrees</div><div class="oua-saved">0 Saved degrees (to compare save at least 2)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:flex;justify-content:flex-end"><div class="oua-compare">Compare degrees</div></div>', unsafe_allow_html=True)

    # 分页状态
    st.session_state.setdefault("page", 1)
    PAGE_SIZE = 2
    total = len(st.session_state.DEMO_DEGREES)
    start = (st.session_state.page-1)*PAGE_SIZE
    end = start + PAGE_SIZE
    page_items = st.session_state.DEMO_DEGREES[start:end]

    # 网格两列
    idx = start
    for col, item in oua_colgrid(page_items, per_row=2):
        with col:
            render_oua_degree_card(item, idx)
            idx += 1

    # 分页按钮
    cols = st.columns(3)
    with cols[0]:
        if st.button("‹ Prev", disabled=st.session_state.page==1):
            st.session_state.page -= 1
            st.rerun()
    with cols[1]:
        st.markdown(f"<div class='pag'>Page {st.session_state.page} / { (total+PAGE_SIZE-1)//PAGE_SIZE }</div>", unsafe_allow_html=True)
    with cols[2]:
        if st.button("Next ›", disabled=end>=total):
            st.session_state.page += 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.info("Subjects tab (coming soon) — 保留和 OUA 一致的分栏结构。")
with tab3:
    st.info("Short courses tab (coming soon)。")
