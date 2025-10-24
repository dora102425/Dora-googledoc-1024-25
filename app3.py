import os
import io
import time
import json
import zipfile
import re
import base64
from typing import List, Dict, Any, Optional

# External Libraries
import streamlit as st
import pandas as pd
import yaml
import mammoth
from docx import Document
from PIL import Image

# Try to import provider SDKs
try: import google.generativeai as genai
except ImportError: genai = None
try: from openai import OpenAI
except ImportError: OpenAI = None
try:
    from xai_sdk import Client
    from xai_sdk.chat import user as grok_user, system as grok_system
except ImportError: Client, grok_user, grok_system = None, None, None

# ==============================================================================
# --- SERVICES ---
# ==============================================================================
class ProviderError(Exception): pass

def run_gemini(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("google_api_key")
    if not api_key: raise ProviderError("Google API Key not set.")
    if not genai: raise ProviderError("google-generativeai not installed.")
    genai.configure(api_key=api_key)
    full_prompt = f"{system_prompt}\n\n{user_prompt.replace('{{input}}', input_text)}"
    parts = [full_prompt]
    if images:
        for f in images:
            f.seek(0); parts.append(Image.open(io.BytesIO(f.read())))
    return genai.GenerativeModel(model).generate_content(parts, generation_config={"temperature": temperature, "max_output_tokens": max_tokens}).text

def run_openai(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("openai_api_key")
    if not api_key: raise ProviderError("OpenAI API Key not set.")
    if not OpenAI: raise ProviderError("openai not installed.")
    client = OpenAI(api_key=api_key)
    content = [{"type": "text", "text": user_prompt.replace("{{input}}", input_text)}]
    if images:
        for f in images:
            f.seek(0); content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"}})
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    msgs.append({"role": "user", "content": content})
    return client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens).choices[0].message.content or ""

def run_grok(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("xai_api_key")
    if not api_key: raise ProviderError("XAI API Key not set.")
    if not Client: raise ProviderError("xai-sdk not installed.")
    if images: st.warning("Grok does not support images yet. Ignoring.")
    client = Client(api_key=api_key)
    chat = client.chat.create(model=model)
    if system_prompt: chat.append(grok_system(system_prompt))
    chat.append(grok_user(user_prompt.replace("{{input}}", input_text)))
    return getattr(chat.sample(max_len=max_tokens, temp=temperature), "content", "") or ""

def run_agent_step(agent_config: dict, input_text: str, images: Optional[List[Any]]) -> str:
    p = agent_config.get("provider")
    args = {k: agent_config.get(k) for k in ["model", "system_prompt", "user_prompt", "temperature", "max_tokens"]}
    args["input_text"] = input_text
    args["images"] = images
    args["temperature"] = float(args["temperature"] or 0.5)
    args["max_tokens"] = int(args["max_tokens"] or 2048)
    args["system_prompt"] = args["system_prompt"] or ""
    args["user_prompt"] = args["user_prompt"] or "{{input}}"
    
    try:
        if p == "gemini": return run_gemini(**args)
        elif p == "openai": return run_openai(**args)
        elif p == "grok": return run_grok(**args)
        raise ProviderError(f"Unknown provider: {p}")
    except Exception as e: raise ProviderError(str(e))

# ==============================================================================
# --- UTILS ---
# ==============================================================================
def parse_dataset_file(f) -> List[Dict[str, Any]]:
    n = f.name.lower()
    if n.endswith(".csv"): df = pd.read_csv(f)
    elif n.endswith(".json"): return json.load(f)
    elif n.endswith(".xlsx"): df = pd.read_excel(f)
    elif n.endswith(".ods"): df = pd.read_excel(f, engine="odf")
    elif n.endswith(".txt"): return [{"text": l} for l in f.getvalue().decode("utf-8").splitlines() if l.strip()]
    else: raise ValueError("Unsupported file type")
    return df.fillna("").to_dict(orient="records")

def parse_template_file(f) -> str:
    n = f.name.lower()
    if n.endswith((".txt", ".md")): return f.getvalue().decode("utf-8")
    elif n.endswith(".docx"): return mammoth.extract_raw_text(f).value
    raise ValueError("Unsupported template type")

def render_template(tpl: str, rec: Dict[str, Any]) -> str:
    return re.sub(r"\{\{([^}]+)\}\}", lambda m: str(rec.get(m.group(1).strip(), m.group(0))), tpl)

def build_zip(docs, ext):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for i, d in enumerate(docs):
            fn = f"{os.path.splitext(d.get('filename') or f'doc_{i+1}')[0]}.{ext}"
            c = d["content"]
            if ext == "docx":
                doc, db = Document(), io.BytesIO()
                for l in c.splitlines(): doc.add_paragraph(l)
                doc.save(db); z.writestr(fn, db.getvalue())
            else: z.writestr(fn, c.encode("utf-8") if ext=="md" else c)
    b.seek(0); return b.getvalue()

# ==============================================================================
# --- APP ---
# ==============================================================================
st.set_page_config(page_title="Agentic Docs Builder", page_icon="🌸", layout="wide")
ss = st.session_state

# State Init
if "agents" not in ss: ss.agents = yaml.safe_load("- {name: Summarizer, provider: gemini, model: gemini-1.5-flash, temperature: 0.3, max_tokens: 1024, system_prompt: '', user_prompt: 'Summarize:\\n\\n{{input}}'}")
for k in ["dataset", "schema", "generated_docs", "pipeline_history", "images"]:
    if k not in ss: ss[k] = []
for k,v in {"template_text":"", "is_running":False, "selected_theme":"Flora (Default)", "pipeline_input":""}.items():
    if k not in ss: ss[k] = v

THEMES = { "Flora (Default)": ("#A55EEA", "#F0F2F6", "#E6E6FA", "#262730"), "Ocean Breeze": ("#1E90FF", "#F0F8FF", "#D6EAF8", "#17202A"), "Forest Whisper": ("#228B22", "#F0FFF0", "#D5F5E3", "#145A32"), "Sunset Glow": ("#FF4500", "#FFF5EE", "#FADBD8", "#641E16"), "Midnight Slate": ("#34495E", "#2C3E50", "#5D6D7E", "#ECF0F1"), "Sunny Meadow": ("#FFD700", "#FFFFF0", "#FCF3CF", "#873600"), "Lavender Dreams": ("#8A2BE2", "#F8F5FF", "#E8DAEF", "#300D4F"), "Ruby Red": ("#C70039", "#FFF0F0", "#F5B7B1", "#581845"), "Emerald Isle": ("#009B77", "#E8F8F5", "#A9DFBF", "#0E6655"), "Golden Harvest": ("#DAA520", "#FFFAF0", "#FDEBD0", "#7E5109"), "Arctic Ice": ("#5DADE2", "#EBF5FB", "#D4E6F1", "#154360"), "Warm Earth": ("#A0522D", "#FFF8DC", "#F5DEB3", "#512E0C"), "Cyberpunk Neon": ("#00FFFF", "#1B2631", "#283747", "#EAECEE"), "Pastel Cloud": ("#FFB6C1", "#FFF9FA", "#FADADD", "#6C3483"), "Volcanic Ash": ("#36454F", "#212F3D", "#2C3E50", "#FDFEFE"), "Mint Fresh": ("#66CDAA", "#F0FFFA", "#D1F2EB", "#0B5345"), "Berry Fusion": ("#8B0000", "#FAEBD7", "#F5CBA7", "#4A235A"), "Grape Vine": ("#6A0DAD", "#F4ECF7", "#D7BDE2", "#2C1B4B"), "Teal Focus": ("#008080", "#E0FFFF", "#B2DFDB", "#004D40"), "Mustard Zing": ("#FFDB58", "#FFFACD", "#F9E79F", "#5C4033")}
p, bg, sbg, txt = THEMES[ss.selected_theme]
st.markdown(f"<style>.stApp{{background-color:{bg};}} h1,h2,h3,p,label,div[data-testid='stMarkdownContainer'] p{{color:{txt} !important;}} .stButton>button{{background-color:{p};color:white !important;border:1px solid {p};}} div[data-testid='stSidebarUserContent'],.panel{{background-color:{sbg};}} .panel{{padding:10px;border-radius:8px;border:1px solid #ccc;}} .badge{{padding:2px 8px;border-radius:8px;font-size:12px;margin-right:6px;}} .badge-ok{{background:#e6ffe6;color:#1a7f37;border:1px solid #1a7f37;}} .badge-err{{background:#ffe6e6;color:#a12622;border:1px solid #a12622;}} .status-dot{{height:10px;width:10px;border-radius:50%;display:inline-block;margin-right:6px;}} .dot-green{{background:#00c853;}} .dot-yellow{{background:#ffd600;}} .dot-red{{background:#d50000;}}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🌸 Flora Controls")
    st.selectbox("Theme", list(THEMES.keys()), key="selected_theme")
    st.divider(); st.caption("API Keys")
    def key_ui(env, k, lbl):
        if env in os.environ: ss[k] = os.environ[env]
        ok = ss.get(k); st.markdown(f"<span class='badge {'badge-ok' if ok else 'badge-err'}'>{lbl}</span>", unsafe_allow_html=True)
        if not ok and st.text_input(f"Enter {lbl} Key", type="password", key=f"i_{k}"): ss[k] = ss[f"i_{k}"]; st.rerun()
    c1,c2,c3 = st.columns(3)
    with c1: key_ui("GOOGLE_API_KEY", "google_api_key", "Gemini")
    with c2: key_ui("OPENAI_API_KEY", "openai_api_key", "OpenAI")
    with c3: key_ui("XAI_API_KEY", "xai_api_key", "Grok")
    st.divider(); st.caption("Global Overrides")
    ss.global_provider_override = st.selectbox("Provider", ["None", "gemini", "openai", "grok"])
    if ss.global_provider_override == "None": ss.global_provider_override = None
    ss.global_model_override = st.selectbox("Model", ["None", "gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini", "llama3-70b-8192", "llama3-8b-8192"])
    if ss.global_model_override == "None": ss.global_model_override = None
    st.divider(); st.caption("Configuration")
    if (yf := st.file_uploader("Load agents.yaml", type=["yaml", "yml"])): ss.agents = yaml.safe_load(yf)
    st.download_button("Save agents.yaml", yaml.safe_dump(ss.agents), "agents.yaml")
    ss.images = st.file_uploader("Vision Images", type=["png","jpg","jpeg"], accept_multiple_files=True)

st.title("Agentic Docs Builder – Flora Edition")
t1,t2,t3,t4,t5,t6 = st.tabs(["1.Data","2.Template","3.Docs","4.Agents","5.Pipeline","6.Export"])

with t1:
    if (df := st.file_uploader("Upload Data", type=["csv","json","xlsx","ods","txt"])):
        try: ss.dataset = parse_dataset_file(df); ss.schema = sorted(list(set().union(*(d.keys() for d in ss.dataset[:50]))))
        except Exception as e: st.error(e)
    st.metric("Records", len(ss.dataset)); if ss.dataset: st.dataframe(pd.DataFrame(ss.dataset).head(), use_container_width=True)

with t2:
    c1, c2 = st.columns(2)
    with c1:
        if (tf := st.file_uploader("Upload Template", type=["txt","md","docx"])): ss.template_text = parse_template_file(tf)
        st.text_area("Template Content", key="template_text", height=300)
        if st.button("Generate Docs", type="primary", disabled=not (ss.dataset and ss.template_text)):
            ss.generated_docs = [{"filename": r.get("filename",f"doc_{i}.txt"), "content": render_template(ss.template_text, r)} for i,r in enumerate(ss.dataset,1)]
            if ss.generated_docs: ss.pipeline_input = ss.generated_docs[0]["content"] # AUTO-FILL PIPELINE INPUT
            st.success(f"Generated {len(ss.generated_docs)} docs.")
    with c2:
        if ss.dataset and ss.template_text: st.markdown(f"**Preview:**<div class='panel'>{render_template(ss.template_text, ss.dataset[0])[:2000]}</div>", unsafe_allow_html=True)

with t3:
    for i,d in enumerate(ss.generated_docs):
        with st.expander(f"Doc {i+1}: {d['filename']}"):
            d["filename"] = st.text_input("Name", d["filename"], key=f"fn{i}")
            d["content"] = st.text_area("Content", d["content"], height=200, key=f"fc{i}")

with t4:
    for i,a in enumerate(ss.agents):
        with st.expander(f"{i+1}. {a.get('name','Agent')}"):
            c1,c2,c3 = st.columns(3)
            a["name"] = c1.text_input("Name", a.get("name"), key=f"an{i}")
            a["provider"] = c2.selectbox("Provider", ["gemini","openai","grok"], ["gemini","openai","grok"].index(a.get("provider","gemini")), key=f"ap{i}")
            a["model"] = c3.text_input("Model", a.get("model","gemini-1.5-flash"), key=f"am{i}")
            a["temperature"] = st.slider("Temp",0.0,1.0,float(a.get("temperature",0.5)),key=f"at{i}")
            a["max_tokens"] = st.number_input("Tokens",1,32768,int(a.get("max_tokens",2048)),key=f"amt{i}")
            a["system_prompt"] = st.text_area("System Prompt", a.get("system_prompt",""), height=100, key=f"asp{i}")
            a["user_prompt"] = st.text_area("User Prompt", a.get("user_prompt","{{input}}"), height=150, key=f"aup{i}")
    if st.button("➕ Add Agent"): ss.agents.append({"name":"New","provider":"gemini","model":"gemini-1.5-flash","user_prompt":"{{input}}"}); st.rerun()

with t5:
    # Explicitly manage pipeline input with a loader
    if ss.generated_docs:
        doc_opts = {f"Doc {i+1}: {d['filename']}": d['content'] for i,d in enumerate(ss.generated_docs)}
        if (sel := st.selectbox("Load Document", [""] + list(doc_opts.keys()))) and st.button("Load Selected Doc"):
            ss.pipeline_input = doc_opts[sel]; st.rerun()
            
    st.text_area("Pipeline Input", key="pipeline_input", height=250)
    
    if st.button("Run Pipeline", type="primary", disabled=ss.is_running):
        ss.is_running, ss.pipeline_history = True, []
        prog, stat = st.progress(0), st.empty()
        curr = ss.pipeline_input
        for i,a in enumerate(ss.agents,1):
            ac = a.copy()
            if ss.global_provider_override: ac["provider"] = ss.global_provider_override
            if ss.global_model_override: ac["model"] = ss.global_model_override
            stat.markdown(f"<span class='status-dot dot-yellow'></span> Running **{ac['name']}**...", unsafe_allow_html=True)
            prog.progress(i/len(ss.agents))
            try:
                out = run_agent_step(ac, curr, ss.images)
                ss.pipeline_history.append({"agent":ac['name'],"input":curr,"output":out,"error":None})
                curr = out
            except Exception as e:
                ss.pipeline_history.append({"agent":ac['name'],"input":curr,"output":None,"error":str(e)})
                st.error(str(e)); break
        stat.markdown("<span class='status-dot dot-green'></span> Done.", unsafe_allow_html=True)
        ss.is_running = False

    for i,s in enumerate(ss.pipeline_history):
        with st.expander(f"Step {i+1}: {s['agent']}", expanded=bool(s['error'])):
            st.text_area("In", s["input"], height=100, disabled=True, key=f"si{i}")
            if s['error']: st.error(s['error'])
            else: st.text_area("Out", s["output"], height=150, disabled=True, key=f"so{i}")

with t6:
    if not ss.generated_docs: st.info("No docs to export.")
    else:
        fmt = st.selectbox("Format", ["zip-txt", "zip-md", "zip-docx", "txt", "md", "docx"])
        if "zip" in fmt:
            if st.button("Generate ZIP"): st.download_button("Download ZIP", build_zip(ss.generated_docs, fmt.split("-")[1]), "docs.zip")
        else:
            idx = st.number_input("Doc Index", 1, len(ss.generated_docs), 1) - 1
            d = ss.generated_docs[idx]
            ext, mime = (fmt, "text/plain") if fmt != "docx" else ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            data = d["content"].encode("utf-8") if fmt != "docx" else export_docx(d["content"]).getvalue()
            st.download_button(f"Download .{ext}", data, f"{os.path.splitext(d['filename'])[0]}.{ext}", mime=mime)
