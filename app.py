import os
import io
import time
import json
import zipfile
import re
import base64
from typing import List, Dict, Any, Optional

# External Libraries - Ensure these are in requirements.txt
import streamlit as st
import pandas as pd
import yaml
import mammoth
from docx import Document
from PIL import Image

# Try to import provider SDKs, handle gracefully if not installed
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    from xai_sdk import Client
    from xai_sdk.chat import user as grok_user, system as grok_system
except ImportError:
    Client, grok_user, grok_system = None, None, None


# ==============================================================================
# --- SERVICES ---
# ==============================================================================

class ProviderError(Exception):
    pass

# --- Gemini Service ---
def run_gemini(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("google_api_key")
    if not api_key: raise ProviderError("Google API Key is not set. Please provide it in the sidebar.")
    if not genai: raise ProviderError("google-generativeai is not installed.")
    
    genai.configure(api_key=api_key)
    user_prompt_filled = user_prompt.replace("{{input}}", input_text)
    full_prompt = f"{system_prompt}\n\n{user_prompt_filled}"
    
    parts = [full_prompt]
    if images:
        for f in images:
            # Ensure we are at the start of the file before reading
            f.seek(0)
            parts.append(Image.open(io.BytesIO(f.read())))

    m = genai.GenerativeModel(model)
    # Safety settings can sometimes block valid content, so we set them to block only high probability
    safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
              {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
              {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
              {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}]
              
    resp = m.generate_content(parts, generation_config={"temperature": temperature, "max_output_tokens": max_tokens}, safety_settings=safety)
    return resp.text if hasattr(resp, 'text') else "Error: No text returned from Gemini (possible safety block)."

# --- OpenAI Service ---
def run_openai(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("openai_api_key")
    if not api_key: raise ProviderError("OpenAI API Key is not set. Please provide it in the sidebar.")
    if not OpenAI: raise ProviderError("openai is not installed.")

    client = OpenAI(api_key=api_key)
    content_items = [{"type": "text", "text": user_prompt.replace("{{input}}", input_text)}]
    if images:
        for f in images:
            f.seek(0)
            b64 = base64.b64encode(f.read()).decode("utf-8")
            content_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    msgs.append({"role": "user", "content": content_items})
    
    completion = client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
    return completion.choices[0].message.content or ""

# --- Grok Service ---
def run_grok(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    api_key = st.session_state.get("xai_api_key")
    if not api_key: raise ProviderError("XAI (Grok) API Key is not set. Please provide it in the sidebar.")
    if not Client: raise ProviderError("xai-sdk is not installed.")
    if images: st.warning("Grok provider does not currently support images in this app integration. Images ignored.")

    client = Client(api_key=api_key)
    chat = client.chat.create(model=model)
    if system_prompt: chat.append(grok_system(system_prompt))
    chat.append(grok_user(user_prompt.replace("{{input}}", input_text)))
    
    response = chat.sample(max_len=max_tokens, temp=temperature)
    return getattr(response, "content", "") or ""

# --- Model Router ---
def run_agent_step(agent_config: dict, input_text: str, images: Optional[List[Any]]) -> str:
    """Filters args and calls the correct provider, preventing 'unexpected keyword' errors."""
    provider = agent_config.get("provider")
    args = {
        "model": agent_config.get("model"),
        "system_prompt": agent_config.get("system_prompt", ""),
        "user_prompt": agent_config.get("user_prompt", "{{input}}"),
        "input_text": input_text,
        "temperature": float(agent_config.get("temperature", 0.5)),
        "max_tokens": int(agent_config.get("max_tokens", 2048)),
        "images": images
    }
    try:
        if provider == "gemini": return run_gemini(**args)
        elif provider == "openai": return run_openai(**args)
        elif provider == "grok": return run_grok(**args)
        else: raise ProviderError(f"Unknown provider: {provider}")
    except Exception as e:
        raise ProviderError(f"{provider} error: {str(e)}")

# ==============================================================================
# --- UTILS ---
# ==============================================================================
def parse_dataset_file(file) -> List[Dict[str, Any]]:
    name = file.name.lower()
    if name.endswith(".csv"): df = pd.read_csv(file)
    elif name.endswith(".json"): return json.load(file)
    elif name.endswith(".xlsx"): df = pd.read_excel(file)
    elif name.endswith(".ods"): df = pd.read_excel(file, engine="odf")
    elif name.endswith(".txt"): return [{"text": l} for l in file.getvalue().decode("utf-8").splitlines() if l.strip()]
    else: raise ValueError("Unsupported file type")
    return df.fillna("").to_dict(orient="records")

def parse_template_file(file) -> str:
    name = file.name.lower()
    if name.endswith((".txt", ".md")): return file.getvalue().decode("utf-8")
    elif name.endswith(".docx"): return mammoth.extract_raw_text(file).value
    raise ValueError("Unsupported template type")

def render_template(template: str, record: Dict[str, Any]) -> str:
    return re.sub(r"\{\{([^}]+)\}\}", lambda m: str(record.get(m.group(1).strip(), m.group(0))), template)

def export_docx(text: str) -> io.BytesIO:
    doc, buf = Document(), io.BytesIO()
    for line in str(text).splitlines(): doc.add_paragraph(line)
    doc.save(buf)
    buf.seek(0)
    return buf

def build_zip(docs: List[Dict[str,str]], ext: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, d in enumerate(docs):
            fname = os.path.splitext(d.get("filename") or f"doc_{i+1}")[0] + f".{ext}"
            content = d.get("content", "")
            if ext == "docx": zf.writestr(fname, export_docx(content).getvalue())
            else: zf.writestr(fname, content)
    buf.seek(0)
    return buf.getvalue()

# ==============================================================================
# --- STREAMLIT APP ---
# ==============================================================================
st.set_page_config(page_title="Agentic Docs Builder", page_icon="🌸", layout="wide")
ss = st.session_state

# --- UI THEMES ---
THEMES = { "Flora (Default)": ("#A55EEA", "#F0F2F6", "#E6E6FA", "#262730"), "Ocean Breeze": ("#1E90FF", "#F0F8FF", "#D6EAF8", "#17202A"), "Forest Whisper": ("#228B22", "#F0FFF0", "#D5F5E3", "#145A32"), "Sunset Glow": ("#FF4500", "#FFF5EE", "#FADBD8", "#641E16"), "Midnight Slate": ("#34495E", "#2C3E50", "#5D6D7E", "#ECF0F1"), "Sunny Meadow": ("#FFD700", "#FFFFF0", "#FCF3CF", "#873600"), "Lavender Dreams": ("#8A2BE2", "#F8F5FF", "#E8DAEF", "#300D4F"), "Ruby Red": ("#C70039", "#FFF0F0", "#F5B7B1", "#581845"), "Emerald Isle": ("#009B77", "#E8F8F5", "#A9DFBF", "#0E6655"), "Golden Harvest": ("#DAA520", "#FFFAF0", "#FDEBD0", "#7E5109"), "Arctic Ice": ("#5DADE2", "#EBF5FB", "#D4E6F1", "#154360"), "Warm Earth": ("#A0522D", "#FFF8DC", "#F5DEB3", "#512E0C"), "Cyberpunk Neon": ("#00FFFF", "#1B2631", "#283747", "#EAECEE"), "Pastel Cloud": ("#FFB6C1", "#FFF9FA", "#FADADD", "#6C3483"), "Volcanic Ash": ("#36454F", "#212F3D", "#2C3E50", "#FDFEFE"), "Mint Fresh": ("#66CDAA", "#F0FFFA", "#D1F2EB", "#0B5345"), "Berry Fusion": ("#8B0000", "#FAEBD7", "#F5CBA7", "#4A235A"), "Grape Vine": ("#6A0DAD", "#F4ECF7", "#D7BDE2", "#2C1B4B"), "Teal Focus": ("#008080", "#E0FFFF", "#B2DFDB", "#004D40"), "Mustard Zing": ("#FFDB58", "#FFFACD", "#F9E79F", "#5C4033")}
def get_theme_css(theme_name):
    p, bg, sbg, txt = THEMES.get(theme_name, THEMES["Flora (Default)"])
    return f"""<style>.badge{{padding:2px 8px;border-radius:8px;font-size:12px;display:inline-block;margin-right:6px}}.badge-ok{{background:#e6ffe6;color:#1a7f37;border:1px solid #1a7f37}}.badge-err{{background:#ffe6e6;color:#a12622;border:1px solid #a12622}}.status-dot{{height:10px;width:10px;border-radius:50%;display:inline-block;margin-right:6px}}.dot-green{{background:#00c853}}.dot-yellow{{background:#ffd600}}.dot-red{{background:#d50000}}.panel{{padding:10px 12px;border:1px solid #ccc;border-radius:8px;background:{sbg}}}.stApp{{background-color:{bg}}}h1,h2,h3,p,label,div[data-baseweb="tooltip"],div[data-testid="stMarkdownContainer"] p{{color:{txt} !important}}.stButton>button{{background-color:{p};color:white !important;border:1px solid {p}}}div[data-testid="stSidebarUserContent"]{{background-color:{sbg}}}</style>"""

# --- INIT STATE ---
DEFAULT_AGENTS = [{"name":"Summarizer","provider":"gemini","model":"gemini-2.5-flash","temperature":0.3,"max_tokens":1024,"system_prompt":"Summarize accurately.","user_prompt":"Summarize in 3 bullets:\n\n{{input}}"}]
for k,v in {"dataset":[],"schema":[],"generated_docs":[],"pipeline_history":[],"images":[],"template_text":"", "is_running":False,"selected_theme":"Flora (Default)","pipeline_input":""}.items():
    if k not in ss: ss[k] = v
if "agents" not in ss: ss.agents = DEFAULT_AGENTS
st.markdown(get_theme_css(ss.selected_theme), unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌸 Flora Controls")
    st.selectbox("UI Theme", list(THEMES.keys()), key="selected_theme")
    st.divider()
    def apikey(env, key, lbl):
        if env in os.environ: ss[key] = os.environ[env]
        val = ss.get(key)
        if not val:
            val = st.text_input(f"{lbl} API Key", type="password", key=f"in_{key}")
            if val: ss[key] = val; st.rerun()
        return f"<span class='badge {'badge-ok' if val else 'badge-err'}'>{lbl}</span>"
    c1,c2,c3 = st.columns(3)
    c1.markdown(apikey("GOOGLE_API_KEY","google_api_key","Gemini"), unsafe_allow_html=True)
    c2.markdown(apikey("OPENAI_API_KEY","openai_api_key","OpenAI"), unsafe_allow_html=True)
    c3.markdown(apikey("XAI_API_KEY","xai_api_key","Grok"), unsafe_allow_html=True)
    st.divider()
    ss.global_provider_override = st.selectbox("Provider Override", ["None","gemini","openai","grok"])
    if ss.global_provider_override == "None": ss.global_provider_override = None
    ss.global_model_override = st.selectbox("Model Override", ["None","gemini-2.5-flash","gemini-2.5-flash-lite","gpt-4.1-mini","gpt-4o-mini","grok-3-mini"])
    if ss.global_model_override == "None": ss.global_model_override = None
    st.divider()
    yf = st.file_uploader("Load agents.yaml", type=["yaml","yml"])
    if yf:
        try: ss.agents = yaml.safe_load(yf); st.success("Loaded agents.")
        except Exception as e: st.error(e)
    st.download_button("Save agents.yaml", yaml.safe_dump(ss.agents), "agents.yaml")
    st.divider()
    ss.images = st.file_uploader("Vision Images", type=["png","jpg","jpeg"], accept_multiple_files=True)

# --- MAIN ---
st.title("Agentic Docs Builder – Flora Edition")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1.Data", "2.Template", "3.Docs", "4.Agents", "5.Pipeline", "6.Export"])

with tab1:
    f = st.file_uploader("Upload Data", type=["csv","json","xlsx","txt"])
    if f:
        try: ss.dataset = parse_dataset_file(f); ss.schema = sorted(list(ss.dataset[0].keys())) if ss.dataset else []
        except Exception as e: st.error(e)
    if ss.dataset: st.dataframe(pd.DataFrame(ss.dataset).head(), use_container_width=True)

with tab2:
    c1,c2 = st.columns(2)
    with c1:
        tf = st.file_uploader("Upload Template", type=["txt","md","docx"])
        if tf: ss.template_text = parse_template_file(tf)
        st.text_area("Template Content", key="template_text", height=300)
        if st.button("Generate Docs", type="primary", disabled=not (ss.dataset and ss.template_text)):
            ss.generated_docs = [{"filename": r.get("filename",f"doc_{i}.txt"), "content": render_template(ss.template_text, r)} for i,r in enumerate(ss.dataset,1)]
            st.success(f"Generated {len(ss.generated_docs)} docs.")
    with c2:
        if ss.dataset and ss.template_text: st.markdown(f"<div class='panel'>{render_template(ss.template_text, ss.dataset[0])[:1500]}</div>", unsafe_allow_html=True)

with tab3:
    if not ss.generated_docs: st.info("No docs yet.")
    for i,d in enumerate(ss.generated_docs):
        with st.expander(f"Doc {i+1}: {d.get('filename')}"):
            d["filename"] = st.text_input("Filename", d["filename"], key=f"fn{i}")
            d["content"] = st.text_area("Content", d["content"], key=f"fc{i}", height=200)

with tab4:
    for i,a in enumerate(ss.agents):
        with st.expander(f"{i+1}. {a.get('name','Agent')}"):
            c1,c2,c3 = st.columns(3)
            a["name"] = c1.text_input("Name", a.get("name"), key=f"an{i}")
            a["provider"] = c2.selectbox("Provider", ["gemini","openai","grok"], ["gemini","openai","grok"].index(a.get("provider","gemini")), key=f"ap{i}")
            a["model"] = c3.text_input("Model", a.get("model","gemini-2.5-flash"), key=f"am{i}")
            a["system_prompt"] = st.text_area("System Prompt", a.get("system_prompt",""), key=f"asp{i}", height=100)
            a["user_prompt"] = st.text_area("User Prompt", a.get("user_prompt","{{input}}"), key=f"aup{i}", height=150)
    if st.button("➕ Add Agent"): ss.agents.append(DEFAULT_AGENTS[0].copy()); st.rerun()
    if st.button("🔄 Reset"): ss.agents = DEFAULT_AGENTS.copy(); st.rerun()

with tab5:
    st.subheader("Execute Pipeline")
    # FIXED: Explicit document loading to ensure input is not empty
    if ss.generated_docs:
        c1, c2 = st.columns([3, 1])
        sel_idx = c1.selectbox("Load Doc into Pipeline", range(len(ss.generated_docs)), format_func=lambda i: f"Doc {i+1}: {ss.generated_docs[i]['filename']}")
        if c2.button("Load Doc"):
            ss.pipeline_input = ss.generated_docs[sel_idx]["content"]
            st.rerun()

    st.text_area("Pipeline Input", key="pipeline_input", height=250)
    
    if st.button("Run Pipeline", type="primary", disabled=(ss.is_running or not ss.agents or not ss.pipeline_input)):
        ss.is_running, ss.pipeline_history = True, []
        prog, stat = st.progress(0), st.empty()
        curr = ss.pipeline_input
        for i, agent in enumerate(ss.agents, 1):
            ac = agent.copy()
            if ss.global_provider_override: ac["provider"] = ss.global_provider_override
            if ss.global_model_override: ac["model"] = ss.global_model_override
            stat.markdown(f"Running {i}/{len(ss.agents)}: **{ac['name']}**")
            prog.progress(i/len(ss.agents))
            try:
                out = run_agent_step(ac, curr, ss.images)
                ss.pipeline_history.append({"agent":ac['name'], "input":curr, "output":out, "error":None})
                curr = out
            except Exception as e:
                ss.pipeline_history.append({"agent":ac['name'], "input":curr, "output":None, "error":str(e)})
                st.error(e); break
        stat.success("Done!"); ss.is_running = False

    if ss.pipeline_history:
        for i, step in enumerate(ss.pipeline_history):
            with st.expander(f"Step {i+1}: {step['agent']}", expanded=bool(step["error"])):
                if step["error"]: st.error(step["error"])
                else: st.text_area("Output", step["output"], height=150, key=f"o{i}")

with tab6:
    if not ss.generated_docs: st.info("No docs.")
    else:
        fmt = st.selectbox("Format", ["txt","md","docx","zip-txt","zip-md","zip-docx"])
        if "zip" in fmt and st.button("Build ZIP"):
            st.download_button("Download ZIP", build_zip(ss.generated_docs, fmt.split("-")[1]), "docs.zip")
        elif "zip" not in fmt:
            idx = st.number_input("Doc Index", 1, len(ss.generated_docs), 1)-1
            d = ss.generated_docs[idx]
            data = export_docx(d["content"]) if fmt=="docx" else d["content"].encode("utf-8")
            st.download_button(f"Download .{fmt}", data, f"{os.path.splitext(d['filename'])[0]}.{fmt}")
