import os
import io
import time
import json
import zipfile
import re
import base64
from typing import List, Dict, Any, Optional

# External Libraries - Make sure these are in requirements.txt
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
    Client = None
    grok_user = None
    grok_system = None


# ==============================================================================
# --- SERVICES (Originally in services/ folder) ---
# ==============================================================================

# --- Gemini Service ---
def _build_gemini_parts(system_prompt: str, user_prompt_filled: str, images: Optional[List[Any]]):
    parts = []
    # Gemini API prefers system instructions to be part of the first user message
    full_prompt = f"{system_prompt}\n\n{user_prompt_filled}"
    parts.append(full_prompt)
    
    if images:
        for f in images:
            try:
                # Reset buffer pointer if it has been read before
                f.seek(0)
                img_bytes = f.read()
                img = Image.open(io.BytesIO(img_bytes))
                parts.append(img)
            except Exception:
                # Silently ignore bad images
                pass
    return parts

def run_gemini(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    if not genai:
        raise RuntimeError("google-generativeai is not installed. Please add it to requirements.txt")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Please set it in your environment secrets.")
    
    genai.configure(api_key=api_key)
    
    user_prompt_filled = user_prompt.replace("{{input}}", input_text)
    m = genai.GenerativeModel(model)
    parts = _build_gemini_parts(system_prompt, user_prompt_filled, images)

    resp = m.generate_content(
        parts,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
    )
    return resp.text if hasattr(resp, "text") and resp.text else ""

# --- OpenAI Service ---
def _image_part(file) -> Optional[dict]:
    try:
        file.seek(0)
        b = file.read()
        b64 = base64.b64encode(b).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    except Exception:
        return None

def run_openai(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    if not OpenAI:
        raise RuntimeError("openai is not installed. Please add it to requirements.txt")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Please set it in your environment secrets.")
    
    client = OpenAI(api_key=api_key)
    
    content_items = [{"type": "text", "text": user_prompt.replace("{{input}}", input_text)}]
    if images:
        for f in images:
            part = _image_part(f)
            if part:
                content_items.append(part)

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": content_items})

    completion = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens
    )
    return completion.choices[0].message.content or ""

# --- Grok Service ---
def run_grok(model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    if not Client:
        raise RuntimeError("xai-sdk is not installed. Please add it to requirements.txt")
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY. Please set it in your environment secrets.")
    
    client = Client(api_key=api_key)
    chat = client.chat.create(model=model)
    
    if system_prompt:
        chat.append(grok_system(system_prompt))
    
    chat.append(grok_user(user_prompt.replace("{{input}}", input_text)))
    
    response = chat.sample(max_len=max_tokens, temp=temperature)
    return getattr(response, "content", "") or ""

# --- Model Router ---
class ProviderError(Exception):
    pass

def run_agent_step(provider: str, model: str, system_prompt: str, user_prompt: str, input_text: str, temperature: float, max_tokens: int, images: Optional[List[Any]] = None) -> str:
    try:
        if provider == "gemini":
            return run_gemini(model, system_prompt, user_prompt, input_text, temperature, max_tokens, images)
        elif provider == "openai":
            return run_openai(model, system_prompt, user_prompt, input_text, temperature, max_tokens, images)
        elif provider == "grok":
            if images:
                st.warning("Grok provider does not support images in this implementation. Ignoring images.")
            return run_grok(model, system_prompt, user_prompt, input_text, temperature, max_tokens, images)
        else:
            raise ProviderError(f"Unknown provider: {provider}")
    except Exception as e:
        # Catch-all for API errors, key issues, etc.
        raise ProviderError(str(e))


# ==============================================================================
# --- UTILS (Originally in utils/ folder) ---
# ==============================================================================

# --- Parser Utils ---
def parse_dataset_file(uploaded_file) -> List[Dict[str, Any]]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".json"):
        data = json.load(uploaded_file)
        if isinstance(data, list): return data
        raise ValueError("JSON must be an array of records.")
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".ods"):
        df = pd.read_excel(uploaded_file, engine="odf")
    elif name.endswith(".txt"):
        lines = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
        return [{"text": line} for line in lines if line.strip()]
    else:
        raise ValueError("Unsupported file type.")
    return df.fillna("").to_dict(orient="records")

def parse_template_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith((".txt", ".md")):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        result = mammoth.extract_raw_text(uploaded_file)
        return result.value
    raise ValueError("Unsupported template type. Use txt, md, or docx.")

def guess_schema_from_records(records: List[Dict[str, Any]]) -> List[str]:
    if not records: return []
    cols = set()
    for r in records[:50]:
        cols.update(r.keys())
    return sorted(list(cols))

# --- Template Utils ---
PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")

def render_template_on_record(template: str, record: Dict[str, Any]) -> str:
    def repl(match):
        key = match.group(1).strip()
        return str(record.get(key, f"{{{{{key}}}}}"))
    return PLACEHOLDER_RE.sub(repl, template)

def render_template_on_dataset(dataset: List[Dict[str, Any]], template: str) -> List[Dict[str, str]]:
    docs = []
    for i, rec in enumerate(dataset, start=1):
        content = render_template_on_record(template, rec)
        filename = rec.get("filename") or rec.get("title") or f"doc_{i}.txt"
        docs.append({"filename": str(filename), "content": content})
    return docs

# --- Exporter Utils ---
def build_zip_from_docs(docs: List[Dict[str,str]], ext: str = "txt") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, d in enumerate(docs, start=1):
            base, _ = os.path.splitext(d.get("filename") or f"doc_{i}")
            filename = f"{base}.{ext}"
            if ext == "docx":
                doc_buf = export_docx(d.get("content", ""))
                zf.writestr(filename, doc_buf.getvalue())
            else:
                zf.writestr(filename, d.get("content", ""))
    buf.seek(0)
    return buf.getvalue()

def export_docx(text: str) -> io.BytesIO:
    doc = Document()
    for line in str(text).splitlines():
        doc.add_paragraph(line)
    dbuf = io.BytesIO()
    doc.save(dbuf)
    dbuf.seek(0)
    return dbuf


# ==============================================================================
# --- STREAMLIT APP ---
# ==============================================================================

# -------------------------
# App Config & UI Themes
# -------------------------
st.set_page_config(
    page_title="Agentic Docs Builder - Flora Edition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEMES = {
    "Flora (Default)": ("#A55EEA", "#F0F2F6", "#E6E6FA", "#262730"), "Ocean Breeze": ("#1E90FF", "#F0F8FF", "#D6EAF8", "#17202A"),
    "Forest Whisper": ("#228B22", "#F0FFF0", "#D5F5E3", "#145A32"), "Sunset Glow": ("#FF4500", "#FFF5EE", "#FADBD8", "#641E16"),
    "Midnight Slate": ("#34495E", "#2C3E50", "#5D6D7E", "#ECF0F1"), "Sunny Meadow": ("#FFD700", "#FFFFF0", "#FCF3CF", "#873600"),
    "Lavender Dreams": ("#8A2BE2", "#F8F5FF", "#E8DAEF", "#300D4F"), "Ruby Red": ("#C70039", "#FFF0F0", "#F5B7B1", "#581845"),
    "Emerald Isle": ("#009B77", "#E8F8F5", "#A9DFBF", "#0E6655"), "Golden Harvest": ("#DAA520", "#FFFAF0", "#FDEBD0", "#7E5109"),
    "Arctic Ice": ("#5DADE2", "#EBF5FB", "#D4E6F1", "#154360"), "Warm Earth": ("#A0522D", "#FFF8DC", "#F5DEB3", "#512E0C"),
    "Cyberpunk Neon": ("#00FFFF", "#1B2631", "#283747", "#EAECEE"), "Pastel Cloud": ("#FFB6C1", "#FFF9FA", "#FADADD", "#6C3483"),
    "Volcanic Ash": ("#36454F", "#212F3D", "#2C3E50", "#FDFEFE"), "Mint Fresh": ("#66CDAA", "#F0FFFA", "#D1F2EB", "#0B5345"),
    "Berry Fusion": ("#8B0000", "#FAEBD7", "#F5CBA7", "#4A235A"), "Grape Vine": ("#6A0DAD", "#F4ECF7", "#D7BDE2", "#2C1B4B"),
    "Teal Focus": ("#008080", "#E0FFFF", "#B2DFDB", "#004D40"), "Mustard Zing": ("#FFDB58", "#FFFACD", "#F9E79F", "#5C4033")
}

def get_theme_css(theme_name: str) -> str:
    primary, bg, secondary_bg, text = THEMES.get(theme_name, THEMES["Flora (Default)"])
    return f"""
<style>
.badge {{padding: 2px 8px; border-radius: 8px; font-size: 12px; display: inline-block; margin-right: 6px;}}
.badge-ok {{background: #e6ffe6; color: #1a7f37; border: 1px solid #1a7f37;}}
.badge-err {{background: #ffe6e6; color: #a12622; border: 1px solid #a12622;}}
.status-dot {{height:10px; width:10px; border-radius:50%; display:inline-block; margin-right:6px;}}
.dot-green {{background:#00c853;}} .dot-yellow {{background:#ffd600;}} .dot-red {{background:#d50000;}}
.panel {{padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px; background: {secondary_bg};}}
.stApp {{ background-color: {bg}; }}
h1, h2, h3, h4, h5, h6, p, label, div[data-baseweb="tooltip"], div[data-testid="stMarkdownContainer"] p {{color: {text} !important;}}
.stButton>button {{background-color: {primary};color: white !important;border-radius: 8px;border: 1px solid {primary};}}
.stButton>button:hover {{background-color: {primary}e0; border: 1px solid {primary}e0;}}
div[data-testid="stSidebarUserContent"] {{background-color: {secondary_bg};}}
</style>"""

# -------------------------
# Default Agents YAML
# -------------------------
DEFAULT_AGENTS_YAML = """
- name: Summarizer
  provider: gemini
  model: gemini-1.5-flash
  temperature: 0.3
  max_tokens: 1024
  system_prompt: You are an expert summarizer. Your goal is to produce a concise, accurate summary.
  user_prompt: 'Summarize the following text in 3-5 clear bullet points:\\n\\n{{input}}'
- name: Style_Rewriter
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7
  max_tokens: 1024
  system_prompt: You are a professional editor. Rewrite text to be more clear, professional, and engaging.
  user_prompt: 'Rewrite this text to enhance its clarity and professional tone, preserving the core message:\\n\\n{{input}}'
"""
DEFAULT_AGENTS = yaml.safe_load(DEFAULT_AGENTS_YAML)

# -------------------------
# Session State Initialization
# -------------------------
ss = st.session_state
if "dataset" not in ss: ss.dataset = []
if "schema" not in ss: ss.schema = []
if "template_text" not in ss: ss.template_text = ""
if "generated_docs" not in ss: ss.generated_docs = []
if "agents" not in ss: ss.agents = DEFAULT_AGENTS.copy()
if "pipeline_history" not in ss: ss.pipeline_history = []
if "is_running" not in ss: ss.is_running = False
if "global_model_override" not in ss: ss.global_model_override = None
if "global_provider_override" not in ss: ss.global_provider_override = None
if "last_export_zip" not in ss: ss.last_export_zip = None
if "images" not in ss: ss.images = []
if "selected_theme" not in ss: ss.selected_theme = "Flora (Default)"

# Apply selected theme
st.markdown(get_theme_css(ss.selected_theme), unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("🌸 Flora Controls")
    st.selectbox("Choose a Theme", options=list(THEMES.keys()), key="selected_theme")
    st.divider()

    st.markdown("<div class='panel'>Set API keys in HF Secrets.</div>", unsafe_allow_html=True)
    st.write("API Key Status")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<span class='badge {'badge-ok' if os.getenv('GOOGLE_API_KEY') else 'badge-err'}'>Gemini</span>", unsafe_allow_html=True)
    c2.markdown(f"<span class='badge {'badge-ok' if os.getenv('OPENAI_API_KEY') else 'badge-err'}'>OpenAI</span>", unsafe_allow_html=True)
    c3.markdown(f"<span class='badge {'badge-ok' if os.getenv('XAI_API_KEY') else 'badge-err'}'>Grok</span>", unsafe_allow_html=True)
    st.divider()

    st.caption("Global Provider/Model Override")
    ALL_MODELS = ["None", "gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "llama3-70b-8192", "llama3-8b-8192"]
    prov_choice = st.selectbox("Provider override", ["None", "gemini", "openai", "grok"])
    mod_choice = st.selectbox("Model override", ALL_MODELS)
    ss.global_provider_override = None if prov_choice == "None" else prov_choice
    ss.global_model_override = None if mod_choice == "None" else mod_choice
    st.divider()

    st.caption("Agents Configuration")
    yaml_file = st.file_uploader("Load agents.yaml", type=["yaml", "yml"])
    if yaml_file:
        try:
            agents_loaded = yaml.safe_load(yaml_file)
            if isinstance(agents_loaded, list) and all(isinstance(i, dict) for i in agents_loaded):
                ss.agents = agents_loaded
                st.success("Loaded agents.yaml.")
            else: st.error("YAML must be a list of agent dictionaries.")
        except Exception as e: st.error(f"Error parsing YAML: {e}")
    
    st.download_button("Download Current agents.yaml", data=yaml.safe_dump(ss.agents).encode("utf-8"), file_name="agents.yaml")
    st.divider()

    st.caption("Images for Vision Models")
    imgs = st.file_uploader("Upload images", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
    if imgs: ss.images = imgs

# -------------------------
# Main App UI
# -------------------------
st.title("Agentic Docs Builder – Flora Edition")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(ss.dataset))
c2.metric("Schema Columns", len(ss.schema))
c3.metric("Generated Docs", len(ss.generated_docs))
c4.metric("Agents", len(ss.agents))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1. Data", "2. Template", "3. Docs", "4. Agents", "5. Pipeline", "6. Export"])

with tab1: # Data Ingestion
    st.subheader("Upload Your Dataset")
    ds_file = st.file_uploader("Supports CSV, JSON, XLSX, ODS, TXT", type=["csv","json","xlsx","ods","txt"])
    if ds_file:
        try:
            with st.spinner("Parsing file..."):
                records = parse_dataset_file(ds_file)
                ss.dataset, ss.schema = records, guess_schema_from_records(records)
            st.success(f"Loaded {len(records)} records.")
            if records: st.dataframe(pd.DataFrame(records).head(10), use_container_width=True)
        except Exception as e: st.error(f"Failed to parse: {e}")

with tab2: # Template
    st.subheader("Define a Document Template")
    c1, c2 = st.columns(2)
    with c1:
        tpl_file = st.file_uploader("Upload template (.txt, .md, .docx)", type=["txt","md","docx"])
        if tpl_file:
            try: ss.template_text = parse_template_file(tpl_file)
            except Exception as e: st.error(f"Failed to read: {e}")
        st.text_area("Template (use `{{column}}` placeholders)", key="template_text", height=300)
        if st.button("Generate Docs", type="primary", disabled=(not ss.dataset or not ss.template_text)):
            ss.generated_docs = render_template_on_dataset(ss.dataset, ss.template_text)
            st.success(f"Generated {len(ss.generated_docs)} documents.")
    with c2:
        st.markdown("#### Live Preview")
        if ss.dataset and ss.template_text:
            preview = render_template_on_record(ss.template_text, ss.dataset[0])
            st.markdown(f"<div class='panel'>{preview[:2000]}</div>", unsafe_allow_html=True)
        else: st.info("Upload data and a template for a preview.")

with tab3: # Generated Docs
    st.subheader("Review and Edit Generated Documents")
    if not ss.generated_docs: st.info("No documents generated yet.")
    for i, doc in enumerate(ss.generated_docs):
        with st.expander(f"Doc {i+1}: {doc.get('filename','doc_{i+1}')}"):
            ss.generated_docs[i]["filename"] = st.text_input("Filename", doc.get("filename"), key=f"fn_{i}")
            ss.generated_docs[i]["content"] = st.text_area("Content", doc.get("content"), height=250, key=f"fc_{i}")

with tab4: # Agent Config
    st.subheader("Configure Agent Chain")
    for idx, agent in enumerate(ss.agents):
        with st.expander(f"Agent {idx+1} • {agent.get('name', 'Agent')}"):
            c1, c2, c3 = st.columns(3)
            agent["name"] = c1.text_input("Name", agent.get("name"), key=f"a_name_{idx}")
            agent["provider"] = c2.selectbox("Provider", ["gemini","openai","grok"], index=["gemini","openai","grok"].index(agent.get("provider","gemini")), key=f"a_prov_{idx}")
            try: model_idx = ALL_MODELS.index(agent.get("model"))
            except ValueError: model_idx = 0
            agent["model"] = c3.selectbox("Model", ALL_MODELS, index=model_idx, key=f"a_model_{idx}")
            agent["temperature"] = st.slider("Temp", 0.0, 1.0, float(agent.get("temperature",0.5)), 0.05, key=f"a_temp_{idx}")
            agent["max_tokens"] = st.number_input("Max Tokens", 128, 16384, int(agent.get("max_tokens",2048)), 64, key=f"a_maxt_{idx}")
            agent["system_prompt"] = st.text_area("System Prompt", agent.get("system_prompt"), height=120, key=f"a_sysp_{idx}")
            agent["user_prompt"] = st.text_area("User Prompt (use {{input}})", agent.get("user_prompt"), height=180, key=f"a_usrp_{idx}")
    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("➕ Add New Agent"):
        ss.agents.append({"name": f"New Agent {len(ss.agents)+1}", "provider": "gemini", "model": "gemini-1.5-flash", "temperature": 0.5, "max_tokens": 2048, "system_prompt": "", "user_prompt": "{{input}}"})
        st.rerun()
    if c2.button("🔄 Reset to Default"):
        ss.agents = DEFAULT_AGENTS.copy()
        st.rerun()

with tab5: # Pipeline
    st.subheader("Execute Agent Pipeline")
    default_input = ss.generated_docs[0]["content"] if ss.generated_docs else ""
    input_text = st.text_area("Pipeline Input Text", default_input, height=250)
    if st.button("Run Pipeline", type="primary", disabled=(ss.is_running or not ss.agents)):
        ss.is_running, ss.pipeline_history = True, []
        progress, status_area = st.progress(0, "Initializing..."), st.empty()
        current_text = input_text
        for idx, agent in enumerate(ss.agents, 1):
            exec_agent = agent.copy()
            if ss.global_provider_override: exec_agent["provider"] = ss.global_provider_override
            if ss.global_model_override: exec_agent["model"] = ss.global_model_override
            status_area.markdown(f"<span class='status-dot dot-yellow'></span> Running {idx}/{len(ss.agents)}: **{exec_agent['name']}**", unsafe_allow_html=True)
            progress.progress(idx/len(ss.agents), f"Running: {exec_agent['name']}")
            try:
                output = run_agent_step(**exec_agent, input_text=current_text, images=ss.images)
                ss.pipeline_history.append({"agent": exec_agent['name'], "input": current_text, "output": output, "error": None})
                current_text = output
            except ProviderError as e:
                ss.pipeline_history.append({"agent": exec_agent['name'], "input": current_text, "output": None, "error": str(e)})
                status_area.markdown(f"<span class='status-dot dot-red'></span> Error on agent {idx}", unsafe_allow_html=True)
                st.error(f"Pipeline stopped: {e}")
                break
        status_area.markdown("<span class='status-dot dot-green'></span> Pipeline finished.", unsafe_allow_html=True)
        ss.is_running = False
    if ss.pipeline_history:
        st.markdown("--- \n### Pipeline Results")
        for i, step in enumerate(ss.pipeline_history):
            with st.expander(f"Step {i+1}: <{step['agent']}>", expanded=(step["error"] is not None)):
                st.text_area("Input", step["input"], height=150, disabled=True, key=f"in_{i}")
                if step["error"]: st.error(step["error"])
                else: st.text_area("Output", step["output"], height=150, disabled=True, key=f"out_{i}")

with tab6: # Export
    st.subheader("Export Documents")
    if not ss.generated_docs: st.info("Generate documents in the 'Template' tab to export.")
    else:
        fmt = st.selectbox("Export format", ["txt", "md", "docx", "zip-txt", "zip-md", "zip-docx"])
        if "zip" in fmt:
            if st.button("Build ZIP Archive", type="primary"):
                with st.spinner("Creating ZIP..."):
                    ss.last_export_zip = build_zip_from_docs(ss.generated_docs, ext=fmt.split("-")[1])
                st.success("ZIP ready.")
            if ss.last_export_zip:
                st.download_button("Download ZIP", ss.last_export_zip, f"docs_{int(time.time())}.zip")
        else:
            idx = st.number_input("Doc index (1-based)", 1, len(ss.generated_docs), 1) - 1
            doc = ss.generated_docs[idx]
            fname_base, _ = os.path.splitext(doc.get("filename") or f"doc_{idx+1}")
            if fmt == "docx":
                st.download_button("Download DOCX", export_docx(doc["content"]), f"{fname_base}.docx")
            else:
                st.download_button(f"Download .{fmt.upper()}", doc["content"].encode("utf-8"), f"{fname_base}.{fmt}")
