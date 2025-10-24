import os
import io
import time
import json
import zipfile
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import yaml

# Local Imports (will now work with the __init__.py files)
from utils.parser import parse_dataset_file, parse_template_file, guess_schema_from_records
from utils.template import render_template_on_record, render_template_on_dataset
from utils.exporters import build_zip_from_docs, export_docx
from services.model_router import run_agent_step, ProviderError

# -------------------------
# App Config & UI Themes
# -------------------------
st.set_page_config(
    page_title="Agentic Docs Builder - Flora Edition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 20 "Wow" Themes ---
THEMES = {
    "Flora (Default)": ("#A55EEA", "#F0F2F6", "#E6E6FA", "#262730"),
    "Ocean Breeze": ("#1E90FF", "#F0F8FF", "#D6EAF8", "#17202A"),
    "Forest Whisper": ("#228B22", "#F0FFF0", "#D5F5E3", "#145A32"),
    "Sunset Glow": ("#FF4500", "#FFF5EE", "#FADBD8", "#641E16"),
    "Midnight Slate": ("#34495E", "#2C3E50", "#5D6D7E", "#ECF0F1"),
    "Sunny Meadow": ("#FFD700", "#FFFFF0", "#FCF3CF", "#873600"),
    "Lavender Dreams": ("#8A2BE2", "#F8F5FF", "#E8DAEF", "#300D4F"),
    "Ruby Red": ("#C70039", "#FFF0F0", "#F5B7B1", "#581845"),
    "Emerald Isle": ("#009B77", "#E8F8F5", "#A9DFBF", "#0E6655"),
    "Golden Harvest": ("#DAA520", "#FFFAF0", "#FDEBD0", "#7E5109"),
    "Arctic Ice": ("#5DADE2", "#EBF5FB", "#D4E6F1", "#154360"),
    "Warm Earth": ("#A0522D", "#FFF8DC", "#F5DEB3", "#512E0C"),
    "Cyberpunk Neon": ("#00FFFF", "#1B2631", "#283747", "#EAECEE"),
    "Pastel Cloud": ("#FFB6C1", "#FFF9FA", "#FADADD", "#6C3483"),
    "Volcanic Ash": ("#36454F", "#212F3D", "#2C3E50", "#FDFEFE"),
    "Mint Fresh": ("#66CDAA", "#F0FFFA", "#D1F2EB", "#0B5345"),
    "Berry Fusion": ("#8B0000", "#FAEBD7", "#F5CBA7", "#4A235A"),
    "Grape Vine": ("#6A0DAD", "#F4ECF7", "#D7BDE2", "#2C1B4B"),
    "Teal Focus": ("#008080", "#E0FFFF", "#B2DFDB", "#004D40"),
    "Mustard Zing": ("#FFDB58", "#FFFACD", "#F9E79F", "#5C4033")
}

def get_theme_css(theme_name: str) -> str:
    primary, bg, secondary_bg, text = THEMES.get(theme_name, THEMES["Flora (Default)"])
    return f"""
<style>
.badge {{padding: 2px 8px; border-radius: 8px; font-size: 12px; display: inline-block; margin-right: 6px;}}
.badge-ok {{background: #e6ffe6; color: #1a7f37; border: 1px solid #1a7f37;}}
.badge-warn {{background: #fffbe6; color: #8a6d3b; border: 1px solid #8a6d3b;}}
.badge-err {{background: #ffe6e6; color: #a12622; border: 1px solid #a12622;}}
.status-dot {{height:10px; width:10px; border-radius:50%; display:inline-block; margin-right:6px;}}
.dot-green {{background:#00c853;}}
.dot-yellow {{background:#ffd600;}}
.dot-red {{background:#d50000;}}
.panel {{padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px; background: {secondary_bg};}}
.metric-card {{border:1px solid #ddd; border-radius:8px; padding:6px 10px; background: #fff; color: #333;}}

.stApp {{ background-color: {bg}; }}
h1, h2, h3, h4, h5, h6, p, label, div[data-baseweb="tooltip"], div[data-testid="stMarkdownContainer"] p {{
    color: {text} !important;
}}
.stButton>button {{
    background-color: {primary};
    color: white !important;
    border-radius: 8px;
    border: 1px solid {primary};
}}
.stButton>button:hover {{
    background-color: {primary}e0; /* Add transparency for hover effect */
    border: 1px solid {primary}e0;
}}
div[data-testid="stSidebarUserContent"] {{ /* Sidebar background */
    background-color: {secondary_bg};
}}
</style>
"""

# -------------------------
# Default Agents YAML
# -------------------------
DEFAULT_AGENTS_YAML = """
- name: Summarizer
  provider: gemini
  model: gemini-1.5-flash
  temperature: 0.3
  max_tokens: 1024
  system_prompt: You are an expert summarizer. Your goal is to produce a concise and accurate summary of the provided text.
  user_prompt: 'Summarize the following text in 3-5 clear bullet points:

    {{input}}'
- name: Style_Rewriter
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7
  max_tokens: 1024
  system_prompt: You are a professional editor. Rewrite the text to be more clear, professional, and engaging.
  user_prompt: 'Rewrite this text to enhance its clarity and professional tone, while preserving the core message:

    {{input}}'
"""
DEFAULT_AGENTS = yaml.safe_load(DEFAULT_AGENTS_YAML)

# -------------------------
# Session State Initialization
# -------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = []
if "schema" not in st.session_state:
    st.session_state.schema = []
if "template_text" not in st.session_state:
    st.session_state.template_text = ""
if "generated_docs" not in st.session_state:
    st.session_state.generated_docs = []
if "agents" not in st.session_state:
    st.session_state.agents = DEFAULT_AGENTS.copy()
if "pipeline_history" not in st.session_state:
    st.session_state.pipeline_history = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "global_model_override" not in st.session_state:
    st.session_state.global_model_override = None
if "global_provider_override" not in st.session_state:
    st.session_state.global_provider_override = None
if "last_export_zip" not in st.session_state:
    st.session_state.last_export_zip = None
if "images" not in st.session_state:
    st.session_state.images = []
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "Flora (Default)"

# Apply selected theme
st.markdown(get_theme_css(st.session_state.selected_theme), unsafe_allow_html=True)


# -------------------------
# Sidebar: Keys & Global Controls
# -------------------------
with st.sidebar:
    st.header("🌸 Flora Controls")
    
    # Theme Selector
    st.selectbox(
        "Choose a Theme",
        options=list(THEMES.keys()),
        key="selected_theme",
    )
    st.divider()

    st.markdown("<div class='panel'>Set API keys in Secrets for Spaces.</div>", unsafe_allow_html=True)

    google_key_ok = bool(os.getenv("GOOGLE_API_KEY"))
    openai_key_ok = bool(os.getenv("OPENAI_API_KEY"))
    xai_key_ok = bool(os.getenv("XAI_API_KEY"))

    st.write("API Key Status")
    colk1, colk2, colk3 = st.columns(3)
    colk1.markdown(f"<span class='badge {'badge-ok' if google_key_ok else 'badge-err'}'>Gemini</span>", unsafe_allow_html=True)
    colk2.markdown(f"<span class='badge {'badge-ok' if openai_key_ok else 'badge-err'}'>OpenAI</span>", unsafe_allow_html=True)
    colk3.markdown(f"<span class='badge {'badge-ok' if xai_key_ok else 'badge-err'}'>Grok</span>", unsafe_allow_html=True)

    st.divider()
    st.caption("Global Provider/Model Override")
    ALL_MODELS = [
        "None", "gemini-1.5-flash", "gemini-1.5-pro",
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        "llama3-70b-8192", "llama3-8b-8192"
    ]
    provider_choice = st.selectbox("Provider override", ["None", "gemini", "openai", "grok"], index=0, key="global_provider_select")
    model_choice = st.selectbox("Model override", ALL_MODELS, index=0, key="global_model_select")
    st.session_state.global_provider_override = None if provider_choice == "None" else provider_choice
    st.session_state.global_model_override = None if model_choice == "None" else model_choice

    st.divider()
    st.caption("Agents Configuration")
    yaml_file = st.file_uploader("Load agents.yaml", type=["yaml", "yml"])
    if yaml_file is not None:
        try:
            agents_loaded = yaml.safe_load(yaml_file)
            if isinstance(agents_loaded, list) and all(isinstance(item, dict) for item in agents_loaded):
                st.session_state.agents = agents_loaded
                st.success("Loaded agents.yaml successfully.")
            else:
                st.error("Invalid YAML: Must be a list of agent dictionaries.")
        except Exception as e:
            st.error(f"Error parsing YAML: {e}")

    def download_agents_yaml():
        buf = io.StringIO()
        yaml.safe_dump(st.session_state.agents, buf, sort_keys=False, allow_unicode=True)
        return buf.getvalue().encode("utf-8")

    st.download_button("Download Current agents.yaml", data=download_agents_yaml(), file_name="agents.yaml")

    st.divider()
    st.caption("Images for Vision Models")
    imgs = st.file_uploader("Upload images (for Gemini/OpenAI)", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
    if imgs:
        st.session_state.images = imgs

# -------------------------
# Header / Dashboard
# -------------------------
st.title("Agentic Docs Builder – Flora Edition")
colA, colB, colC, colD = st.columns(4)
colA.metric("Records Loaded", len(st.session_state.dataset))
colB.metric("Schema Columns", len(st.session_state.schema))
colC.metric("Generated Docs", len(st.session_state.generated_docs))
colD.metric("Configured Agents", len(st.session_state.agents))

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data Ingestion", "2. Template", "3. Generated Docs", "4. Agent Configuration", "5. Run Pipeline", "6. Export"
])

with tab1:
    st.subheader("Upload Your Dataset")
    st.markdown("Supports CSV, JSON (array of objects), XLSX, ODS, or TXT (one record per line).")
    dataset_file = st.file_uploader("Choose a file", type=["csv","json","xlsx","ods","txt"])
    if dataset_file is not None:
        try:
            with st.spinner("Parsing file..."):
                records = parse_dataset_file(dataset_file)
                st.session_state.dataset = records
                st.session_state.schema = guess_schema_from_records(records)
            st.success(f"Successfully loaded {len(records)} records.")
            if len(records) > 0:
                st.dataframe(pd.DataFrame(records).head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to parse dataset: {e}")

with tab2:
    st.subheader("Define a Document Template")
    st.markdown("Use `{{column_name}}` placeholders to insert data from your dataset.")
    colT1, colT2 = st.columns(2)
    with colT1:
        template_file = st.file_uploader("Or upload a template", type=["txt","md","docx"])
        if template_file is not None:
            try:
                st.session_state.template_text = parse_template_file(template_file)
                st.success("Template loaded.")
            except Exception as e:
                st.error(f"Failed to read template: {e}")
        st.text_area("Template Content", key="template_text", height=300)
        
        gen_col1, gen_col2 = st.columns(2)
        if gen_col1.button("Generate Documents", type="primary", disabled=(len(st.session_state.dataset)==0 or not st.session_state.template_text.strip())):
            try:
                docs = render_template_on_dataset(st.session_state.dataset, st.session_state.template_text)
                st.session_state.generated_docs = docs
                st.success(f"Generated {len(docs)} documents.")
            except Exception as e:
                st.error(f"Generation error: {e}")
        if gen_col2.button("Clear Documents"):
            st.session_state.generated_docs = []
            st.info("Cleared generated documents.")
            
    with colT2:
        st.markdown("#### Live Preview (using first record)")
        if len(st.session_state.dataset) > 0 and st.session_state.template_text.strip():
            preview = render_template_on_record(st.session_state.template_text, st.session_state.dataset[0])
            st.markdown(f"<div class='panel'>{preview[:4000]}</div>", unsafe_allow_html=True)
        else:
            st.info("Upload data and define a template to see a preview.")

with tab3:
    st.subheader("Review and Edit Generated Documents")
    if not st.session_state.generated_docs:
        st.info("No documents generated yet. Go to the 'Template' tab to create them.")
    else:
        for i, doc in enumerate(st.session_state.generated_docs):
            with st.expander(f"Document {i+1}: {doc.get('filename','doc_'+str(i+1))}"):
                new_name = st.text_input("Filename", value=doc.get("filename", f"doc_{i+1}.txt"), key=f"fn_{i}")
                new_content = st.text_area("Content", value=doc.get("content",""), height=250, key=f"fc_{i}")
                st.session_state.generated_docs[i]["filename"] = new_name
                st.session_state.generated_docs[i]["content"] = new_content

with tab4:
    st.subheader("Configure Agent Chain")
    st.caption("Edit agent parameters. Use the sidebar to override providers/models globally for testing.")
    for idx, agent in enumerate(st.session_state.agents):
        with st.expander(f"Agent {idx+1} • {agent.get('name', 'Agent')}"):
            c1, c2, c3 = st.columns([1,1,1])
            agent["name"] = c1.text_input("Name", value=agent.get("name","Agent"), key=f"a_name_{idx}")
            agent["provider"] = c2.selectbox("Provider", ["gemini","openai","grok"], index=["gemini","openai","grok"].index(agent.get("provider","gemini")), key=f"a_provider_{idx}")
            
            current_model = agent.get("model","gemini-1.5-flash")
            try: model_index = ALL_MODELS.index(current_model)
            except ValueError: model_index = 0
            
            agent["model"] = c3.selectbox( "Model", ALL_MODELS, index=model_index, key=f"a_model_{idx}" )
            agent["temperature"] = st.slider("Temperature", 0.0, 1.0, float(agent.get("temperature",0.5)), 0.05, key=f"a_temp_{idx}")
            agent["max_tokens"] = st.number_input("Max Tokens", 128, 16384, int(agent.get("max_tokens",2048)), 64, key=f"a_maxtok_{idx}")
            agent["system_prompt"] = st.text_area("System Prompt", value=agent.get("system_prompt",""), height=120, key=f"a_sysp_{idx}")
            agent["user_prompt"] = st.text_area("User Prompt (use {{input}} token)", value=agent.get("user_prompt",""), height=180, key=f"a_usrp_{idx}")

    st.markdown("---")
    colBA1, colBA2 = st.columns(2)
    if colBA1.button("➕ Add New Agent"):
        st.session_state.agents.append({
            "name": f"New Agent {len(st.session_state.agents) + 1}", "provider": "gemini", 
            "model": "gemini-1.5-flash", "temperature": 0.5, "max_tokens": 2048,
            "system_prompt": "You are a helpful assistant.", "user_prompt": "{{input}}"})
        st.rerun()
    if colBA2.button("🔄 Reset to Default Agents"):
        st.session_state.agents = DEFAULT_AGENTS.copy()
        st.rerun()

with tab5:
    st.subheader("Execute Agent Pipeline")
    st.caption("Run a single input through the configured agent chain to test its performance.")

    if st.session_state.images:
        uses_grok = any(agent.get("provider") == "grok" for agent in st.session_state.agents)
        if uses_grok:
            st.warning("🖼️ Images have been uploaded, but the **Grok** provider in this app currently does not support image input. Images will be ignored for Grok agents.")

    default_input = st.session_state.generated_docs[0]["content"] if st.session_state.generated_docs else ""
    input_text = st.text_area("Pipeline Input Text", value=default_input, height=250, key="pipeline_input_text")

    if st.button("Run Pipeline", type="primary", disabled=(st.session_state.is_running or not st.session_state.agents)):
        st.session_state.is_running = True
        st.session_state.pipeline_history = []
        progress = st.progress(0, "Initializing pipeline...")
        status_area = st.empty()
        
        current_text = input_text
        total_agents = len(st.session_state.agents)
        
        for idx, agent in enumerate(st.session_state.agents, start=1):
            start_t = time.time()
            exec_agent = agent.copy()
            if st.session_state.global_provider_override: exec_agent["provider"] = st.session_state.global_provider_override
            if st.session_state.global_model_override: exec_agent["model"] = st.session_state.global_model_override
            
            status_area.markdown(f"<span class='status-dot dot-yellow'></span> Running agent {idx}/{total_agents}: **{exec_agent.get('name')}**", unsafe_allow_html=True)
            progress.progress(idx / total_agents, f"Running: {exec_agent.get('name')}")
            
            try:
                output_text = run_agent_step(
                    provider=exec_agent["provider"], model=exec_agent["model"],
                    system_prompt=exec_agent.get("system_prompt",""), user_prompt=exec_agent.get("user_prompt","{{input}}"),
                    input_text=current_text, temperature=float(exec_agent.get("temperature",0.5)),
                    max_tokens=int(exec_agent.get("max_tokens",2048)),
                    images=st.session_state.images if exec_agent["provider"] != "grok" else None
                )
                elapsed = time.time() - start_t
                st.session_state.pipeline_history.append({"agent": exec_agent.get("name"), "input": current_text, "output": output_text, "error": None, "elapsed": elapsed})
                current_text = output_text
                status_area.markdown(f"<span class='status-dot dot-green'></span> Completed agent {idx}/{total_agents}: **{exec_agent.get('name')}**", unsafe_allow_html=True)
            except ProviderError as e:
                elapsed = time.time() - start_t
                st.session_state.pipeline_history.append({"agent": exec_agent.get("name"), "input": current_text, "output": None, "error": str(e), "elapsed": elapsed})
                status_area.markdown(f"<span class='status-dot dot-red'></span> Error in agent {idx}/{total_agents}: **{exec_agent.get('name')}**", unsafe_allow_html=True)
                st.error(f"Pipeline stopped due to an error: {e}")
                break
        
        st.session_state.is_running = False

    if st.session_state.pipeline_history:
        st.markdown("--- \n### Pipeline Results")
        for step in st.session_state.pipeline_history:
            dot_color = "dot-green" if step["error"] is None else "dot-red"
            with st.expander(f"<{step['agent']}> - {step['elapsed']:.2f}s", expanded=(step["error"] is not None)):
                st.markdown(f"<span class='status-dot {dot_color}'></span> **Agent:** {step['agent']}", unsafe_allow_html=True)
                st.text_area("Input", value=step["input"] or "", height=150, disabled=True, key=f"in_{step['agent']}_{int(step['elapsed'])}")
                if step["error"]:
                    st.error(step["error"])
                else:
                    st.text_area("Output", value=step["output"] or "", height=150, disabled=True, key=f"out_{step['agent']}_{int(step['elapsed'])}")

with tab6:
    st.subheader("Export Documents")
    if not st.session_state.generated_docs:
        st.info("No documents to export. Please generate documents in the 'Template' tab first.")
    else:
        export_fmt = st.selectbox("Export format", ["txt", "md", "docx", "zip-txt", "zip-md", "zip-docx"])
        
        if "zip" in export_fmt:
            if st.button("Build ZIP Archive", type="primary"):
                ext = export_fmt.split("-")[1]
                with st.spinner(f"Creating .{ext} ZIP archive..."):
                    zbuf = build_zip_from_docs(st.session_state.generated_docs, ext=ext)
                    st.session_state.last_export_zip = zbuf
                st.success("ZIP archive is ready for download.")
            if st.session_state.last_export_zip:
                st.download_button("Download ZIP", data=st.session_state.last_export_zip, file_name=f"docs_export_{int(time.time())}.zip")
        else:
            doc_idx = st.number_input("Select document index to download (1-based)", 1, len(st.session_state.generated_docs), 1)
            doc = st.session_state.generated_docs[doc_idx - 1]
            filename_base = os.path.splitext(doc.get("filename") or f"doc_{doc_idx}")[0]
            
            if export_fmt == "docx":
                buf = export_docx(doc["content"])
                st.download_button("Download DOCX", data=buf, file_name=f"{filename_base}.docx")
            else:
                ext = "md" if export_fmt == "md" else "txt"
                st.download_button(f"Download .{ext.upper()}", data=doc["content"].encode("utf-8"), file_name=f"{filename_base}.{ext}")

st.divider()
st.caption("Powered by Gemini, OpenAI, and Grok. Image understanding available on vision-capable models.")
