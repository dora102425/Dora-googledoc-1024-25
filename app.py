import os
import io
import time
import json
import zipfile
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import yaml

# Local Imports
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
.st-emotion-cache-16txtl3 {{ padding: 2rem 2rem; }} /* Main content padding */
h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-zt3s72, .st-emotion-cache-ue6h4q  {{
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
.st-emotion-cache-1y4p8pa {{ /* Main sidebar background */
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
  model: gemini-2.5-flash
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
        on_change=lambda: st.experimental_rerun()
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
        "None", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "gpt-4o", "gpt-4o-mini", "gpt-4.1-turbo",
        "grok-4-fast-reasoning", "grok-3-mini"
    ]
    provider_choice = st.selectbox("Provider override", ["None", "gemini", "openai", "grok"], index=0)
    model_choice = st.selectbox("Model override", ALL_MODELS, index=0)
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

# ... (The rest of the tabs remain largely the same, but with minor key fixes and robustness checks)
# The full code for tabs would be too long, but here are the key improvements:

with tab4:
    st.subheader("Configure Agent Chain")
    st.caption("Edit agent parameters. Use the sidebar to override providers/models globally for testing.")
    for idx, agent in enumerate(st.session_state.agents):
        # Create a unique key for the expander to prevent state issues
        expander_key = f"agent_expander_{idx}_{agent.get('name', 'Agent')}"
        with st.expander(f"Agent {idx+1} • {agent.get('name', 'Agent')}"):
            c1, c2, c3 = st.columns([1,1,1])
            agent["name"] = c1.text_input("Name", value=agent.get("name","Agent"), key=f"a_name_{idx}")
            agent["provider"] = c2.selectbox("Provider", ["gemini","openai","grok"], index=["gemini","openai","grok"].index(agent.get("provider","gemini")), key=f"a_provider_{idx}")
            
            # Safe index finding for model
            current_model = agent.get("model","gemini-2.5-flash")
            try:
                model_index = ALL_MODELS.index(current_model)
            except ValueError:
                model_index = 0 # Default to 'None' if model not in list
            
            agent["model"] = c3.selectbox( "Model", ALL_MODELS, index=model_index, key=f"a_model_{idx}" )

            agent["temperature"] = st.slider("Temperature", 0.0, 1.0, float(agent.get("temperature",0.5)), 0.05, key=f"a_temp_{idx}")
            agent["max_tokens"] = st.number_input("Max Tokens", 128, 16384, int(agent.get("max_tokens",2048)), 64, key=f"a_maxtok_{idx}")

            agent["system_prompt"] = st.text_area("System Prompt", value=agent.get("system_prompt",""), height=120, key=f"a_sysp_{idx}")
            agent["user_prompt"] = st.text_area("User Prompt (use {{input}} token)", value=agent.get("user_prompt",""), height=180, key=f"a_usrp_{idx}")

    st.markdown("---")
    colBA1, colBA2, colBA3 = st.columns([1, 1, 2])
    if colBA1.button("➕ Add New Agent"):
        st.session_state.agents.append({
            "name": f"New Agent {len(st.session_state.agents) + 1}",
            "provider": "gemini", "model": "gemini-2.5-flash",
            "temperature": 0.5, "max_tokens": 2048,
            "system_prompt": "You are a helpful assistant.", "user_prompt": "{{input}}"
        })
        st.experimental_rerun()
    if colBA2.button("🔄 Reset to Default"):
        st.session_state.agents = DEFAULT_AGENTS.copy()
        st.experimental_rerun()

with tab5:
    st.subheader("Execute Pipeline")
    st.caption("Run a selected input through the agent chain. You can pull from generated docs or paste custom input.")

    if st.session_state.images:
        # Bug fix: Check if Grok is in the pipeline and warn about image incompatibility
        uses_grok = any(agent.get("provider") == "grok" for agent in st.session_state.agents)
        if uses_grok:
            st.warning("🖼️ Images have been uploaded, but the **Grok** provider currently does not support image input. Images will be ignored for Grok agents.")

    default_input = st.session_state.generated_docs[0]["content"] if st.session_state.generated_docs else ""
    input_text = st.text_area("Pipeline Input", value=default_input, height=200, key="pipeline_input_text")
    
    # ... (rest of the pipeline execution logic is the same and robust)


# --- The rest of the app.py file (tabs 1, 2, 3, 5, 6) can remain as is, ---
# --- as the provided code for those sections is functional.           ---
# --- Key changes are UI (themes), state management, and agent config. ---
# --- For brevity, I'm omitting the unchanged tab code.                ---

# Placeholder for unchanged tabs
with tab1:
    st.subheader("Upload dataset")
    # ... code from original ...
with tab2:
    st.subheader("Define Template")
    # ... code from original ...
with tab3:
    st.subheader("Review and Edit Generated Documents")
    # ... code from original ...
with tab6:
    st.subheader("Export Documents")
    # ... code from original ...
