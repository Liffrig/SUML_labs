import os
import datetime
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="PL → Esperanto", page_icon="🌍")

st.markdown("""
<style>
.stApp {
    background-color: #1a3a1a;
    color: #e8f5e8;
}
h1, h2, h3, h4, p, label, .stMarkdown {
    color: #e8f5e8 !important;
}
.stTextArea textarea {
    background-color: #2a4a2a;
    color: #e8f5e8;
    border-color: #4a7a4a;}
.stDateInput input {
    background-color: #2a4a2a;
    color: #e8f5e8;
    border-color: #4a7a4a;
}
.stExpanderHeader {
    color: #e8f5e8 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Flaga i tytuł ---
FLAG_PATH = os.path.join(os.path.dirname(__file__), "Screenshot 2026-04-16 at 13-39-43 .png")
col1, col2 = st.columns([1, 4])
with col1:
    st.image(FLAG_PATH, width=120)
with col2:
    st.title("Tłumacz: Polski → Esperanto")
    st.write("Wpisz tekst w języku polskim. Aplikacja przetłumaczy go na **esperanto** w dwóch krokach: najpierw na angielski, a następnie na esperanto.")

st.divider()

# --- Data z opisem w esperanto ---
MIESIAC_EO = {
    1: "januaro", 2: "februaro", 3: "marto", 4: "aprilo",
    5: "majo", 6: "junio", 7: "julio", 8: "aŭgusto",
    9: "septembro", 10: "oktobro", 11: "novembro", 12: "decembro",
}
DZIEN_EO = {
    0: "lundo", 1: "mardo", 2: "merkredo", 3: "ĵaŭdo",
    4: "vendredo", 5: "sabato", 6: "dimanĉo",
}

today = datetime.date.today()
miesiac = MIESIAC_EO[today.month]
dzien = DZIEN_EO[today.weekday()]
st.markdown(f"<div style='text-align:center'><b>{dzien.capitalize()}, {today.day} {miesiac} {today.year}. Bonvenon, uzanto!</b></div>", unsafe_allow_html=True)

st.divider()

# --- Modele tłumaczenia ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _model_path(folder):
    local = os.path.join(MODELS_DIR, folder)
    if os.path.isdir(local):
        return local
    return f"Helsinki-NLP/opus-mt-{folder}"

@st.cache_resource(show_spinner="Wczytuję model PL → EN...")
def load_pl_en():
    path = _model_path("pl-en")
    return MarianTokenizer.from_pretrained(path), MarianMTModel.from_pretrained(path)

@st.cache_resource(show_spinner="Wczytuję model EN → EO...")
def load_en_eo():
    path = _model_path("en-eo")
    return MarianTokenizer.from_pretrained(path), MarianMTModel.from_pretrained(path)

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Wstępne ładowanie modeli ---
load_pl_en()
load_en_eo()

# --- Session state ---
if "form_key" not in st.session_state:
    st.session_state.form_key = 0
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

# --- Interfejs tłumaczenia ---
with st.form(f"translation_form_{st.session_state.form_key}", enter_to_submit=False):
    text = st.text_area(
        "Tekst w języku polskim:",
        placeholder="Wpisz tekst do przetłumaczenia...",
        height=150,
    )
    submitted = st.form_submit_button("Przetłumacz", type="primary")

if submitted and (text or "").strip():
    word_count = len((text or "").split())
    if word_count > 25:
        st.error(f"Zbyt długi tekst ({word_count} słów). Maksymalnie 25 słów na raz.")
    else:
        try:
            with st.spinner("Krok 1/2: polski → angielski..."):
                tok_pl_en, mdl_pl_en = load_pl_en()
                en_text = translate(text, tok_pl_en, mdl_pl_en)

            with st.spinner("Krok 2/2: angielski → esperanto..."):
                tok_en_eo, mdl_en_eo = load_en_eo()
                eo_text = translate(en_text, tok_en_eo, mdl_en_eo)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.history.append({
                "timestamp": timestamp,
                "pl": text.strip(),
                "eo": eo_text.strip(),
            })
            st.session_state.result = {"eo": eo_text, "en": en_text}
            st.session_state.form_key += 1
            st.rerun()

        except Exception as e:
            st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")

if st.session_state.result:
    eo_text = st.session_state.result["eo"]
    en_text = st.session_state.result["en"]

    st.success("Tłumaczenie zakończone!")
    st.subheader("Wynik (Esperanto):")
    st.markdown(f"""
    <div style="
        border: 2px solid #4a7a4a;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        font-size: 20px;
        color: #e8f5e8;
        background-color: #2a4a2a;
        margin: 10px 0 16px 0;
    ">{eo_text}</div>
    """, unsafe_allow_html=True)

    with st.expander("Pokaż pośrednie tłumaczenie (angielski)"):
        st.write(en_text)

st.divider()

# --- Historia tłumaczeń ---
st.subheader("Ostatnie tłumaczenia")
if st.session_state.history:
    for entry in reversed(st.session_state.history[-5:]):
        st.markdown(f"🕐 `{entry['timestamp']}`")
        st.markdown(f"- **PL:** {entry['pl']}")
        st.markdown(f"- **EO:** {entry['eo']}")
        st.markdown("---")
else:
    st.write("Brak tłumaczeń w tej sesji.")

st.divider()
st.markdown("<div style='text-align:center; color:#4a7a4a;'>Autor: s26701</div>", unsafe_allow_html=True)
