import streamlit as st
import speech_recognition as sr
import nltk
import io
import re
import string
from typing import Tuple, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(
    page_title="Voice-Enabled Chatbot",
    page_icon="🎤",
    layout="centered"
)

# =========================================================
# NLTK bootstrap
# =========================================================
def ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

@st.cache_resource
def load_nltk():
    ensure_nltk_data()

load_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

# =========================================================
# NLP utilities
# =========================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punct_table = str.maketrans("", "", string.punctuation)

def normalize_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text)

def nltk_tokenizer(doc: str) -> List[str]:
    doc = doc.lower().translate(punct_table)
    tokens = word_tokenize(doc)
    return [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]

def build_index(
    corpus_text: str
) -> Tuple[List[str], Optional[TfidfVectorizer], Optional[object]]:
    corpus_text = normalize_text(corpus_text)
    if not corpus_text:
        return [], None, None

    sentences = sent_tokenize(corpus_text)
    vectorizer = TfidfVectorizer(
        tokenizer=nltk_tokenizer,
        token_pattern=None
    )
    X = vectorizer.fit_transform(sentences)
    return sentences, vectorizer, X

# =========================================================
# Chatbot logic
# =========================================================
def is_greeting(text: str) -> bool:
    greetings = {"hi", "hello", "hey", "morning", "afternoon", "evening"}
    return bool(set(nltk_tokenizer(text)) & greetings)

def smalltalk(text: str) -> Optional[str]:
    t = text.lower()
    if "your name" in t:
        return "I'm your Streamlit chatbot 😊"
    if "how are you" in t:
        return "I'm doing great! How can I help?"
    if "thank" in t:
        return "You're welcome!"
    return None

def retrieve_response(
    query: str,
    sentences: List[str],
    vectorizer: Optional[TfidfVectorizer],
    X
) -> str:
    if not sentences or vectorizer is None or X is None:
        return "Please upload or paste some text to build my knowledge base."

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()

    best_idx = sims.argmax()
    if sims[best_idx] < 0.1:
        return "I'm not sure about that. Could you rephrase?"

    return sentences[best_idx]

def chatbot_reply(
    text: str,
    sentences,
    vectorizer,
    X
) -> str:
    if not text.strip():
        return "Please type or say something."

    if is_greeting(text):
        return "Hello! Ask me something about the uploaded text."

    sm = smalltalk(text)
    if sm:
        return sm

    return retrieve_response(text, sentences, vectorizer, X)

# =========================================================
# Speech recognition via uploaded audio
# =========================================================
def transcribe_uploaded_audio(uploaded_file, language="en-US"):
    recognizer = sr.Recognizer()
    try:
        audio_bytes = uploaded_file.read()
        audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
        with audio_file as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return True, text
    except Exception as e:
        return False, f"Voice error: {e}"

# =========================================================
# Session state
# =========================================================
if "sentences" not in st.session_state:
    st.session_state.sentences = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "X" not in st.session_state:
    st.session_state.X = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

# =========================================================
# UI
# =========================================================
st.title("🎤 Voice-Enabled Chatbot")
st.write("Upload a text file or paste text to create a knowledge base.")

# ---------------- Sidebar ----------------
st.sidebar.header("Knowledge Base")

uploaded = st.sidebar.file_uploader("Upload .txt file", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8", errors="replace")
    st.session_state.sentences, st.session_state.vectorizer, st.session_state.X = build_index(text)
    st.sidebar.success("Knowledge base updated!")

kb_text = st.sidebar.text_area("Or paste text here", height=200)

if st.sidebar.button("Build / Update Bot"):
    st.session_state.sentences, st.session_state.vectorizer, st.session_state.X = build_index(kb_text)
    st.sidebar.success("Bot updated!")

st.sidebar.header("Voice Settings")
language = st.sidebar.selectbox("Language", ["en-US", "en-GB", "fr-FR"])

# ---------------- Chat ----------------
st.subheader("Conversation")

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

mode = st.radio(
    "Input method",
    ["Text", "Voice"],
    horizontal=True
)

# ---------------- Text ----------------
if mode == "Text":
    user_input = st.chat_input("Type a message")
    if user_input:
        st.session_state.chat.append(("user", user_input))
        reply = chatbot_reply(
            user_input,
            st.session_state.sentences,
            st.session_state.vectorizer,
            st.session_state.X
        )
        st.session_state.chat.append(("assistant", reply))
        st.rerun()

# ---------------- Voice ----------------
else:
  uploaded_audio = st.file_uploader(
    "Upload an audio file (.wav recommended)",
    type=["wav"]
)
    if uploaded_audio:
        ok, result = transcribe_uploaded_audio(uploaded_audio, language=language)
        if ok:
            st.session_state.chat.append(("user", result))
            reply = chatbot_reply(
                result,
                st.session_state.sentences,
                st.session_state.vectorizer,
                st.session_state.X
            )
            st.session_state.chat.append(("assistant", reply))
            st.rerun()
        else:
            st.error(result)

# ---------------- Utilities ----------------
st.divider()

col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.rerun()

with col2:
    if st.button("Export Chat"):
        buf = io.StringIO()
        for role, msg in st.session_state.chat:
            buf.write(f"{role}: {msg}\n")
        st.download_button(
            "Download chat.txt",
            buf.getvalue(),
            "chat.txt",
            "text/plain"
        )
