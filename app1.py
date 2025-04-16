# POS Taggers for Indian Languages - Enhanced with Transformer fallback

# Requirements:
# pip install nltk indic-nlp-library streamlit transformers torch langdetect sklearn_crfsuite

import streamlit as st
from langdetect import detect
import nltk
from nltk.corpus import indian
from nltk.tag import UnigramTagger, DefaultTagger
from nltk.tokenize import wordpunct_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Download necessary NLTK resources
nltk_resources = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/indian', 'indian'),
]
for path, name in nltk_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

# Supported NLTK languages
nltk_languages = {
    'hi': 'hindi.pos',
    'mr': 'marathi.pos',
    'gu': 'gujarati.pos'
}

# HuggingFace multilingual model
model_checkpoint = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
nlp_pos = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# NLTK POS tagging with backoff
def pos_nltk(text, lang_code):
    try:
        tagged_sents = indian.tagged_sents(nltk_languages[lang_code])
        default_tagger = DefaultTagger('NN')
        unigram_tagger = UnigramTagger(tagged_sents, backoff=default_tagger)
        tokens = wordpunct_tokenize(text)
        return unigram_tagger.tag(tokens)
    except Exception as e:
        return None

# Transformer tagging
def pos_transformer(text):
    try:
        entities = nlp_pos(text)
        return [(entity['word'], entity['entity_group']) for entity in entities]
    except:
        return []

# Supported languages for UI
lang_labels = {
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'ta': 'Tamil',
    'te': 'Telugu'
}

# Streamlit UI
st.set_page_config(page_title="POS Tagger for Indian Languages", layout="centered")
st.title("🇮🇳 POS Tagger for Indian Languages")
st.caption("Supports Hindi, Marathi, Gujarati (NLTK) + Tamil, Telugu & others (Transformer Model)")

text_input = st.text_area("Enter a sentence in any Indian language:", "मैं स्कूल जा रहा हूँ।")

if st.button("Tag POS"):
    if not text_input.strip():
        st.warning("Please enter a valid sentence.")
    else:
        lang = detect(text_input)
        lang_name = lang_labels.get(lang, lang.upper())
        st.write(f"Detected Language: `{lang_name}`")

        if lang in nltk_languages:
            st.subheader(f"POS Tags using NLTK ({lang_name})")
            tags = pos_nltk(text_input, lang)
            if tags:
                st.write(tags)
            else:
                st.warning("NLTK tagging failed. Falling back to Transformer model.")
                tags = pos_transformer(text_input)
                st.write(tags)
        else:
            st.subheader("POS Tags using Transformer Model")
            tags = pos_transformer(text_input)
            if tags:
                st.write(tags)
            else:
                st.error("POS tagging failed. Please check your sentence.")

st.markdown("---")
st.caption("All Rights Reserved Siddhesh · Hindi, Marathi, Gujarati via NLTK · Tamil, Telugu & others via Transformer")
