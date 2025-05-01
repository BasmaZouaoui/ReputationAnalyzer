import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Analyse de Réputation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5rem;
        margin-top: 1rem;
        color: #424242;
        font-weight: 600;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadein 0.5s;
    }
    .positive {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
    }
    .negative {
        background-color: rgba(244, 67, 54, 0.2);
        border-left: 5px solid #F44336;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        padding: 10px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.8rem;
    }
    .examples-box {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .example-btn {
        margin: 0.2rem;
        padding: 0.3rem 0.8rem;
        background-color: #E0E0E0;
        border-radius: 15px;
        font-size: 0.8rem;
        cursor: pointer;
        border: none;
    }
    .example-btn:hover {
        background-color: #BDBDBD;
    }
    .sidebar-title {
        font-weight: 600;
        color: #1E88E5;
    }
    .example-label {
        font-size: 0.7rem;
        color: #757575;
        margin-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("basmazouaoui/ReputationAnalyzer_BERT", device_map=None)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model, device

if 'history' not in st.session_state:
    st.session_state.history = []

tokenizer, model, device = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][pred].item()
        
    sentiment = "positif" if pred == 1 else "négatif"
    return sentiment, confidence

with st.sidebar:

    st.markdown("<div class='sidebar-title'>À propos</div>", unsafe_allow_html=True)
    st.info("""
    Cette application utilise un modèle BERT fine-tuné pour analyser le sentiment des avis clients.
    
    Le modèle classifie les textes en deux catégories:
    - 🟢 **Positif**: Opinion favorable
    - 🔴 **Négatif**: Opinion défavorable
    """)
    
    st.markdown("---")
    
    if st.session_state.history:
        st.markdown("<div class='sidebar-title'>Historique récent</div>", unsafe_allow_html=True)
        for i, (text, sentiment, conf, timestamp) in enumerate(st.session_state.history[-5:]):
            sentiment_emoji = "🟢" if sentiment == "positif" else "🔴"
            st.markdown(f"**{timestamp}**  \n{sentiment_emoji} {text[:20]}..." + ("" if len(text) <= 20 else "..."))
        
        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.experimental_rerun()

st.markdown("<h1 class='main-header'>Analyse de la Réputation des Entreprises</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='subheader'>Entrez un avis client</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='examples-box'>", unsafe_allow_html=True)
    st.markdown("**Exemples d'avis:**")
    
    example_texts = [
        "Amazing product! Worth every penny.",
        "Terrible customer service. Would not recommend.",
        "I absolutely love this product! I've been using it for three months now and the quality is outstanding. The customer service was excellent when I had questions. Will definitely purchase again and recommend to all my friends and family.",
        "This was a complete waste of money. The product arrived damaged and didn't work as advertised. When I tried to contact support, nobody responded for days. Finally got a response but they refused to issue a refund. Extremely disappointed with my purchase experience."
    ]
    
    for ex in example_texts:
        if st.button(ex[:30] + "...", key=f"ex_{ex[:10]}"):
            st.session_state.text_input = ex
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    text = st.text_area("", key="text_input", height=150, placeholder="Tapez ou collez votre texte ici...")
    
    analyze_btn = st.button("Analyser", use_container_width=True)

with col2:
    st.markdown("<div class='subheader'>Statistiques</div>", unsafe_allow_html=True)
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Analyses", len(st.session_state.history))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with metrics_col2:
        positive_count = sum(1 for _, s, _, _ in st.session_state.history if s == "positif")
        if st.session_state.history:
            positive_rate = positive_count / len(st.session_state.history) * 100
        else:
            positive_rate = 0
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Taux positif", f"{positive_rate:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.history:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        
        sentiments = [s for _, s, _, _ in st.session_state.history]
        sentiment_counts = {"positif": sentiments.count("positif"), "négatif": sentiments.count("négatif")}
        
        fig = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            color=list(sentiment_counts.keys()),
            color_discrete_map={"positif": "#4CAF50", "négatif": "#F44336"},
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn and text:
    with st.spinner("Analyse en cours..."):
        time.sleep(0.5)  
        sentiment, confidence = predict_sentiment(text)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.history.append((text, sentiment, confidence, timestamp))
    
    result_class = "positive" if sentiment == "positif" else "negative"
    emoji = "🟢" if sentiment == "positif" else "🔴"
    
    st.markdown(f"""
    <div class='result-box {result_class}'>
        <h3>{emoji} Sentiment {sentiment.upper()}</h3>
        <p>Confiance: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
        
    col_analysis1, col_analysis2 = st.columns(2)
    
    st.progress(confidence)

st.markdown("<div class='footer'>© 2025 Analyse de Réputation - Créé avec <span class='heart'>❤</span> par Basma</div>", unsafe_allow_html=True)