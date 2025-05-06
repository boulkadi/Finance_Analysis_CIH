"""
# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Config de page
st.set_page_config(page_title="Visualisation Risques Financiers", layout="wide")
st.title("📊 Visualisation Interactive des Risques Financiers Pondérés (2013–2024)")

# Chargement des données
df = pd.read_csv("output/output_result1.csv")

# Affichage des données
st.subheader("🧾 Données Brutes")
st.dataframe(df)

# Sidebar : choix des indicateurs et filtre par année
st.sidebar.header("🎛️ Options")
indicateurs = df.columns[1:]
selected_cols = st.sidebar.multiselect("Indicateurs à afficher :", indicateurs, default=indicateurs)

min_year, max_year = int(df["Année"].min()), int(df["Année"].max())
year_range = st.sidebar.slider("Plage d'années :", min_year, max_year, (min_year, max_year))
df_filtered = df[(df["Année"] >= year_range[0]) & (df["Année"] <= year_range[1])]

# Graphique 1 : Courbe interactive
st.subheader("📈 Évolution Temporelle ")
if selected_cols:
    fig_line = px.line(df_filtered, x="Année", y=selected_cols,
                       labels={"value": "Montant (MAD)", "variable": "Indicateur"},
                       title="Évolution des risques financiers")
    fig_line.update_traces(mode='lines+markers')
    st.plotly_chart(fig_line, use_container_width=True)

# Graphique 2 : Barres interactives
st.subheader("📊 Comparaison par Année ")
if selected_cols:
    df_melt = df_filtered.melt(id_vars="Année", value_vars=selected_cols, var_name="Indicateur", value_name="Valeur")
    fig_bar = px.bar(df_melt, x="Année", y="Valeur", color="Indicateur", barmode="group")
    st.plotly_chart(fig_bar, use_container_width=True)

# Graphique 3 : Aires empilées interactives
st.subheader("📐 Aires empilées")
if selected_cols:
    fig_area = go.Figure()
    for col in selected_cols:
        fig_area.add_trace(go.Scatter(x=df_filtered["Année"], y=df_filtered[col],
                                      stackgroup='one', name=col, mode='lines'))
    fig_area.update_layout(title="Répartition des Risques par Année", yaxis_title="Montant (MAD)")
    st.plotly_chart(fig_area, use_container_width=True)

# Graphique 4 : Camembert pour une année
st.subheader(" Camembert Interactif pour une Année")
year_selected = st.selectbox("Choisis une année :", df_filtered["Année"])
row = df[df["Année"] == year_selected][selected_cols].iloc[0]
fig_pie = px.pie(values=row.values, names=selected_cols,
                 title=f"Répartition des Risques en {year_selected}",
                 hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Pour les embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")

# Configurer Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Fonction pour extraire le texte d'un fichier PDF
def load_extracted_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Fonction de création de la vector store avec LangChain
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    return FAISS.from_texts(texts, embeddings)

output_text_path = "output/output_text1.txt"
output_text = load_extracted_text(output_text_path)

# Étape 2 : Découper le texte en morceaux (chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=9500,  # Taille maximale des morceaux en caractères
    chunk_overlap=100,  # Superposition entre les morceaux pour maintenir le contexte
    is_separator_regex=False
)

text_chunks = text_splitter.split_text(output_text)

# Étape 3 : Créer un vector store avec les embeddings
vector_store = create_vector_store(text_chunks)

# Fonction pour traiter une question et retourner la réponse
def process_question(question, vector_store):
    prompt_template = """
        Use the following pieces of context to answer the question at the end.

        Check context very carefully and reference and try to make sense of that before responding.
        If you don't know the answer, just say you don't know.
        Don't try to make up an answer.
        Answer must be to the point.
        Think step-by-step.
        do not introduct the answer, only the answer.

        Context: {context}

        Question: {question}

        Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combines all retrieved docs into single context
        retriever=vector_store.as_retriever(search_kwargs={"k": 100}),  # Retrieve top  relevant chunks
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,  # Include source documents in response
    )
    response = qa_chain.invoke({"query": question})
    return response['result']

# Config de page
st.set_page_config(page_title="Visualisation Risques Financiers", layout="wide")
st.title("📊 Visualisation Interactive des Risques Financiers Pondérés (2013–2024)")

# Chargement des données
df = pd.read_csv("output/output_result1.csv")

# Affichage des données
st.subheader("🧾 Données Brutes")
st.dataframe(df)

# Sidebar : choix des indicateurs et filtre par année
st.sidebar.header("🎛️ Options")
indicateurs = df.columns[1:]
selected_cols = st.sidebar.multiselect("Indicateurs à afficher :", indicateurs, default=indicateurs)

min_year, max_year = int(df["Année"].min()), int(df["Année"].max())
year_range = st.sidebar.slider("Plage d'années :", min_year, max_year, (min_year, max_year))
df_filtered = df[(df["Année"] >= year_range[0]) & (df["Année"] <= year_range[1])]

# Graphique 1 : Courbe interactive
st.subheader("📈 Évolution Temporelle ")
if selected_cols:
    fig_line = px.line(df_filtered, x="Année", y=selected_cols,
                       labels={"value": "Montant (MAD)", "variable": "Indicateur"},
                       title="Évolution des risques financiers")
    fig_line.update_traces(mode='lines+markers')
    st.plotly_chart(fig_line, use_container_width=True)

# Graphique 2 : Barres interactives
st.subheader("📊 Comparaison par Année ")
if selected_cols:
    df_melt = df_filtered.melt(id_vars="Année", value_vars=selected_cols, var_name="Indicateur", value_name="Valeur")
    fig_bar = px.bar(df_melt, x="Année", y="Valeur", color="Indicateur", barmode="group")
    st.plotly_chart(fig_bar, use_container_width=True)

# Graphique 3 : Aires empilées interactives
st.subheader("📐 Aires empilées")
if selected_cols:
    fig_area = go.Figure()
    for col in selected_cols:
        fig_area.add_trace(go.Scatter(x=df_filtered["Année"], y=df_filtered[col],
                                      stackgroup='one', name=col, mode='lines'))
    fig_area.update_layout(title="Répartition des Risques par Année", yaxis_title="Montant (MAD)")
    st.plotly_chart(fig_area, use_container_width=True)

# Graphique 4 : Camembert pour une année
st.subheader("🍰 Camembert Interactif pour une Année")
year_selected = st.selectbox("Choisis une année :", df_filtered["Année"])
row = df[df["Année"] == year_selected][selected_cols].iloc[0]
fig_pie = px.pie(values=row.values, names=selected_cols,
                 title=f"Répartition des Risques en {year_selected}",
                 hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)

# Zone de saisie pour poser des questions à Gemini
st.subheader("❓ Pose ta question à Gemini concernant les données")
question = st.text_input("Posez votre question ici :")

if question:
    # Traitement de la question
    with st.spinner("En attente de la réponse..."):
        response = process_question(question, vector_store=vector_store)
    # Affichage de la réponse
    st.write(f"Réponse : {response}")
