import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Preditor de Sobrevivência do Titanic",
    layout="wide"
)

st.title("Preditor de Sobrevivência do Titanic")
st.markdown("""
Esta aplicação utiliza modelos de Machine Learning para prever a probabilidade de sobrevivência 
de um passageiro do Titanic com base em suas características.
""")

@st.cache_resource
def carregar_modelos():
    try:
        with open('logistic_regression_titanic.pickle', 'rb') as f:
            log_model = pickle.load(f)
        
        with open('random_forest_titanic.pickle', 'rb') as f:
            rf_model = pickle.load(f)
            
        with open('svm_titanic.pickle', 'rb') as f:
            svm_model = pickle.load(f)
            
        return {
            'Regressão Logística': log_model,
            'Random Forest': rf_model,
            'SVM': svm_model
        }
    except Exception as e:
        st.error(f"Erro ao carregar os modelos: {e}")
        return None

modelos = carregar_modelos()

st.sidebar.header("Configurações")
modelo_selecionado = st.sidebar.selectbox(
    "Selecione o modelo",
    list(modelos.keys()) if modelos else []
)

st.header("Dados do Passageiro")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe", [1, 2, 3], help="1 = 1ª Classe, 2 = 2ª Classe, 3 = 3ª Classe")
    sex = st.radio("Sexo", ["Masculino", "Feminino"])
    age = st.slider("Idade", 0, 80, 30)
    fare = st.number_input("Tarifa (£)", min_value=0.0, max_value=300.0, value=20.0, step=1.0)

with col2:
    sibsp = st.slider("Número de Irmãos/Cônjuges a bordo", 0, 8, 0)
    parch = st.slider("Número de Pais/Filhos a bordo", 0, 6, 0)
    embarked = st.selectbox("Porto de Embarque", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    title = st.selectbox("Título", ["Mr", "Mrs", "Miss", "Master", "Other"])

def processar_dados(pclass, sex, age, fare, sibsp, parch, embarked, title):
    sex_numeric = 1 if sex == "Feminino" else 0
    
    family_size = sibsp + parch + 1
    
    is_alone = 1 if family_size == 1 else 0
    
    embarked_c = 1 if embarked == "Cherbourg (C)" else 0
    embarked_q = 1 if embarked == "Queenstown (Q)" else 0
    embarked_s = 1 if embarked == "Southampton (S)" else 0
    
    title_master = 1 if title == "Master" else 0
    title_miss = 1 if title == "Miss" else 0
    title_mr = 1 if title == "Mr" else 0
    title_mrs = 1 if title == "Mrs" else 0
    title_other = 1 if title == "Other" else 0
    
    data = {
        'Pclass': pclass,
        'Sex': sex_numeric,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'Embarked_C': embarked_c,
        'Embarked_Q': embarked_q,
        'Embarked_S': embarked_s,
        'Title_Master': title_master,
        'Title_Miss': title_miss,
        'Title_Mr': title_mr,
        'Title_Mrs': title_mrs,
        'Title_Other': title_other
    }
    
    return pd.DataFrame([data])

if st.button("Prever Sobrevivência"):
    if modelos:
        dados_processados = processar_dados(pclass, sex, age, fare, sibsp, parch, embarked, title)
        
        modelo = modelos[modelo_selecionado]
        
        if modelo_selecionado in ["SVM", "Regressão Logística"]:

            probabilidade = modelo.predict_proba(dados_processados)[0][1]
        else:
            probabilidade = modelo.predict_proba(dados_processados)[0][1]
        
        st.header("Resultado da Previsão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            colors = ['#ff9999', '#66b3ff'] if probabilidade < 0.5 else ['#66b3ff', '#99ff99']
            ax.pie([1-probabilidade, probabilidade], colors=colors, startangle=90, 
                   wedgeprops=dict(width=0.3))
            ax.add_artist(plt.Circle((0, 0), 0.3, fc='white'))
            ax.text(0, 0, f"{probabilidade:.1%}", ha='center', va='center', fontsize=24)
            
            plt.axis('equal')
            st.pyplot(fig)
        
        with col2:
            if probabilidade >= 0.5:
                st.success(f"Probabilidade de sobrevivência: {probabilidade:.1%}")
                st.markdown("""
                **Interpretação**: Este passageiro teria uma boa chance de sobreviver ao naufrágio do Titanic.
                """)
            else:
                st.error(f"Probabilidade de sobrevivência: {probabilidade:.1%}")
                st.markdown("""
                **Interpretação**: Este passageiro teria uma baixa chance de sobreviver ao naufrágio do Titanic.
                """)
        
        st.subheader("Fatores que influenciam a sobrevivência:")
        
        if sex == "Feminino":
            st.info("Ser mulher aumentava significativamente as chances de sobrevivência (política 'mulheres e crianças primeiro')")
        else:
            st.warning("Homens tinham menor probabilidade de sobrevivência")
            
        if pclass == 1:
            st.info("Passageiros da 1ª classe tinham acesso prioritário aos botes salva-vidas")
        elif pclass == 3:
            st.warning("Passageiros da 3ª classe enfrentaram mais dificuldades para acessar os botes")
            
        if age < 10:
            st.info("Crianças tinham prioridade nos botes salva-vidas")
        

st.sidebar.header("Download dos Modelos")

if modelos:
    modelo_para_download = st.sidebar.selectbox(
        "Selecione o modelo para download",
        list(modelos.keys())
    )
    
    if st.sidebar.button("Download do Modelo"):
        modelo = modelos[modelo_para_download]
        st.sidebar.download_button(
            label="Baixar Modelo",
            data=pickle.dumps(modelo),
            file_name=f"{modelo_para_download.lower().replace(' ', '_')}_titanic.pkl",
            mime="application/octet-stream"
        )

st.sidebar.header("Sobre os Modelos")
st.sidebar.markdown("""
- **Regressão Logística**: Modelo linear simples e interpretável
- **Random Forest**: Conjunto de árvores de decisão, geralmente com melhor desempenho
- **SVM**: Support Vector Machine com kernel RBF, bom para dados complexos
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido para análise do dataset do Titanic")
