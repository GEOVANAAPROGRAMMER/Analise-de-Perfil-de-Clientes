from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
import re

app = Flask(__name__)

# Carregar dados e modelos
url_perfil = 'datafreme/Perfil de Clientes.xlsx'
url_ofertas = 'datafreme/Ofertas.xlsx'
perfil_df = pd.read_excel(url_perfil)
ofertas_df = pd.read_excel(url_ofertas)

numeric_features = ['Idade', 'Finalidade Profissional', 'Finalidade Acadêmica', 'Finalidade Entreterimento']
categorical_features = ['Faixa idade', 'Gênero', 'Estado onde mora', 'UF onde mora', 'Região onde mora',
                        'Nível de Ensino', 'Área de Formação', 'Situação atual de trabalho', 'Setor', 'Forma de trabalho', 'Faixa salarial']

def preparar_modelo():
    X = perfil_df.drop(columns=['Comprou'])
    y = perfil_df['Comprou']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Treinando o modelo de KNN para recomendação
    model_knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    X_preprocessado = clf.named_steps['preprocessor'].transform(X)
    model_knn.fit(X_preprocessado)

    return clf, model_knn

modelo, modelo_knn = preparar_modelo()

def extrair_valores(faixa_salarial):
    faixa_salarial = faixa_salarial.lower().strip()

    # Substituir ',' por '.' para lidar com decimais corretamente
    faixa_salarial = faixa_salarial.replace(',', '.')

    if faixa_salarial.startswith('menos de'):
        maximo = float(re.sub(r'[^\d.]', '', faixa_salarial))  # Remover caracteres não numéricos
        return 0, maximo
    elif faixa_salarial.startswith('acima de'):
        minimo = float(re.sub(r'[^\d.]', '', faixa_salarial))
        return minimo, float('inf')
    elif faixa_salarial.startswith('de') and 'a' in faixa_salarial:
        partes = re.findall(r'\d+\.\d+', faixa_salarial)
        if len(partes) == 2:
            minimo = float(partes[0])
            maximo = float(partes[1])
            return minimo, maximo
        else:
            raise ValueError(f"Formato desconhecido de faixa salarial: {faixa_salarial}")
    elif faixa_salarial.startswith('até'):
        maximo = float(re.sub(r'[^\d.]', '', faixa_salarial))
        return 0, maximo
    else:
        raise ValueError(f"Formato desconhecido de faixa salarial: {faixa_salarial}")

def recomendar_ofertas(faixa_salarial_cliente, ofertas_df):
    try:
        minimo_cliente, maximo_cliente = extrair_valores(faixa_salarial_cliente)
    except ValueError as e:
        return str(e)
    
    ofertas_recomendadas = []

    for _, row in ofertas_df.iterrows():
        try:
            minimo_oferta, maximo_oferta = extrair_valores(row['Faixa Salarial'])
            if minimo_oferta is not None and maximo_oferta is not None:
                if minimo_cliente >= minimo_oferta and (maximo_cliente <= maximo_oferta or maximo_oferta == float('inf')):
                    ofertas_recomendadas.append(row)
        except ValueError:
            continue

    if not ofertas_recomendadas:
        return "Nenhuma oferta encontrada para a faixa salarial especificada."

    return pd.DataFrame(ofertas_recomendadas)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Verifique se 'perfil_df' está corretamente carregado
    if perfil_df is None or perfil_df.empty:
        return "Erro: Dados do perfil não carregados."

    # Gráfico 1: Top 5 Características que Influenciam a Compra
    try:
        feature_names = numeric_features + list(modelo.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
        importances = modelo.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        top_features = feature_importances.head(5)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax.set_xlabel('Importância')
        ax.set_ylabel('Características')
        ax.set_title('Top 5 Características que Influenciam a Compra')
        plt.gca().invert_yaxis()
        top_features_url = plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        return f"Erro ao criar gráfico de características: {e}"

    # Gráfico 2: Distribuição de Gênero
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        gender_counts = perfil_df['Gênero'].value_counts()
        ax.bar(gender_counts.index, gender_counts.values, color='green')
        ax.set_title('Distribuição de Gênero')
        ax.set_xlabel('Gênero')
        ax.set_ylabel('Quantidade')
        plt.xticks(rotation=45, ha='right')
        gender_distribution_url = plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        return f"Erro ao criar gráfico de distribuição de gênero: {e}"

    # Gráfico 3: Número de Compras por Estado
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        compras_por_estado = perfil_df[perfil_df['Comprou'] == 1]['Estado onde mora'].value_counts()
        ax.plot(compras_por_estado.index, compras_por_estado.values, marker='o', color='red', linestyle='-')
        ax.set_title('Número de Compras por Estado')
        ax.set_xlabel('Estado')
        ax.set_ylabel('Número de Compras')
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        compras_por_estado_url = plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        return f"Erro ao criar gráfico de compras por estado: {e}"

    # Gráfico 4: Distribuição de Finalidades
    try:
        finalidade_profissional = perfil_df['Finalidade Profissional'].sum()
        finalidade_academica = perfil_df['Finalidade Acadêmica'].sum()
        finalidade_entretenimento = perfil_df['Finalidade Entreterimento'].sum()
        values = [finalidade_profissional, finalidade_academica, finalidade_entretenimento]
        labels = ['Finalidade Profissional', 'Finalidade Acadêmica', 'Finalidade Entretenimento']

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title('Distribuição de Finalidades')
        finalidades_url = plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        return f"Erro ao criar gráfico de distribuição de finalidades: {e}"

    return render_template('dashboard.html',
                           top_features_url=top_features_url,
                           gender_distribution_url=gender_distribution_url,
                           compras_por_estado_url=compras_por_estado_url,
                           finalidades_url=finalidades_url)

@app.route('/add_client', methods=['GET', 'POST'])
def add_client():
    if request.method == 'POST':
        idade = int(request.form.get('age'))
        genero = request.form.get('gender')
        estado = request.form.get('state')
        finalidade_profissional = int(request.form.get('purpose_professional', 0))
        finalidade_academica = int(request.form.get('purpose_academic', 0))
        finalidade_entretenimento = int(request.form.get('purpose_entertainment', 0))
        faixa_salarial = request.form.get('salary_range')

        # Criar um DataFrame com as informações do cliente
        client_data = pd.DataFrame([{
            'Idade': idade,
            'Finalidade Profissional': finalidade_profissional,
            'Finalidade Acadêmica': finalidade_academica,
            'Finalidade Entreterimento': finalidade_entretenimento,
            'Faixa idade': '',  # Necessário para compatibilidade com o modelo
            'Gênero': genero,
            'Estado onde mora': estado,
            'UF onde mora': '',  # Pode ser adicionado conforme necessário
            'Região onde mora': '',  # Pode ser adicionado conforme necessário
            'Nível de Ensino': '',  # Adicione valores reais conforme necessário
            'Área de Formação': '',  # Adicione valores reais conforme necessário
            'Situação atual de trabalho': '',  # Adicione valores reais conforme necessário
            'Setor': '',  # Adicione valores reais conforme necessário
            'Forma de trabalho': '',  # Adicione valores reais conforme necessário
            'Faixa salarial': faixa_salarial
        }])

        # Preprocessar o novo cliente
        cliente_preprocessado = modelo.named_steps['preprocessor'].transform(client_data)

        # Prever a probabilidade de compra
        probabilidade_compra = modelo.named_steps['classifier'].predict_proba(cliente_preprocessado)[0][1]

        # Recomendação de ofertas com base na faixa salarial
        ofertas_recomendadas = recomendar_ofertas(faixa_salarial, ofertas_df)

        if isinstance(ofertas_recomendadas, pd.DataFrame):
            ofertas_html = ofertas_recomendadas.to_html()
        else:
            ofertas_html = ofertas_recomendadas  # Mensagem de erro se não for um DataFrame

        return render_template('recommendation.html', probabilidade_compra=probabilidade_compra * 100, ofertas=ofertas_html)
    
    return render_template('add_client.html')


if __name__ == '__main__':
    app.run(debug=True)
