#main.py

#IMPORTS NECESSÁRIOS
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel #Contrato de dados de entrada.
import pandas as pd
import uvicorn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict #Tipos para validação e clareza.


#CONTRATO DE DADOS (PYDANTIC)

#Modelo de entrada (Body Payload do POST)
class DadosNovoAluno(BaseModel):
    """
    RECEBE O INPUT: Estrutura dos hábitos de um novo aluno para a análise.
    O usuário insere estes valores no Body (corpo) da requisição POST.
    """
    horas_estudo: float = 5.0 #Exemplo: 5.0
    horas_sono: float = 7.5 #Exemplo: 7.5
    presenca_percentual: float = 85.0 #Exemplo: 85.0 (Valor percentual)
    nota_anterior: int = 70 #Exemplo: 70
    risco_estresse_simulado: Optional[int] = 3 #Nível de estresse de 1 (baixo) a 5 (alto). Exemplo: 3


#MODELOS DE RESPOSTA (OUTPUT)
#Modelos para o JSON de saída, garantindo que o usuário entenda o resultado.

class AnaliseSugestoes(BaseModel):
    #Estrutura interna das sugestões.
    total_pontos_de_melhoria: int
    sugestoes_personalizadas: List[str] 

class RespostaModelo(BaseModel):
    #Estrutura do resultado da previsão.
    nota_prevista_simulada: float
    analise_de_habitos_e_risco: AnaliseSugestoes
    aviso_modelo_ml: str


#SETUP INICIAL DO ML

#Caminho do nosso dataset principal.
CAMINHO_DADOS_NOTAS = "dados/student_exam_scores.csv" 

#Variáveis globais para armazenar o estado do ML.
DADOS_HISTORICOS = []
TOTAL_REGISTROS = 0
MODELO_REGRESSAO = None
PRE_PROCESSADOR = None


def carregar_e_treinar_modelo():
    #Função de setup: carrega dados, faz Engenharia de Features e treina o modelo.
    global DADOS_HISTORICOS, TOTAL_REGISTROS, MODELO_REGRESSAO, PRE_PROCESSADOR
    
    try:
        df_scores = pd.read_csv(CAMINHO_DADOS_NOTAS)
        
        #Engenharia de Features: Cria a coluna 'risco_estresse'
        np.random.seed(42)
        df_scores['risco_estresse'] = np.random.randint(1, 6, size=len(df_scores)) 
        
        DADOS_HISTORICOS = df_scores.to_dict('records')
        TOTAL_REGISTROS = len(DADOS_HISTORICOS)
        
        #Configuração e Treinamento do Modelo de Regressão Linear
        
        #Nomes das colunas do CSV para o ML.
        FEATURES = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'risco_estresse'] 
        TARGET = 'exam_score'
        
        X = df_scores[FEATURES]
        y = df_scores[TARGET]
        
        #Normalização é crucial pro ML.
        PRE_PROCESSADOR = StandardScaler()
        X_escalado = PRE_PROCESSADOR.fit_transform(X) 
        
        MODELO_REGRESSAO = LinearRegression()
        MODELO_REGRESSAO.fit(X_escalado, y)
        
        print("Modelo de Regressão Linear treinado com sucesso! Serviço de IA pronto.")
        
    except FileNotFoundError:
        print("ERRO: CSV 'student_exam_scores.csv' não encontrado. API vai rodar, mas sem análise preditiva.")
        MODELO_REGRESSAO = None
    except Exception as e:
        print(f"Erro inesperado durante o setup de ML: {e}")
        MODELO_REGRESSAO = None

#Inicializando o setup.
carregar_e_treinar_modelo()


#LÓGICA DE ANÁLISE PREDITIVA
def realizar_analise_preditiva(dados_aluno: DadosNovoAluno) -> Dict:
    #Recebe dados, prevê nota e gera sugestões de hábito.
    if MODELO_REGRESSAO is None:
        return {"erro": "O modelo de ML está offline. Não foi possível carregar os dados."}
    
    # NOVAS FEATURES (TRADUZIDAS E NA ORDEM CORRETA)
    FEATURES = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'risco_estresse']

    # CORREÇÃO DO WARNING: Criar um DataFrame com os nomes das colunas para o transform()
    
    # 1. Cria um dicionário com os dados de entrada na ordem correta
    dados_dict = {
        'hours_studied': [dados_aluno.horas_estudo], 
        'sleep_hours': [dados_aluno.horas_sono], 
        'attendance_percent': [dados_aluno.presenca_percentual], 
        'previous_scores': [dados_aluno.nota_anterior], 
        'risco_estresse': [dados_aluno.risco_estresse_simulado]
    }

    # 2. Converte para um DataFrame (X_novo)
    X_novo = pd.DataFrame(dados_dict, columns=FEATURES) 
    
    # Escalonando e Prevendo a nota.
    dados_escalados = PRE_PROCESSADOR.transform(X_novo) 
    nota_prevista = MODELO_REGRESSAO.predict(dados_escalados)[0]
    
    #Geração de Feedback Intuitivo (Regras de Negócio)
    sugestoes = []
    
    if dados_aluno.horas_estudo < 4:
        sugestoes.append(f"Suas {dados_aluno.horas_estudo} horas de estudo estão baixas. Aumentar esse tempo para elevar a nota prevista.")
    
    if dados_aluno.horas_sono < 7:
        sugestoes.append(f"O sono está comprometido com {dados_aluno.horas_sono} horas. Priorize 7-9 horas para uma melhor performance cognitiva.")
        
    if dados_aluno.presenca_percentual < 75:
        sugestoes.append(f"Presença de {dados_aluno.presenca_percentual}% é um fator de risco. A frequência em aula tem impacto direto no sucesso.")
    
    if dados_aluno.risco_estresse_simulado >= 4:
          sugestoes.append("Alto nível de estresse detectado. Inclua pausas e atividades de lazer; o descanso melhora a absorção do conteúdo.")

    if not sugestoes:
          sugestoes.append("Seus hábitos são exemplares! Continue assim.")

    
    return {
        "nota_prevista_simulada": round(nota_prevista, 2),
        "analise_de_habitos_e_risco": {
            "total_pontos_de_melhoria": len(sugestoes),
            "sugestoes_personalizadas": sugestoes
        },
        "aviso_modelo_ml": "A previsão é baseada em Regressão Linear. É uma tendência, não uma certeza."
    }


#ROTAS DA API

#Configurando o domínio personalizado e a descrição da funcionalidade.
app = FastAPI(
    title="StudyMetrics-API | Análise Preditiva Acadêmica", 
    description="PROPOSITO: API que prevê o desempenho de exames e sugere melhorias de hábitos, usando Machine Learning (ML).",
    version="1.0.0"
)

#ROTA 1: ENDPOINT GERAL (HOME)
@app.get("/", tags=["1. Geral"])
def home():
    """Endpoint que retorna informações sobre a API (Requisito 2.1 - Home)."""
    return {
        "projeto": "Projeto 4: Análise Preditiva de Estudantes",
        "autor": "Seu Nome",
        "total_registros_treinamento": TOTAL_REGISTROS,
        "instrucao_principal": "Use o POST /analisar_novos_dados para obter a análise personalizada (Método Intermediário)."
    }

#ROTA 2: ENDPOINT BÁSICO (LISTAGEM COMPLETA)
@app.get("/dados_historicos", tags=["1. Geral"])
def listar_todos():
    """
    Endpoint BÁSICO (Nível 1): Retorna o dataset completo usado para o treinamento do ML (Requisito 2.1 - Listar todos).
    """
    if not DADOS_HISTORICOS:
        raise HTTPException(status_code=500, detail={"erro": "Dados históricos indisponíveis."})
        
    return DADOS_HISTORICOS


#ROTA 3: O ENDPOINT INTERMEDIÁRIO DE UTILIDADE (POST)
@app.post("/analisar_novos_dados", response_model=RespostaModelo, tags=["2. Análise Preditiva (POST - Intermediário)"])
def analisar_desempenho_novo_aluno(dados: DadosNovoAluno):
    """
    ENDPOINT PRINCIPAL (Nível 2) - **Aonde você coloca seu INPUT**.
    Recebe os dados de hábitos (Body Payload) e usa o modelo de ML para PREVER a nota e dar SUGESTÕES.
    """
    
    if MODELO_REGRESSAO is None:
        raise HTTPException(status_code=503, detail="Serviço de Análise (ML) indisponível.")
        
    analise_preditiva = realizar_analise_preditiva(dados)
    
    # Retorna o resultado JSON para o usuário.
    return JSONResponse(content={
        "dados_inseridos_pelo_usuario": dados.dict(),
        "resultado_do_modelo": analise_preditiva,
        "mensagem_sucesso": "Análise concluída. Confira a previsão e as sugestões personalizadas."
    })


#ROTA 4: FILTRO POR PATH PARAMETER (Busca por ID)
@app.get("/aluno_historico/{id_aluno}", tags=["3. Filtro de Dados"])
def buscar_aluno_historico(id_aluno: str):
    """
    Endpoint de Busca: Filtra um registro histórico pelo ID do aluno (Path Parameter).
    Exemplo de ID Válido: S001, S005, S008.
    """
    
    id_aluno_formatado = id_aluno.upper()
    aluno_encontrado = None
    
    #Busca o ID no dicionário de dados
    for aluno in DADOS_HISTORICOS:
        if str(aluno.get('student_id')).upper() == id_aluno_formatado: 
            aluno_encontrado = aluno
            break
            
    if not aluno_encontrado:
        raise HTTPException(status_code=404, detail=f"Aluno com ID '{id_aluno}' não encontrado nos dados históricos.")
        
    return aluno_encontrado


#BLOCO DE EXECUÇÃO: Roda a API no terminal Python
if __name__ == "__main__":
    # Mensagens de boas-vindas e instruções CLARAS para o usuário no terminal
    HOST = "127.0.0.1"
    PORT = 8000
    
    print("\n--- SERVIÇO DE API INICIADO ---")
    print("PROPOSITO: Análise Preditiva de Notas e Hábitos.")
    print("-" * 30)
    print(f"STATUS DO ML: {'ONLINE' if MODELO_REGRESSAO is not None else 'OFFLINE'}")
    print("\n-- INSTRUÇÕES DE ACESSO --")
    print(f"ACESSO À API: http://{HOST}:{PORT}")
    print(f"DOCUMENTAÇÃO: http://{HOST}:{PORT}/docs")
    print("\nUse a documentação para testar o endpoint POST /analisar_novos_dados (O principal).")
    print("-" * 30)
    
    # Inicia o servidor Uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)