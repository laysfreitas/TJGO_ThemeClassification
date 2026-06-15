"""
Teste de sanidade do pipeline BERT + Projection Head + SupCon Loss
seguido de BERT + Classifier Head + Cross-Entropy Loss.

Objetivo:
- Verificar se o pipeline roda na máquina local antes do treino completo.
- Usar poucas classes, poucas amostras e max_length reduzido.

Como executar:
    python teste_supcon_bert.py

Antes de executar, ajuste a variável PARQUET_PATH para o caminho correto do seu arquivo.
"""

import os
import re
import random
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import joblib

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. Configurações gerais
# ============================================================

PARQUET_PATH = r"C:\Users\lfmelo\Documents\Github\TJGO_ThemeClassification\data\dataset_base.parquet"

# Nome da coluna textual no parquet
TEXT_COL = "texto"

# Nome da coluna de classe no parquet.
# Ajuste caso sua coluna seja, por exemplo: "tema_codigo", "classe", "label_encoded", etc.
LABEL_COL = "label"

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

SEED = 42

# Configuração leve para teste de sanidade
MAX_LENGTH = 128
BATCH_SIZE = 4
N_CLASSES_TESTE = 3
N_EXEMPLOS_POR_CLASSE = 10

PROJECTION_DIM = 64
TEMPERATURE = 0.07

EPOCHS_SUPCON = 1
EPOCHS_CLASSIFIER = 1

LR_SUPCON = 2e-5
LR_CLASSIFIER = 2e-5


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. Limpeza textual
# ============================================================

_INTEIRO_TEOR_PATTERNS = [
    (re.compile(r"(Erro Parser)?>>>>>inicio<<<<<\n?", re.MULTILINE | re.IGNORECASE), ""),
    (re.compile(r"fimid:\d+|#####fim#####id:\d+\n?", re.MULTILINE), ""),
]

_MULTI_SPACE = re.compile(r" {2,}")


def limpar_inteiro_teor(text: str) -> str:
    text = str(text)

    for pattern, replacement in _INTEIRO_TEOR_PATTERNS:
        text = pattern.sub(replacement, text)

    return _MULTI_SPACE.sub(" ", text).strip()


# ============================================================
# 3. SupCon Loss
# ============================================================

from losses import SupConLoss

# ============================================================
# 4. Modelo BERT + Projection Head
# ============================================================

class BertSupConModel(nn.Module):
    def __init__(self, model_name: str, projection_dim: int = 128):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Embedding do token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        projected_embedding = self.projection_head(cls_embedding)

        # Normalização L2, comum em aprendizado contrastivo
        projected_embedding = nn.functional.normalize(
            projected_embedding,
            p=2,
            dim=1,
        )

        return projected_embedding


# ============================================================
# 5. Modelo BERT + Classifier Head
# ============================================================

class BertClassifier(nn.Module):
    def __init__(self, bert, num_labels: int):
        super().__init__()

        self.bert = bert

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)

        return logits


# ============================================================
# 6. Carregamento e preparação dos dados
# ============================================================

def carregar_dados() -> pd.DataFrame:
    print("Carregando parquet...")
    dataf = pl.read_parquet(PARQUET_PATH)
    df = dataf.to_pandas()

    if TEXT_COL not in df.columns:
        raise ValueError(f"Coluna de texto '{TEXT_COL}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Coluna de label '{LABEL_COL}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

    df[TEXT_COL] = df[TEXT_COL].map(limpar_inteiro_teor)

    # Remove textos vazios depois da limpeza
    df = df[df[TEXT_COL].str.len() > 0].copy()

    print("Shape original após limpeza:", df.shape)

    return df


def criar_amostra_teste(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma amostra pequena para testar se o pipeline roda.

    Mantém apenas classes com pelo menos N_EXEMPLOS_POR_CLASSE exemplos.
    Depois seleciona N_CLASSES_TESTE classes e N_EXEMPLOS_POR_CLASSE exemplos por classe.
    """

    contagem = df[LABEL_COL].value_counts()

    classes_validas = contagem[contagem >= N_EXEMPLOS_POR_CLASSE].index.tolist()

    if len(classes_validas) < N_CLASSES_TESTE:
        raise ValueError(
            f"Não há classes suficientes com pelo menos {N_EXEMPLOS_POR_CLASSE} exemplos. "
            f"Classes válidas encontradas: {len(classes_validas)}"
        )

    classes_amostra = classes_validas[:N_CLASSES_TESTE]

    df_teste = df[df[LABEL_COL].isin(classes_amostra)].copy()

    df_teste = (
        df_teste
        .groupby(LABEL_COL, group_keys=False)
        .head(N_EXEMPLOS_POR_CLASSE)
        .reset_index(drop=True)
    )

    print("\nAmostra de teste criada:")
    print(df_teste[LABEL_COL].value_counts())

    return df_teste


def codificar_labels(df: pd.DataFrame):
    label_encoder = LabelEncoder()

    df = df.copy()
    df["label"] = label_encoder.fit_transform(df[LABEL_COL])

    num_labels = df["label"].nunique()

    print("\nClasses codificadas:")
    for classe_original, classe_codificada in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"{classe_original} -> {classe_codificada}")

    print("\nNúmero de classes:", num_labels)

    return df, label_encoder, num_labels


# ============================================================
# 7. Tokenização e DataLoaders
# ============================================================

def criar_dataloaders(df: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset_total = Dataset.from_pandas(df[[TEXT_COL, "label"]])

    dataset_split = dataset_total.train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    def tokenizar(batch):
        return tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset_tokenizado = dataset_split.map(
        tokenizar,
        batched=True,
    )

    dataset_tokenizado = dataset_tokenizado.remove_columns([TEXT_COL])

    dataset_tokenizado.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    train_dataloader = DataLoader(
        dataset_tokenizado["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset_tokenizado["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nTamanho treino:", len(dataset_tokenizado["train"]))
    print("Tamanho teste:", len(dataset_tokenizado["test"]))

    return train_dataloader, test_dataloader, tokenizer


# ============================================================
# 8. Treinamento SupCon
# ============================================================

def treinar_supcon(train_dataloader, device):
    model_supcon = BertSupConModel(
        model_name=MODEL_NAME,
        projection_dim=PROJECTION_DIM,
    )

    model_supcon.to(device)

    loss_fn_supcon = SupConLoss(temperature=TEMPERATURE)

    optimizer = AdamW(
        model_supcon.parameters(),
        lr=LR_SUPCON,
    )

    model_supcon.train()

    print("\nIniciando treino SupCon...")

    for epoch in range(EPOCHS_SUPCON):
        total_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            embeddings_1 = model_supcon(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            embeddings_2 = model_supcon(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            features = torch.stack(
                [embeddings_1, embeddings_2],
                dim=1
            )

            loss = loss_fn_supcon(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{EPOCHS_SUPCON} | Loss SupCon: {avg_loss:.4f}")

    # Remove a projection head na prática: preserva apenas o BERT treinado
    bert_treinado = model_supcon.bert

    return bert_treinado


# ============================================================
# 9. Treinamento Cross-Entropy
# ============================================================

def treinar_classificador(bert_treinado, num_labels, train_dataloader, device):
    model_classifier = BertClassifier(
        bert=bert_treinado,
        num_labels=num_labels,
    )

    model_classifier.to(device)

    loss_fn_cls = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model_classifier.parameters(),
        lr=LR_CLASSIFIER,
    )

    model_classifier.train()

    print("\nIniciando treino do classificador...")

    for epoch in range(EPOCHS_CLASSIFIER):
        total_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model_classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn_cls(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{EPOCHS_CLASSIFIER} | Loss CE: {avg_loss:.4f}")

    return model_classifier


# ============================================================
# 10. Avaliação
# ============================================================

def avaliar_modelo(model_classifier, test_dataloader, device):
    model_classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model_classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print("\nResultados do teste de sanidade:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

# ============================================================
# 11. Função para prever classe de um texto
# ============================================================

def prever_tema(
    texto: str, 
    model_classifier,
    tokenizer,
    label_encoder,
    device,
    max_length: int = MAX_LENGTH
):

    model_classifier.eval()

    texto = limpar_inteiro_teor(texto)

    inputs = tokenizer(
        texto,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model_classifier(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilidades = torch.softmax(logits, dim=1)

        classe_predita_id = torch.argmax(probabilidades, dim=1).item()

        confianca = probabilidades[0, classe_predita_id].item()

    classe_original = label_encoder.inverse_transform(
        [classe_predita_id]
    )[0]

    return {
        "classe_predita": classe_original,
        "classe_id": classe_predita_id,
        "confianca": confianca
    }

# ============================================================
# 1. Execução principal
# ============================================================

def main():
    print("===============================================")
    print("Teste BERT + SupCon Loss + Cross-Entropy")
    print("===============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = carregar_dados()

    df_teste = criar_amostra_teste(df)

    df_teste, label_encoder, num_labels = codificar_labels(df_teste)

    train_dataloader, test_dataloader , tokenizer = criar_dataloaders(df_teste)

    bert_treinado = treinar_supcon(
        train_dataloader=train_dataloader,
        device=device,
    )

    model_classifier = treinar_classificador(
        bert_treinado=bert_treinado,
        num_labels=num_labels,
        train_dataloader=train_dataloader,
        device=device,
    )

    avaliar_modelo(
        model_classifier=model_classifier,
        test_dataloader=test_dataloader,
        device=device,
    )
    

    texto_exemplo = """
    Trata-se de ação em que a parte autora busca reparação por danos morais
    em razão de suposta falha na prestação de serviço.
    """

    resultado = prever_tema(
        texto=texto_exemplo,
        model_classifier=model_classifier,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        device=device,
        max_length=MAX_LENGTH
    )

    print("\nPredição para texto de exemplo:", resultado)

    print("\nTeste finalizado com sucesso.")


if __name__ == "__main__":
    main()
