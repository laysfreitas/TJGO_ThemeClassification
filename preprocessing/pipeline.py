"""Orquestração do pipeline de preparação de petições.

Compõe as funções puras de :mod:`cleaners` na ordem canônica,
preserva ``texto_original`` e gera métricas por etapa para logs e para o
arquivo ``references/transformation-log.md``.

Design:
- :class:`EtapaLimpeza` encapsula nome + função + métrica agregada.
- :func:`construir_etapas` devolve a lista padrão (1 única fonte da verdade).
- :func:`aplicar_pipeline_texto` roda todas as etapas num texto, devolvendo
  o texto final + dicionário de tamanhos após cada etapa (usado em
  diagnóstico/amostragem).
- :func:`aplicar_pipeline_df` roda sobre um DataFrame completo, gerando
  as colunas ``texto_limpo`` e ``n_marcadores_original`` e deduplicando
  por ``texto_limpo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import cleaners


@dataclass(frozen=True)
class EtapaLimpeza:
    """Uma etapa do pipeline: nome + função pura str→str + referência ao ruído."""

    nome: str
    ruido: str
    funcao: Callable[[str], str]


def construir_etapas() -> list[EtapaLimpeza]:
    """Construir a sequência canônica de etapas de limpeza.

    Ordem definida em ``noise-patterns.md`` (seção "Sequência
    recomendada de limpeza").
    """

    return [
        EtapaLimpeza(
            nome="normalizar_unicode",
            ruido="N7",
            funcao=cleaners.normalizar_unicode,
        ),
        EtapaLimpeza(
            nome="separar_e_consolidar_documentos",
            ruido="Q1+N1",
            funcao=cleaners.separar_e_consolidar_documentos,
        ),
        EtapaLimpeza(
            nome="remover_urls_camscanner",
            ruido="N3",
            funcao=cleaners.remover_urls_camscanner,
        ),
        EtapaLimpeza(
            nome="remover_urls_genericas",
            ruido="N8",
            funcao=cleaners.remover_urls_genericas,
        ),
        EtapaLimpeza(
            nome="cortar_cabecalho_institucional",
            ruido="N2",
            funcao=cleaners.cortar_cabecalho_institucional,
        ),
        EtapaLimpeza(
            nome="cortar_assinatura_final",
            ruido="N4",
            funcao=cleaners.cortar_assinatura_final,
        ),
        EtapaLimpeza(
            nome="remover_codigos_controle",
            ruido="N6",
            funcao=cleaners.remover_codigos_controle,
        ),
        EtapaLimpeza(
            nome="colapsar_espacos",
            ruido="N7-final",
            funcao=cleaners.colapsar_espacos,
        ),
    ]


def aplicar_pipeline_texto(
    texto: str,
    etapas: list[EtapaLimpeza] | None = None,
) -> tuple[str, dict[str, int]]:
    """Aplicar todas as etapas em um único texto.

    Args:
        texto: texto bruto da coluna ``inteiro_teor``.
        etapas: sequência de etapas. Default: :func:`construir_etapas`.

    Returns:
        Tupla ``(texto_limpo, tamanhos)`` onde ``tamanhos`` é um dict
        ``{nome_etapa: len_apos_etapa}`` mais a chave ``"_original"``.
    """

    if etapas is None:
        etapas = construir_etapas()
    tamanhos: dict[str, int] = {"_original": len(texto or "")}
    atual = texto or ""
    for etapa in etapas:
        atual = etapa.funcao(atual)
        tamanhos[etapa.nome] = len(atual)
    return atual, tamanhos

def split_512_palavras(texto, tamanho=512):
    
    if pd.isna(texto):
        return []
    
    palavras = str(texto).split()
    
    return [
        " ".join(palavras[i:i + tamanho])
        for i in range(0, len(palavras), tamanho)
    ]

def gerar_chunks(df):

    df["chunks"] = df["text"].apply(split_512_palavras)

    df_chunks = df.explode("chunks").reset_index(drop=True)

    df_chunks = df_chunks.rename(columns={
        "chunks": "texts"
    })

    return df_chunks[['labels','texts']].rename(columns={
        "labels":"label",
        "texts":"text"
    })

import evaluate

def load_evaluation_metrics() :
    return {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
    }
def compute_metrics(eval_prediction):
      logits, labels = eval_prediction
      predictions = np.argmax(logits, axis=-1)
      return calculate_metrics(predictions, labels)


def calculate_metrics(predictions, labels):
      metrics = load_evaluation_metrics()
      metric_inputs = {"predictions": predictions, "references": labels}
      return {
          "accuracy": float(metrics["accuracy"].compute(**metric_inputs)["accuracy"]),
          "f1_macro": float(metrics["f1"].compute(**metric_inputs, average="macro")["f1"]),
          "f1_weighted": float(metrics["f1"].compute(**metric_inputs, average="weighted")["f1"]),
          "precision_macro": float(
              metrics["precision"].compute(
                  **metric_inputs,
                  average="macro",
                  zero_division=0,
              )["precision"]
          ),
          "recall_macro": float(
              metrics["recall"].compute(
                  **metric_inputs,
                  average="macro",
                  zero_division=0,
              )["recall"]
          ),
      }
print(compute_metrics(([[0.5, -0.5], [-0.5, 0.5]], [0, 1])))