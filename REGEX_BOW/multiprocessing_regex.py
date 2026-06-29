import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# Variáveis globais usadas pelos processos
_TEXTOS_GLOBAL = None
_TEMAS_REAIS_GLOBAL = None
_CORTE_GLOBAL = None


def _inicializar_worker(textos, temas_reais, corte_porcentagem):
    """
    Inicializa dados globais dentro de cada processo.
    Evita passar o DataFrame inteiro em cada tarefa.
    """
    global _TEXTOS_GLOBAL, _TEMAS_REAIS_GLOBAL, _CORTE_GLOBAL

    _TEXTOS_GLOBAL = textos
    _TEMAS_REAIS_GLOBAL = temas_reais
    _CORTE_GLOBAL = corte_porcentagem


def _extrair_blocos_regex(regex_completa):
    """
    Extrai os blocos internos da regex, seguindo a mesma lógica da função original.
    """
    if pd.isna(regex_completa) or str(regex_completa).upper() == "NAN":
        return []

    regex_completa = str(regex_completa)

    blocos_sujos = re.findall(r"\?=\.\*\?\(([^)]+)\)", regex_completa)

    if not blocos_sujos:
        blocos_sujos = re.findall(r"\(([^)]+)\)", regex_completa)

    blocos = []

    for bloco in blocos_sujos:
        bloco_limpo = bloco.replace(r"\(", "(").replace(r"\)", ")")
        blocos.append(bloco_limpo)

    return blocos


def _normalizar_matches(matches):
    """
    Normaliza os resultados do re.findall.

    O re.findall pode retornar:
    - strings
    - tuplas, quando há grupos de captura
    """
    termos = []

    for match in matches:
        if isinstance(match, tuple):
            for item in match:
                if item:
                    termos.append(str(item))
        else:
            if match:
                termos.append(str(match))

    return termos


def _processar_uma_regex(tarefa):
    """
    Processa uma única regex contra todos os textos do DataFrame.

    Retorna apenas as linhas em que a porcentagem atingiu o corte.
    """
    regex_completa, tema_codigo = tarefa

    blocos = _extrair_blocos_regex(regex_completa)
    total_blocos = len(blocos)

    if total_blocos == 0:
        return []

    tema_codigo_str = str(tema_codigo)
    resultados = []

    for idx, texto in _TEXTOS_GLOBAL.items():
        if pd.isna(texto) or not isinstance(texto, str):
            continue

        blocos_com_match = 0
        termos_da_linha = []

        for bloco in blocos:
            try:
                matches_bloco = re.findall(
                    bloco,
                    texto,
                    flags=re.IGNORECASE
                )

            except re.error:
                bloco_seguro = "|".join(
                    [
                        re.escape(termo.strip().replace("\\", ""))
                        for termo in bloco.split("|")
                    ]
                )

                matches_bloco = re.findall(
                    bloco_seguro,
                    texto,
                    flags=re.IGNORECASE
                )

            if matches_bloco:
                blocos_com_match += 1
                termos_da_linha.extend(_normalizar_matches(matches_bloco))

        porcentagem_sucesso = (blocos_com_match / total_blocos) * 100

        if porcentagem_sucesso >= _CORTE_GLOBAL:
            termos_unicos = list(
                set(
                    termo
                    for termo in termos_da_linha
                    if isinstance(termo, str) and termo.strip()
                )
            )

            acertou = False

            if idx in _TEMAS_REAIS_GLOBAL:
                acertou = str(_TEMAS_REAIS_GLOBAL[idx]) == tema_codigo_str

            resultados.append(
                (
                    idx,
                    f"{tema_codigo_str} ({porcentagem_sucesso:.1f}%)",
                    f"{tema_codigo_str}: {termos_unicos}",
                    acertou
                )
            )

    return resultados


def aplicar_regex_flexivel_no_dataframe_multiprocessing(
    df,
    df_regex,
    corte_porcentagem=30.0,
    max_workers=4
):
    """
    Aplica expressões regulares de forma flexível usando multiprocessing.

    Paraleliza por REGEX/tema:
    - cada processo recebe uma regex;
    - testa essa regex contra todos os textos;
    - retorna os resultados;
    - o processo principal consolida no DataFrame.
    """

    df = df.copy()

    df["temas_encontrados"] = [[] for _ in range(len(df))]
    df["termos_capturados"] = [[] for _ in range(len(df))]
    df["acertou_tema"] = False

    textos = df["inteiro_teor_lematizado"].to_dict()

    if "TEMA_CODIGO" in df.columns:
        temas_reais = df["TEMA_CODIGO"].astype(str).to_dict()
    else:
        temas_reais = {}

    tarefas = []

    for _, row in df_regex.iterrows():
        regex_completa = row["PALAVRAS_LEMATIZADAS_REGEX"]
        tema_codigo = row["TEMA CÓDIGO"]

        if pd.isna(regex_completa) or str(regex_completa).upper() == "NAN":
            continue

        tarefas.append((regex_completa, tema_codigo))

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_inicializar_worker,
        initargs=(textos, temas_reais, corte_porcentagem)
    ) as executor:

        for resultado_regex in executor.map(_processar_uma_regex, tarefas):
            for idx, tema_encontrado, termos_capturados, acertou in resultado_regex:
                df.at[idx, "temas_encontrados"].append(tema_encontrado)
                df.at[idx, "termos_capturados"].append(termos_capturados)

                if acertou:
                    df.at[idx, "acertou_tema"] = True

    return df