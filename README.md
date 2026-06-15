# TJGO Theme Classification

Projeto de classificação automática de temas jurídicos a partir de textos processuais, utilizando modelos baseados em BERT/BERTimbau e aprendizado contrastivo supervisionado.

O objetivo principal é treinar um modelo capaz de representar documentos juridicamente semelhantes em regiões próximas do espaço vetorial e, posteriormente, utilizar essas representações para classificar os textos em suas respectivas classes temáticas.

---

## Visão Geral

Este projeto utiliza um pipeline de treinamento em duas etapas:

1. **Aprendizado de Representação**

   * Treinamento do encoder BERT em conjunto com uma *Projection Head*.
   * Uso da função de perda **Supervised Contrastive Loss (SupCon Loss)**.
   * Objetivo: aproximar os embeddings de documentos pertencentes à mesma classe e afastar documentos de classes diferentes.

2. **Classificação Supervisionada**

   * Remoção da *Projection Head* após o treinamento contrastivo.
   * Preservação do encoder BERT ajustado.
   * Adição de uma camada classificadora linear.
   * Treinamento final com **Cross-Entropy Loss** para previsão da classe temática.

---

## Objetivo do Modelo

O modelo busca aprender representações vetoriais mais discriminativas para textos jurídicos. Durante a primeira etapa, documentos da mesma classe são aproximados no espaço de embeddings, enquanto documentos de classes distintas são afastados.

Após essa etapa, o encoder ajustado é reaproveitado em uma tarefa supervisionada tradicional, na qual uma camada classificadora é treinada para prever o tema jurídico correspondente a cada documento.

---

## Etapas do Pipeline

O fluxo geral do projeto segue as seguintes etapas:

1. Importação das bibliotecas necessárias.
2. Carregamento da base de dados em formato Parquet.
3. Limpeza e normalização dos textos.
4. Codificação dos rótulos das classes.
5. Criação do dataset no formato Hugging Face.
6. Tokenização dos textos.
7. Criação dos `DataLoaders`.
8. Definição do modelo `BertSupConModel`.
9. Treinamento do BERT com `SupConLoss`.
10. Preservação do encoder treinado: `bert_treinado = model_supcon.bert`.
11. Criação do modelo `BertClassifier`.
12. Treinamento da classificação com `CrossEntropyLoss`.
13. Avaliação do modelo com métricas como acurácia e F1-score.

---

## Arquitetura do Treinamento

### Etapa 1 — Aprendizado de Representação

```text
Texto jurídico
     ↓
Tokenizer
     ↓
BERT / BERTimbau
     ↓
Embedding [CLS]
     ↓
Projection Head
     ↓
Embedding contrastivo
     ↓
SupCon Loss
```

Nesta etapa, o modelo aprende a organizar os documentos no espaço vetorial. Textos da mesma classe tendem a ficar próximos, enquanto textos de classes diferentes são afastados.

---

### Etapa 2 — Classificação

```text
Texto jurídico
     ↓
Tokenizer
     ↓
BERT / BERTimbau ajustado
     ↓
Embedding [CLS]
     ↓
Classifier Head
     ↓
Logits
     ↓
Softmax
     ↓
Classe prevista
     ↓
Cross-Entropy Loss
```

Após o treinamento contrastivo, a *Projection Head* é removida. O encoder BERT ajustado é conectado a uma camada classificadora linear, responsável por prever a classe final do documento.

---

## Descrição Técnica

O treinamento foi dividido em duas etapas. Na primeira, o encoder BERT foi treinado em conjunto com uma *Projection Head* utilizando a função de perda **Supervised Contrastive Loss**. Essa etapa tem como objetivo aproximar as representações vetoriais de documentos pertencentes à mesma classe e afastar documentos de classes distintas.

Após essa etapa, a *Projection Head* foi removida, preservando-se apenas o encoder ajustado. Em seguida, foi adicionada uma camada classificadora linear sobre o embedding produzido pelo BERT. Essa nova arquitetura foi treinada com **Cross-Entropy Loss** para a tarefa final de classificação supervisionada.

---

## Hiperparâmetros Iniciais

Os hiperparâmetros iniciais recomendados para os primeiros experimentos são:

```python
projection_dim = 128
temperature = 0.07
batch_size = 16
epochs_supcon = 2
epochs_classifier = 3
lr_supcon = 2e-5
lr_classifier = 2e-5
```

### Ajustes Recomendados

Caso existam muitas classes e poucos exemplos por classe, recomenda-se testar:

```python
batch_size = 32
```

Caso o modelo apresente sinais de *overfitting*, recomenda-se reduzir o número de épocas da etapa classificatória:

```python
epochs_classifier = 2
```

Caso a SupCon Loss não esteja produzindo bons agrupamentos no espaço vetorial, recomenda-se testar:

```python
epochs_supcon = 3
temperature = 0.1
```

---

## Métricas de Avaliação

O desempenho do modelo deve ser avaliado utilizando métricas adequadas para classificação multiclasse, especialmente em cenários com classes desbalanceadas.

As métricas recomendadas são:

* **Accuracy**: mede a proporção geral de acertos.
* **F1-score Macro**: calcula a média do F1-score atribuindo o mesmo peso para todas as classes.
* **F1-score Weighted**: calcula a média do F1-score ponderada pela quantidade de exemplos em cada classe.

Em bases desbalanceadas, o **F1-score Macro** é especialmente importante, pois permite verificar se o modelo está performando bem também nas classes minoritárias.

---

## Estrutura Conceitual do Modelo

```text
FASE 1: Representação Contrastiva

BERT + Projection Head
        ↓
SupCon Loss
        ↓
Encoder ajustado


FASE 2: Classificação Supervisionada

Encoder ajustado
        ↓
Classifier Head
        ↓
Cross-Entropy Loss
        ↓
Classe temática prevista
```

---

## Observações Importantes

A utilização da **SupCon Loss** depende fortemente da composição dos batches. Para que o aprendizado contrastivo seja efetivo, é importante que cada batch contenha exemplos de uma mesma classe, permitindo a formação de pares positivos.

Batches muito pequenos podem prejudicar o treinamento contrastivo, principalmente quando há muitas classes. Por isso, sempre que possível, recomenda-se utilizar `batch_size` igual ou superior a 16.

---

## Modelo Base

O modelo base sugerido para os experimentos é:

```python
neuralmind/bert-base-portuguese-cased
```

Também podem ser testados modelos jurídicos em português, como versões de BERT treinadas ou ajustadas em textos legais, caso estejam disponíveis e sejam compatíveis com a tarefa.

---

## Resultado Esperado

Espera-se que o modelo final seja capaz de:

* Representar textos jurídicos semanticamente semelhantes em regiões próximas do espaço vetorial.
* Separar melhor documentos pertencentes a classes distintas.
* Melhorar o desempenho da classificação temática.
* Produzir resultados avaliáveis por acurácia, F1-score macro e F1-score weighted.

---

## Resumo do Pipeline

```text
Dados brutos
   ↓
Limpeza textual
   ↓
Codificação dos labels
   ↓
Tokenização
   ↓
Treinamento contrastivo com SupCon Loss
   ↓
Remoção da Projection Head
   ↓
Treinamento do classificador com Cross-Entropy
   ↓
Avaliação final
```
