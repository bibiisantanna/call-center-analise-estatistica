import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

PASTA_DADOS = "dados"


def ler_arquivo(caminho):
    with open(caminho, 'r') as f:
        dados = [float(l.strip()) for l in f if l.strip()]
    return np.array(dados)


def converter_acumulado_para_intervalo(dados):
    """Transforma tempo acumulado em tempo entre chamadas."""
    return np.diff(dados, prepend=dados[0])


def extrair_horario(nome_arquivo):
    """Extrai a hora inicial do arquivo e gera rótulo 'Chamadas HH–HH+1'."""
    numeros = re.findall(r'\d+', nome_arquivo)
    if numeros:
        h = int(numeros[0])
        return f"Chamadas {h:02d}h–{(h+1):02d}h"
    return nome_arquivo.replace(".txt", "")


def hora_inicial(nome_arquivo):
    """Retorna a hora inicial para ordenação crescente."""
    numeros = re.findall(r'\d+', nome_arquivo)
    return int(numeros[0]) if numeros else 99


def analisar_dados(nome, dados):
    print(f"\n📊 {nome}")

    media = np.mean(dados)
    variancia = np.var(dados)
    desvio = np.std(dados)
    lam = 1 / media if media > 0 else 0

    print(f"  Média:        {media:.4f} s")
    print(f"  Variância:    {variancia:.4f}")
    print(f"  Desvio padrão:{desvio:.4f} s")
    print(f"  λ estimado:   {lam:.4f} chamadas/s")

    # Histograma com ajuste exponencial
    plt.figure()
    plt.hist(dados, bins=30, density=True, alpha=0.6, label="Dados observados")

    loc, scale = expon.fit(dados)
    x = np.linspace(min(dados), max(dados), 200)
    plt.plot(x, expon.pdf(x, loc, scale), label=f"Exp(λ={lam:.4f})")

    plt.axvline(media, color='orange', linestyle='--', label=f"μ = {media:.3f}")
    plt.title(nome)
    plt.xlabel("Tempo entre chamadas (s)")
    plt.ylabel("Densidade de probabilidade")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if abs(media - desvio) < 0.2 * media:
        print("  ✔ Forte indício de distribuição exponencial (μ ≈ σ)")
    else:
        print("  ⚠ Pode não ser exponencial — verificar gráfico")


def main():
    arquivos = sorted(
        [f for f in os.listdir(PASTA_DADOS) if f.endswith(".txt")],
        key=hora_inicial
    )

    for arquivo in arquivos:
        caminho = os.path.join(PASTA_DADOS, arquivo)
        dados = ler_arquivo(caminho)

        if "Horario" in arquivo:
            dados = converter_acumulado_para_intervalo(dados)

        nome = extrair_horario(arquivo)
        analisar_dados(nome, dados)


if __name__ == "__main__":
    main()