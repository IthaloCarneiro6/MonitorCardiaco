#gera uma amostra real de bradicardia sinusal do MIT-BIH (registro 232), escolhe o trecho de 40 s mais "normais"

import os
import sys
from collections import Counter

import numpy as np

from nucleo_ecg import FS
from preparar_dados import PASTA_DATASET, AAMI, ler_registro, ler_anotacoes, ajustar_picos

DURACAO_S = 40

pasta = sys.argv[1] if len(sys.argv) > 1 else PASTA_DATASET
sinal = ler_registro(pasta, "232")
batidas, _ = ler_anotacoes(pasta, "232")
bat = [(t, s) for t, s in batidas if s in AAMI]
pos = ajustar_picos(sinal, [t for t, _ in bat])
tipos = [s for _, s in bat]

melhor = None
for i in range(len(pos) - 5):
    j = np.searchsorted(pos, pos[i] + DURACAO_S * FS)
    if j >= len(pos):
        break
    rr = np.diff(pos[i:j + 1]) / FS
    extras = sum(t == "A" for t in tipos[i:j])
    nota = -rr.max() - 0.05 * extras
    if melhor is None or nota > melhor[0]:
        melhor = (nota, i, j)

_, i, j = melhor
trecho = sinal[pos[i]:pos[j]]
rr = np.diff(pos[i:j + 1]) / FS
cont = Counter(tipos[i:j])

saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amostra_bradicardia_232.csv")
np.savetxt(saida, trecho, fmt="%.5f")
print(f"Salvo: {saida}")
print(f"Duração: {len(trecho) / FS:.0f} s | FC média: {60 / rr.mean():.0f} bpm | maior pausa: {rr.max():.1f} s")
print(f"Batimentos normais: {cont.get('R', 0) + cont.get('N', 0)} | extrassístoles atriais: {cont.get('A', 0)}")
