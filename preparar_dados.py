import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from nucleo_ecg import FS, filtrar, recortar_batimentos, atributos_rr

PASTA_DATASET = r"C:\Users\raite\OneDrive\Desktop\archive"

FS_ORIGINAL = 360
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASSES = ["N", "S", "V", "F", "Q"]
MARCA_PASSO = {"102", "104", "107", "217"}
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]


def ler_registro(pasta, rec):
    df = pd.read_csv(os.path.join(pasta, f"{rec}.csv"))
    df.columns = [c.strip().strip("'") for c in df.columns]
    if "MLII" not in df.columns:
        return None
    mv = (df["MLII"].values.astype(float) - 1024.0) / 200.0
    sinal = resample_poly(mv, FS, FS_ORIGINAL)
    return filtrar(sinal)


def ler_anotacoes(pasta, rec):
    batidas, ritmos = [], []
    with open(os.path.join(pasta, f"{rec}annotations.txt")) as f:
        next(f)
        for linha in f:
            p = linha.split()
            if len(p) < 6:
                continue
            amostra = int(round(int(p[1]) * FS / FS_ORIGINAL))
            tipo = p[2]
            aux = p[6] if len(p) > 6 else ""
            if tipo == "+" and aux.startswith("("):
                ritmos.append((amostra, aux))
            else:
                batidas.append((amostra, tipo))
    return batidas, ritmos


def ajustar_picos(sinal, posicoes, raio=4):
    out = []
    for p in posicoes:
        a, b = max(0, p - raio), min(len(sinal), p + raio + 1)
        out.append(a + int(np.argmax(np.abs(sinal[a:b]))))
    return np.array(out, int)


def ritmo_em(ritmos, amostra):
    atual = None
    for t, r in ritmos:
        if t > amostra:
            break
        atual = r
    return atual


def melhor_janela(pasta, recs, ritmo, dur_s, criterio):
    melhor = None
    for rec in recs:
        sinal = ler_registro(pasta, rec)
        if sinal is None:
            continue
        batidas, ritmos = ler_anotacoes(pasta, rec)
        bat = [(t, s) for t, s in batidas if s in AAMI]
        if len(bat) < 10:
            continue
        pos = ajustar_picos(sinal, [t for t, _ in bat])
        tipos = [AAMI[s] for _, s in bat]
        n = int(dur_s * FS)
        for i in range(0, len(pos) - 5, 3):
            j = np.searchsorted(pos, pos[i] + n)
            if j >= len(pos):
                break
            if any(ritmo_em(ritmos, p) != ritmo for p in pos[i:j + 1]):
                continue
            nota = criterio(pos[i:j + 1], tipos[i:j])
            if nota is not None and (melhor is None or nota > melhor[0]):
                melhor = (nota, rec, pos[i], pos[j], pos[i:j] - pos[i], tipos[i:j], sinal)
    if melhor is None:
        raise RuntimeError(f"Nenhuma janela encontrada para {ritmo}")
    nota, rec, ini, fim, picos, tipos, sinal = melhor
    return dict(sinal=sinal[ini:fim].astype(np.float32), picos=np.array(picos, np.int32),
                tipos=np.array(["V" if t == "V" else "N" for t in tipos]), registro=rec)


def ajustar_batimento(seg, n):
    L = len(seg)
    if n >= L:
        k = int(0.62 * L)
        return np.concatenate([seg[:k], np.full(n - L, seg[k]), seg[k:]])
    fim = int(0.22 * L)
    return np.concatenate([seg[: n - fim], seg[L - fim:]])


def construir_ritmo(base, bpm, variacao, dur_s, seed=0):
    rng = np.random.default_rng(seed)
    sinal, picos = base["sinal"], list(base["picos"]) + [len(base["sinal"])]
    batimentos = [sinal[picos[k]:picos[k + 1]] for k in range(len(picos) - 1)]
    out, novos, k = [], [], 0
    total = 0
    while total < dur_s * FS:
        n = int(60 / bpm * (1 + rng.uniform(-variacao, variacao)) * FS)
        novos.append(total)
        out.append(ajustar_batimento(batimentos[k % len(batimentos)], n))
        total += n
        k += 1
    return dict(sinal=np.concatenate(out).astype(np.float32), picos=np.array(novos, np.int32),
                tipos=np.array(["N"] * len(novos)), registro=base["registro"] + " (FC ajustada)")


def episodio_mais_longo(pasta, rec, ritmo):
    sinal = ler_registro(pasta, rec)
    batidas, ritmos = ler_anotacoes(pasta, rec)
    ritmos_fim = ritmos[1:] + [(len(sinal), None)]
    ini, fim = max(((a, b) for (a, r), (b, _) in zip(ritmos, ritmos_fim) if r == ritmo), key=lambda x: x[1] - x[0])
    bat = [(t, s) for t, s in batidas if s in AAMI and ini <= t < fim]
    if bat:
        pos = ajustar_picos(sinal, [t for t, _ in bat])
        ini_c, fim_c = pos[0], pos[-1]
        picos = pos[:-1] - ini_c
        tipos = ["V" if AAMI[s] == "V" else "N" for _, s in bat[:-1]]
    else:
        ini_c, fim_c, picos, tipos = ini, fim, [], []
    return dict(sinal=sinal[ini_c:fim_c].astype(np.float32), picos=np.array(picos, np.int32),
                tipos=np.array(tipos), registro=rec)


def gerar_monitor(pasta, saida):
    def so_normais(p, t):
        if any(x != "N" for x in t):
            return None
        rr = np.diff(p) / FS
        return -np.std(rr) / np.mean(rr) - abs(60 / np.mean(rr) - 75) / 100

    def taquicardia(p, t):
        if any(x != "N" for x in t):
            return None
        rr = np.diff(p) / FS
        if np.std(rr) / np.mean(rr) > 0.08:
            return None
        return float(np.mean(60 / rr))

    def fa(p, t):
        if any(x != "N" for x in t):
            return None
        return float(np.std(np.diff(p)) / np.mean(np.diff(p)))

    def extrassistole(p, t):
        frac = np.mean([x == "V" for x in t])
        if not 0.1 <= frac <= 0.3:
            return None
        if any(t[k] == "V" and t[k + 1] == "V" for k in range(len(t) - 1)):
            return None
        return -abs(frac - 0.2)

    print("Procurando trechos para o monitor (leva alguns minutos)...")
    trechos = {
        "NSR": melhor_janela(pasta, DS2, "(N", 40, so_normais),
        "TAQ": melhor_janela(pasta, DS2, "(N", 30, taquicardia),
        "AFIB": melhor_janela(pasta, ["202", "210", "219", "221", "222"], "(AFIB", 40, fa),
        "PVC": melhor_janela(pasta, ["200", "213", "214", "221", "228", "233"], "(N", 40, extrassistole),
        "VT": episodio_mais_longo(pasta, "223", "(VT"),
        "VF": episodio_mais_longo(pasta, "207", "(VFL"),
    }
    trechos["BRAD"] = construir_ritmo(trechos["NSR"], 44, 0.04, 40)
    if 60 * FS / np.mean(np.diff(trechos["TAQ"]["picos"])) < 110:
        trechos["TAQ"] = construir_ritmo(trechos["NSR"], 130, 0.02, 30)
    dados = {}
    for cod, t in trechos.items():
        dados[f"{cod}_sinal"] = t["sinal"]
        dados[f"{cod}_picos"] = t["picos"]
        dados[f"{cod}_tipos"] = t["tipos"]
        dados[f"{cod}_registro"] = np.array(t["registro"])
        fc = 60 * FS / np.mean(np.diff(t["picos"])) if len(t["picos"]) > 1 else 0
        print(f"  {cod:5} registro {t['registro']}  {len(t['sinal']) / FS:5.1f} s  FC média {fc:5.0f} bpm  "
              f"batimentos V: {int(np.sum(t['tipos'] == 'V')) if len(t['tipos']) else 0}")
    np.savez_compressed(saida, **dados)
    print(f"Salvo: {saida}")


def gerar_batimentos(pasta, saida):
    recs = sorted({os.path.basename(f)[:3] for f in glob.glob(os.path.join(pasta, "*annotations.txt"))} - MARCA_PASSO)
    print("Recortando batimentos para o treino da IA...")
    X, y, registro, rr = [], [], [], []
    for rec in recs:
        sinal = ler_registro(pasta, rec)
        if sinal is None:
            print(f"  {rec}: sem derivação MLII, ignorado")
            continue
        batidas, _ = ler_anotacoes(pasta, rec)
        todas = [t for t, s in batidas if s in AAMI]
        pos = ajustar_picos(sinal, todas)
        rotulos = [CLASSES.index(AAMI[s]) for t, s in batidas if s in AAMI]
        segs, validos = recortar_batimentos(sinal, pos)
        feats = atributos_rr(pos, validos)
        usados = 0
        for seg, k, f in zip(segs, validos, feats):
            if 0 < k < len(pos) - 1:
                X.append(seg)
                y.append(rotulos[k])
                registro.append(int(rec))
                rr.append(f)
                usados += 1
        print(f"  {rec}: {usados} batimentos")
    np.savez_compressed(saida, X=np.array(X, np.float16), y=np.array(y, np.int8),
                        registro=np.array(registro, np.int16), rr=np.array(rr, np.float32),
                        classes=np.array(CLASSES))
    cont = np.bincount(y, minlength=5)
    print("Salvo:", saida, "| total", len(y), "|", dict(zip(CLASSES, cont.tolist())))


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else PASTA_DATASET
    if not os.path.exists(os.path.join(pasta, "100.csv")):
        sys.exit(f"Não achei os arquivos do MIT-BIH em: {pasta}")
    destino = os.path.dirname(os.path.abspath(__file__))
    gerar_monitor(pasta, os.path.join(destino, "ecg_monitor.npz"))
    gerar_batimentos(pasta, os.path.join(destino, "batimentos_mitbih.npz"))
