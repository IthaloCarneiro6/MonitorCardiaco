
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

# config
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQ_TRECHOS = os.path.join(PASTA, "ecg_monitor.npz")
ARQ_MODELO = os.path.join(PASTA, "modelo_ecg.keras")
ARQ_BANCO = os.path.join(PASTA, "registros.db")
PASTA_AMOSTRAS = os.path.join(PASTA, "amostras")

FS = 125
ANTES_R = 40
DEPOIS_R = 88
TAM_BATIMENTO = ANTES_R + DEPOIS_R
CLASSES = ["N", "S", "V", "F", "Q"]

LIMIAR_LARGURA_V = 7
LIMIAR_FV = 0.8
LIMIAR_IRREGULAR = 0.12

# condicoes cardiacas
CONDICOES = [
    dict(codigo="NSR",  bpm=75,  alarme=0, spo2=98,   pa=(120, 80), fr=16,   temp=36.6),
    dict(codigo="BRAD", bpm=42,  alarme=1, spo2=96,   pa=(100, 62), fr=14,   temp=36.3),
    dict(codigo="TAQ",  bpm=135, alarme=1, spo2=97,   pa=(132, 86), fr=22,   temp=37.4),
    dict(codigo="AFIB", bpm=110, alarme=1, spo2=95,   pa=(112, 74), fr=18,   temp=36.7),
    dict(codigo="PVC",  bpm=78,  alarme=1, spo2=97,   pa=(118, 78), fr=16,   temp=36.6),
    dict(codigo="VT",   bpm=180, alarme=2, spo2=88,   pa=(78, 48),  fr=28,   temp=36.5),
    dict(codigo="VF",   bpm=0,   alarme=3, spo2=None, pa=None,      fr=None, temp=36.4),
    dict(codigo="ASY",  bpm=0,   alarme=3, spo2=None, pa=None,      fr=None, temp=36.2),
]
POR_CODIGO = {c["codigo"]: c for c in CONDICOES}
NOMES = {
    "NSR": "Ritmo Sinusal Normal", "BRAD": "Bradicardia Sinusal", "TAQ": "Taquicardia Sinusal",
    "AFIB": "Fibrilação Atrial", "PVC": "Extrassístole Ventricular", "VT": "Taquicardia Ventricular",
    "VF": "Fibrilação Ventricular", "ASY": "Assistolia",
}

_SOS = butter(3, [0.5, 40], btype="bandpass", fs=FS, output="sos")


# pré-processamento

def filtrar(sinal):
    """Passa-faixa 0,5–40 Hz: remove a oscilação da linha de base e ruídos de alta frequência."""
    sinal = np.asarray(sinal, float)
    if len(sinal) < 30:
        return sinal - np.median(sinal)
    return sosfiltfilt(_SOS, sinal)


def detectar_picos_r(sinal):
    """Detector de QRS inspirado no Pan-Tompkins: derivada -> quadrado -> integração (150 ms);
    o pico R é refinado no ponto de maior amplitude absoluta, então funciona com QRS invertido."""
    sinal = np.asarray(sinal, float)
    deriv = np.diff(sinal, prepend=sinal[0])
    janela = int(0.15 * FS)
    energia = np.convolve(deriv ** 2, np.ones(janela) / janela, mode="same")
    limiar = 0.3 * np.percentile(energia, 98)
    if limiar <= 1e-9:
        return np.array([], int)
    candidatos, _ = find_peaks(energia, height=limiar, distance=int(0.25 * FS))
    raio = int(0.08 * FS)
    picos = []
    for c in candidatos:
        a, b = max(0, c - raio), min(len(sinal), c + raio)
        picos.append(a + int(np.argmax(np.abs(sinal[a:b]))))
    picos = np.unique(picos)
    if len(picos) > 1:
        manter = [0]
        for k in range(1, len(picos)):
            if picos[k] - picos[manter[-1]] >= int(0.2 * FS):
                manter.append(k)
            elif abs(sinal[picos[k]]) > abs(sinal[picos[manter[-1]]]):
                manter[-1] = k
        picos = picos[manter]
    return np.asarray(picos, int)


def recortar_batimentos(sinal, picos):
    """Recorta cada batimento (0,32 s antes e 0,70 s depois do R) e normaliza (z-score).
    Devolve os batimentos e o índice de cada pico aproveitado."""
    segs, validos = [], []
    for k, p in enumerate(picos):
        a, b = p - ANTES_R, p + DEPOIS_R
        if a < 0 or b > len(sinal):
            continue
        s = np.asarray(sinal[a:b], float)
        segs.append((s - s.mean()) / (s.std() + 1e-6))
        validos.append(k)
    return (np.array(segs, np.float32) if segs else np.empty((0, TAM_BATIMENTO), np.float32)), validos


def atributos_rr(picos, validos):
    """Intervalos RR de cada batimento: anterior, posterior e a razão de cada um pela média local
    (10 batimentos em volta). Batimentos prematuros têm RR anterior curto e RR posterior longo."""
    picos = np.asarray(picos, float)
    rr = np.diff(picos) / FS
    if len(rr) == 0:
        return np.ones((len(validos), 4), np.float32)
    feats = []
    for k in validos:
        ant = rr[k - 1] if k - 1 >= 0 else rr[0]
        pos = rr[k] if k < len(rr) else rr[-1]
        a, b = max(0, k - 5), min(len(rr), k + 5)
        local = np.mean(rr[a:b])
        feats.append([ant, pos, ant / local, pos / local])
    return np.array(feats, np.float32)


def largura_qrs(sinal, p):
    """Largura do QRS (em amostras) na metade da amplitude do pico R."""
    meio = abs(sinal[p]) / 2
    esq = p
    while esq > 0 and p - esq < 25 and abs(sinal[esq]) > meio:
        esq -= 1
    dir_ = p
    while dir_ < len(sinal) - 1 and dir_ - p < 25 and abs(sinal[dir_]) > meio:
        dir_ += 1
    return dir_ - esq


# trechos de ECG para o monitor

def onda_pleth(n, amp=1.0):
    i = np.arange(n)
    return amp * (np.exp(-((i - 22) / 7.0) ** 2) + 0.35 * np.exp(-((i - 42) / 9.0) ** 2))


def _trecho_sintetico(codigo, rng):
    """Usado na assistolia (não existe no MIT-BIH) ou se o ecg_monitor.npz não estiver na pasta."""
    dur = 20 * FS
    t = np.arange(dur) / FS
    if codigo == "ASY":
        return 0.04 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.01, dur), np.array([], int), np.array([])
    if codigo == "VF":
        f = 5 + 1.5 * np.sin(2 * np.pi * 0.23 * t)
        amp = 0.2 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.37 * t))
        return amp * np.sin(2 * np.pi * np.cumsum(f) / FS), np.array([], int), np.array([])
    bpm = POR_CODIGO[codigo]["bpm"]
    i = np.arange(int(0.75 * FS))
    g = lambda c, w, a: a * np.exp(-((i - c) / w) ** 2)
    normal = g(10, 1.6, 1.0) + g(14, 1.6, -0.25) + g(40, 7, 0.25) + g(0, 3, 0.1)
    ventric = g(10, 5, 1.2) + g(22, 6, -0.5) + g(55, 10, -0.3)
    sinal, picos, tipos, pos, k = np.zeros(dur), [], [], 0, 1
    while pos < dur - len(i):
        v = codigo == "VT" or (codigo == "PVC" and k % 4 == 0)
        sinal[pos:pos + len(i)] += ventric if v else normal
        picos.append(pos + 10)
        tipos.append("V" if v else "N")
        var = rng.uniform(-0.3, 0.3) if codigo == "AFIB" else rng.uniform(-0.02, 0.02)
        pos += int(60 / bpm * (1 + var) * FS)
        k += 1
    return sinal + rng.normal(0, 0.01, dur), np.array(picos), np.array(tipos)


class BancoTrechos:
    def __init__(self, caminho=ARQ_TRECHOS, seed=None):
        self.rng = np.random.default_rng(seed)
        self.trechos = {}
        if os.path.exists(caminho):
            d = np.load(caminho, allow_pickle=False)
            for c in CONDICOES:
                cod = c["codigo"]
                if f"{cod}_sinal" in d:
                    self.trechos[cod] = dict(sinal=d[f"{cod}_sinal"].astype(float), picos=d[f"{cod}_picos"].astype(int),
                                             tipos=d[f"{cod}_tipos"], registro=str(d[f"{cod}_registro"]))
        reais = len(self.trechos)
        for c in CONDICOES:
            cod = c["codigo"]
            if cod not in self.trechos:
                s, p, t = _trecho_sintetico(cod, self.rng)
                self.trechos[cod] = dict(sinal=s, picos=p, tipos=t, registro="sintético")
        self.origem = f"MIT-BIH real ({reais} de 8 condições)" if reais else "sintético (ecg_monitor.npz não encontrado)"

    def preparar(self, codigo):
        """Monta o trecho em loop: ECG, pletismografia e a marcação de cada batimento (N ou V)."""
        tr = self.trechos[codigo]
        ecg = np.array(tr["sinal"], float)
        n = len(ecg)
        pleth = np.zeros(n)
        marcas = [None] * n
        picos = list(tr["picos"])
        for k, p in enumerate(picos):
            fim = picos[k + 1] if k + 1 < len(picos) else n
            v = tr["tipos"][k] == "V"
            amp = 0.3 if codigo == "VT" else (0.45 if v else 1.0)
            if codigo == "AFIB":
                amp *= self.rng.uniform(0.7, 1.0)
            pleth[p:fim] = onda_pleth(fim - p, amp)
            marcas[p] = "V" if v else "N"
        if codigo in ("VF", "ASY"):
            pleth = self.rng.normal(0, 0.01, n)
        if not picos:
            borda = min(20, n // 10)
            rampa = np.linspace(0, 1, borda)
            ecg[:borda] = ecg[:borda] * rampa + ecg[-1] * (1 - rampa)
        faixa = (float(np.percentile(ecg, 0.3)), float(np.percentile(ecg, 99.7)))
        if codigo == "ASY":
            faixa = (-0.5, 1.5)
        return ecg, pleth, marcas, faixa


class GeradorECG:
    """Toca o trecho da condição escolhida em loop, amostra por amostra."""

    def __init__(self, banco):
        self.banco = banco
        self.definir(0)

    def definir(self, idx):
        self.idx = idx
        self.codigo = CONDICOES[idx]["codigo"]
        self.ecg, self.pleth, self.marcas, self.faixa = self.banco.preparar(self.codigo)
        self.registro = self.banco.trechos[self.codigo]["registro"]
        self.pos = 0

    def proximas(self, n):
        idx = (self.pos + np.arange(n)) % len(self.ecg)
        self.pos = int((self.pos + n) % len(self.ecg))
        return self.ecg[idx].tolist(), self.pleth[idx].tolist(), [self.marcas[i] for i in idx]

# Analisador: modelo de ML (batimentos) + regras (ritmo)

class Analisador:
    def __init__(self, caminho_modelo=ARQ_MODELO):
        self.modelo = None
        self.metodo = "regra provisória (largura do QRS) — modelo não encontrado"
        if os.path.exists(caminho_modelo):
            try:
                import tensorflow as tf
                self.modelo = tf.keras.models.load_model(caminho_modelo)
                teste = [np.zeros((1, TAM_BATIMENTO, 1)), np.ones((1, 4))] if len(self.modelo.inputs) == 2 else np.zeros((1, TAM_BATIMENTO, 1))
                self.modelo.predict(teste, verbose=0)
                self.metodo = "CNN 1D (MIT-BIH) + regras de ritmo"
            except Exception as e:
                self.metodo = f"regra provisória — falha ao carregar o modelo ({type(e).__name__})"

    def classificar_batimentos(self, segs, sinal, picos, todos_picos, validos):
        if len(segs) == 0:
            return np.array([], int)
        if self.modelo is not None:
            entradas = [segs[..., None], atributos_rr(todos_picos, validos)] if len(self.modelo.inputs) == 2 else segs[..., None]
            return self.modelo.predict(entradas, verbose=0).argmax(axis=1)
        larguras = np.array([largura_qrs(sinal, p) for p in picos])
        return np.where(larguras >= LIMIAR_LARGURA_V, 2, 0)

    def analisar(self, sinal, ja_filtrado=False):
        sinal = np.asarray(sinal, float)
        if not ja_filtrado:
            sinal = filtrar(sinal)
        res = dict(fc=0.0, cv_rr=0.0, batimentos=0, pct={c: 0.0 for c in CLASSES}, metodo=self.metodo)
        amp = np.percentile(sinal, 99.5) - np.percentile(sinal, 0.5)
        if amp < 0.2:
            return self._final(res, "ASY")

        freqs = np.fft.rfftfreq(len(sinal), 1 / FS)
        pot = np.abs(np.fft.rfft(sinal)) ** 2
        conc = pot[(freqs >= 3) & (freqs <= 8)].sum() / (pot[(freqs >= 0.5) & (freqs <= 30)].sum() + 1e-12)
        res["conc_3_8hz"] = round(float(conc), 3)
        picos = detectar_picos_r(sinal)
        if len(picos) < 2:
            return self._final(res, "ASY")
        rr = np.diff(picos) / FS
        fc = 60.0 / np.median(rr)
        res["fc"] = round(float(fc), 1)

        segs, validos = recortar_batimentos(sinal, picos)
        classes = self.classificar_batimentos(segs, sinal, picos[validos], picos, validos)
        res["batimentos"] = int(len(classes))
        for k, nome in enumerate(CLASSES):
            res["pct"][nome] = round(100.0 * float(np.mean(classes == k)), 1) if len(classes) else 0.0
        frac_v = res["pct"]["V"] / 100.0

        normais = np.zeros(len(picos), bool)
        if len(classes):
            normais[np.array(validos, int)[classes == 0]] = True
        rr_nn = [rr[k] for k in range(len(rr)) if normais[k] and normais[k + 1]]
        cv = float(np.std(rr_nn) / np.mean(rr_nn)) if len(rr_nn) >= 4 else float(np.std(rr) / np.mean(rr))
        res["cv_rr"] = round(cv, 3)

        linha_base = float(np.mean(np.abs(sinal - np.median(sinal)) < 0.1 * amp))
        res["linha_base"] = round(linha_base, 2)
        caotico = fc > 150 and cv > 0.25 and linha_base < 0.55
        if (conc > LIMIAR_FV and fc > 180) or caotico:
            codigo = "VF"
        elif fc >= 100 and frac_v >= 0.5:
            codigo = "VT"
        elif frac_v >= 0.05 and np.sum(classes == 2) >= 2:
            codigo = "PVC"
        elif cv > LIMIAR_IRREGULAR:
            codigo = "AFIB"
        elif fc < 60:
            codigo = "BRAD"
        elif fc > 100:
            codigo = "TAQ"
        else:
            codigo = "NSR"
        return self._final(res, codigo)

    @staticmethod
    def _final(res, codigo):
        res.update(codigo=codigo, nome=NOMES[codigo], alarme=POR_CODIGO[codigo]["alarme"])
        return res

# registros das analises

class Registros:
    def __init__(self, caminho=ARQ_BANCO):
        os.makedirs(PASTA_AMOSTRAS, exist_ok=True)
        self.con = sqlite3.connect(caminho)
        self.con.execute("""CREATE TABLE IF NOT EXISTS analises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT, origem TEXT, condicao_painel TEXT,
            codigo TEXT, resultado TEXT, fc REAL, cv_rr REAL, batimentos INTEGER,
            pct_n REAL, pct_s REAL, pct_v REAL, pct_f REAL, pct_q REAL,
            alarme INTEGER, metodo TEXT, arquivo TEXT)""")
        self.con.commit()

    def salvar(self, origem, condicao_painel, res, sinal):
        agora = datetime.now()
        arquivo = os.path.join(PASTA_AMOSTRAS, f"amostra_{agora:%Y%m%d_%H%M%S}.csv")
        np.savetxt(arquivo, np.asarray(sinal, float), fmt="%.5f")
        p = res["pct"]
        cur = self.con.execute(
            """INSERT INTO analises (data_hora, origem, condicao_painel, codigo, resultado, fc, cv_rr,
               batimentos, pct_n, pct_s, pct_v, pct_f, pct_q, alarme, metodo, arquivo)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (agora.strftime("%d/%m/%Y %H:%M:%S"), origem, condicao_painel, res["codigo"], res["nome"],
             float(res["fc"]), float(res["cv_rr"]), int(res["batimentos"]), float(p["N"]), float(p["S"]),
             float(p["V"]), float(p["F"]), float(p["Q"]), int(res["alarme"]), res["metodo"], arquivo))
        self.con.commit()
        return cur.lastrowid

    def listar(self):
        return self.con.execute(
            """SELECT id, data_hora, origem, condicao_painel, resultado, fc, batimentos, pct_v, arquivo
               FROM analises ORDER BY id DESC""").fetchall()


def ler_amostra(caminho):
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        texto = f.read().replace(";", " ").replace(",", " ").replace("\t", " ")
    valores = []
    for item in texto.split():
        try:
            valores.append(float(item))
        except ValueError:
            pass
    valores = np.array(valores, float)
    valores = valores[~np.isnan(valores)]
    if len(valores) < 5 * FS:
        raise ValueError("A amostra precisa ter pelo menos 5 segundos (625 valores a 125 Hz).")
    return valores
