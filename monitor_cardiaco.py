# Monitor
# Executar:  python monitor_cardiaco.py
# Teclas:    → / ← = botão Anterior/Próximo | M = botão demutar alarme | A = botão de analisar

import os
import time
import queue
import threading

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import nucleo_ecg as nu

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None

# config
BAUD = 115200
JANELA_S = 6
DUR_GRAVACAO = 15
QUADRO_MS = 40
SIMULACAO = "Simulação sem ESP32"

NOMES_TELA = nu.NOMES

PRETO, PAINEL, BORDA = "#000000", "#0d1117", "#263238"
COR_ECG, COR_SPO2, COR_PA, COR_FR, COR_TEMP, COR_TXT = "#33ff66", "#29d3ff", "#ff6b6b", "#ffd23f", "#eceff1", "#b0bec5"
FONTE = "Segoe UI"


#comunicacao com o ESP

class ConexaoSerial:
    def __init__(self, porta, fila):
        self.fila = fila
        self.ser = serial.Serial()
        self.ser.port = porta
        self.ser.baudrate = BAUD
        self.ser.timeout = 0.1
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.lock = threading.Lock()
        self.ativo = True
        threading.Thread(target=self._ler, daemon=True).start()

    def _ler(self):
        while self.ativo:
            try:
                dados = self.ser.readline()
            except Exception:
                if self.ativo:
                    self.fila.put("__ERRO__")
                break
            if dados:
                linha = dados.decode("utf-8", errors="ignore").strip()
                if linha:
                    self.fila.put(linha)

    def enviar(self, msg):
        with self.lock:
            try:
                self.ser.write((msg + "\n").encode())
            except Exception:
                self.fila.put("__ERRO__")

    def fechar(self):
        self.ativo = False
        try:
            self.ser.close()
        except Exception:
            pass

#interface

class Monitor:
    def __init__(self, root):
        self.root = root
        root.title("Monitor - Trainee RAITec")
        root.configure(bg=PRETO)
        root.geometry("1320x800")
        root.minsize(1100, 680)

        self.banco = nu.BancoTrechos()
        self.gerador = nu.GeradorECG(self.banco)
        self.analisador = nu.Analisador()
        self.registros = nu.Registros()

        self.fila = queue.Queue()
        self.conexao = None
        self.cond_idx = 0
        self.nivel_alarme = 0
        self.nome_alarme = ""
        self.mudo = False
        self.gravando = False
        self.amostra = []
        self.fim_gravacao = 0.0

        self.N = JANELA_S * nu.FS
        self.buf_ecg = np.full(self.N, np.nan)
        self.buf_pl = np.full(self.N, np.nan)
        self.pos = 0
        self.t0 = time.perf_counter()
        self.consumidas = 0
        self.amostra_global = 0
        self.picos_r = []
        self.vitais = {}
        self.prox_vitais = 0.0
        self.fundo = None

        self._estilo()
        self._layout()
        self._atualizar_portas()
        self._definir_condicao(0, enviar=False)
        self._atualizar_historico()

        root.bind("<Right>", lambda e: self._definir_condicao((self.cond_idx + 1) % len(nu.CONDICOES), True))
        root.bind("<Left>", lambda e: self._definir_condicao((self.cond_idx - 1) % len(nu.CONDICOES), True))
        root.bind("<KeyPress-m>", lambda e: self._mudo())
        root.bind("<KeyPress-M>", lambda e: self._mudo())
        root.bind("<KeyPress-a>", lambda e: self._analisar())
        root.bind("<KeyPress-A>", lambda e: self._analisar())
        root.protocol("WM_DELETE_WINDOW", self._fechar)
        self._tick()

    # montagem da tela
    def _estilo(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=PRETO, foreground=COR_TXT, font=(FONTE, 10))
        st.configure("TFrame", background=PRETO)
        st.configure("Painel.TFrame", background=PAINEL)
        st.configure("TLabel", background=PRETO, foreground=COR_TXT)
        st.configure("TNotebook", background=PRETO, borderwidth=0)
        st.configure("TNotebook.Tab", background=PAINEL, foreground=COR_TXT, padding=(14, 6))
        st.map("TNotebook.Tab", background=[("selected", BORDA)], foreground=[("selected", "white")])
        st.configure("TButton", background=BORDA, foreground="white", padding=(10, 5))
        st.map("TButton", background=[("active", "#37474f")])
        st.configure("TCombobox", fieldbackground=PAINEL, background=BORDA, foreground="white")
        st.configure("Treeview", background=PAINEL, fieldbackground=PAINEL, foreground=COR_TXT, rowheight=24)
        st.configure("Treeview.Heading", background=BORDA, foreground="white")
        st.map("Treeview", background=[("selected", "#1e88e5")])

    def _layout(self):
        topo = ttk.Frame(self.root, padding=(10, 8))
        topo.pack(fill="x")
        ttk.Label(topo, text="Porta:").pack(side="left")
        self.cb_porta = ttk.Combobox(topo, width=28, state="readonly")
        self.cb_porta.pack(side="left", padx=6)
        ttk.Button(topo, text="Atualizar", command=self._atualizar_portas).pack(side="left")
        self.bt_conectar = ttk.Button(topo, text="Conectar", command=self._conectar)
        self.bt_conectar.pack(side="left", padx=6)
        self.lb_status = ttk.Label(topo, text="Modo simulação: use as teclas ← → M A")
        self.lb_status.pack(side="left", padx=12)
        ttk.Label(topo, text=f"Dados: {self.banco.origem}").pack(side="right")

        abas = ttk.Notebook(self.root)
        abas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        aba_mon = ttk.Frame(abas)
        aba_hist = ttk.Frame(abas, padding=8)
        abas.add(aba_mon, text="  Monitor  ")
        abas.add(aba_hist, text="  Histórico de análises  ")

        self.banner = tk.Label(aba_mon, text="", font=(FONTE, 20, "bold"), fg="white", bg="#1b5e20", pady=8)
        self.banner.pack(fill="x", pady=(6, 6))

        ttk.Label(aba_mon, text=f"Classificador: {self.analisador.metodo}").pack(side="bottom", anchor="w")
        base = ttk.Frame(aba_mon, padding=(0, 6))
        base.pack(side="bottom", fill="x")
        grade = tk.Frame(base, bg=PRETO)
        grade.pack(side="left")
        self.bts_cond = []
        for i, c in enumerate(nu.CONDICOES):
            b = tk.Button(grade, text=NOMES_TELA[c["codigo"]], width=22, relief="flat", fg="white", bg=BORDA,
                          activebackground="#455a64", activeforeground="white",
                          command=lambda k=i: self._definir_condicao(k, True))
            b.grid(row=i // 4, column=i % 4, padx=2, pady=2)
            self.bts_cond.append(b)
        acoes = tk.Frame(base, bg=PRETO)
        acoes.pack(side="right")
        tk.Button(acoes, text="Silenciar alarme (M)", width=18, relief="flat", bg="#5d4037", fg="white",
                  command=self._mudo).pack(side="left", padx=3)
        tk.Button(acoes, text="Analisar 15 s (A)", width=16, relief="flat", bg="#1565c0", fg="white",
                  command=self._analisar).pack(side="left", padx=3)
        tk.Button(acoes, text="Carregar amostra", width=16, relief="flat", bg="#37474f", fg="white",
                  command=self._carregar_amostra).pack(side="left", padx=3)

        centro = ttk.Frame(aba_mon)
        centro.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(9, 5), facecolor=PRETO)
        self.ax_ecg = self.fig.add_axes([0.01, 0.32, 0.98, 0.64])
        self.ax_pl = self.fig.add_axes([0.01, 0.03, 0.98, 0.25])
        x = np.arange(self.N) / nu.FS
        for ax, lim in ((self.ax_ecg, (-0.8, 1.3)), (self.ax_pl, (-0.15, 1.25))):
            ax.set_facecolor(PRETO)
            ax.set_xlim(0, JANELA_S)
            ax.set_ylim(*lim)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(BORDA)
        self.txt_ecg = self.ax_ecg.text(0.01, 0.92, "ECG  II", color=COR_ECG, transform=self.ax_ecg.transAxes, fontsize=11)
        self.ax_pl.text(0.01, 0.8, "PLETH", color=COR_SPO2, transform=self.ax_pl.transAxes, fontsize=10)
        self.l_ecg, = self.ax_ecg.plot(x, self.buf_ecg, color=COR_ECG, lw=1.6, animated=True)
        self.l_pl, = self.ax_pl.plot(x, self.buf_pl, color=COR_SPO2, lw=1.4, animated=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=centro)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.canvas.mpl_connect("draw_event", self._capturar_fundo)

        lado = tk.Frame(centro, bg=PRETO, width=300)
        lado.pack(side="right", fill="y", padx=(8, 0))
        lado.pack_propagate(False)
        self.lb_fc = self._vital(lado, "FC", "bpm", COR_ECG, 52)
        self.lb_spo2 = self._vital(lado, "SpO₂", "%", COR_SPO2, 40)
        self.lb_pa = self._vital(lado, "PA (NIBP)", "mmHg", COR_PA, 30)
        self.lb_fr = self._vital(lado, "FR", "rpm", COR_FR, 30)
        self.lb_temp = self._vital(lado, "TEMP", "°C", COR_TEMP, 26)

        caixa = tk.Frame(lado, bg=PAINEL, highlightbackground=BORDA, highlightthickness=1)
        caixa.pack(fill="both", expand=True, pady=(6, 0))
        tk.Label(caixa, text="ÚLTIMA ANÁLISE (ML)", bg=PAINEL, fg=COR_TXT, font=(FONTE, 9, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
        self.lb_analise = tk.Label(caixa, text="Nenhuma análise ainda.\nAperte \"Analisar\" para gravar 15 s.",
                                   bg=PAINEL, fg="white", font=(FONTE, 10), justify="left", anchor="nw", wraplength=280)
        self.lb_analise.pack(fill="both", expand=True, padx=8, pady=6)

        colunas = ("id", "data", "origem", "painel", "resultado", "fc", "bat", "pv")
        titulos = ("ID", "Data/hora", "Origem", "Condição no painel", "Resultado da análise", "FC", "Batimentos", "% V")
        larguras = (50, 150, 230, 190, 200, 70, 90, 70)
        self.tree = ttk.Treeview(aba_hist, columns=colunas, show="headings")
        for c, t, w in zip(colunas, titulos, larguras):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._ver_registro())
        barra = ttk.Frame(aba_hist, padding=(0, 6))
        barra.pack(fill="x")
        ttk.Button(barra, text="Atualizar", command=self._atualizar_historico).pack(side="left")
        ttk.Button(barra, text="Ver sinal do registro", command=self._ver_registro).pack(side="left", padx=6)
        ttk.Label(barra, text=f"Banco: {nu.ARQ_BANCO}").pack(side="right")

    def _vital(self, pai, titulo, unidade, cor, tamanho):
        f = tk.Frame(pai, bg=PAINEL, highlightbackground=BORDA, highlightthickness=1)
        f.pack(fill="x", pady=3)
        topo = tk.Frame(f, bg=PAINEL)
        topo.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(topo, text=titulo, bg=PAINEL, fg=cor, font=(FONTE, 10, "bold")).pack(side="left")
        tk.Label(topo, text=unidade, bg=PAINEL, fg=cor, font=(FONTE, 9)).pack(side="right")
        lb = tk.Label(f, text="--", bg=PAINEL, fg=cor, font=(FONTE, tamanho, "bold"))
        lb.pack(anchor="e", padx=10)
        return lb

    # desenho do ECG
    def _capturar_fundo(self, event=None):
        self.fundo = self.canvas.copy_from_bbox(self.fig.bbox)
        self._desenhar_linhas()

    def _desenhar_linhas(self):
        if self.fundo is None:
            return
        self.canvas.restore_region(self.fundo)
        self.ax_ecg.draw_artist(self.l_ecg)
        self.ax_pl.draw_artist(self.l_pl)
        self.canvas.blit(self.fig.bbox)

    def _escrever(self, ecg, pleth):
        n = len(ecg)
        idx = (self.pos + np.arange(n)) % self.N
        self.buf_ecg[idx] = ecg
        self.buf_pl[idx] = pleth
        self.pos = (self.pos + n) % self.N
        vao = (self.pos + np.arange(int(0.12 * nu.FS))) % self.N
        self.buf_ecg[vao] = np.nan
        self.buf_pl[vao] = np.nan
        self.l_ecg.set_ydata(self.buf_ecg)
        self.l_pl.set_ydata(self.buf_pl)

    
    def _tick(self):
        self._processar_serial()
        agora = time.perf_counter()
        devidas = int((agora - self.t0) * nu.FS) - self.consumidas
        if devidas > nu.FS:
            self.consumidas += devidas - nu.FS // 10
            devidas = nu.FS // 10
        if devidas > 0:
            ecg, pleth, marcas = self.gerador.proximas(devidas)
            self.consumidas += devidas
            self._escrever(ecg, pleth)
            for k, m in enumerate(marcas):
                if m:
                    self._batida(m, self.amostra_global + k)
            if self.gravando:
                self.amostra.extend(ecg)
            self.amostra_global += devidas
        self._desenhar_linhas()
        self._atualizar_vitais(agora)
        self._atualizar_banner(agora)
        if self.gravando and agora >= self.fim_gravacao:
            self._finalizar_gravacao()
        self.root.after(QUADRO_MS, self._tick)

    def _batida(self, marca, idx):
        self.picos_r.append(idx)
        del self.picos_r[:-10]
        if self.conexao:
            self.conexao.enviar("BEAT:V" if marca == "V" else "BEAT")

    # parametros
    def _sortear_vitais(self):
        c = nu.CONDICOES[self.cond_idx]
        self.vitais = dict(spo2=c["spo2"], sis=c["pa"][0] if c["pa"] else None,
                           dia=c["pa"][1] if c["pa"] else None, fr=c["fr"], temp=c["temp"])

    def _atualizar_vitais(self, agora):
        c = nu.CONDICOES[self.cond_idx]
        if self.picos_r and len(self.picos_r) >= 3 and self.amostra_global - self.picos_r[-1] < 2.5 * nu.FS:
            rr = np.diff(self.picos_r[-8:])
            self.lb_fc.config(text=f"{60 * nu.FS / np.mean(rr):.0f}")
        elif c["codigo"] == "ASY":
            self.lb_fc.config(text="0")
        else:
            self.lb_fc.config(text="---")

        if agora < self.prox_vitais:
            return
        self.prox_vitais = agora + 1.0
        rng = np.random.default_rng()
        v = self.vitais
        if c["spo2"] is None:
            self.lb_spo2.config(text="---"); self.lb_pa.config(text="---/---")
            self.lb_fr.config(text="---")
        else:
            v["spo2"] = int(np.clip(v["spo2"] + rng.integers(-1, 2), c["spo2"] - 2, min(100, c["spo2"] + 1)))
            v["sis"] = int(np.clip(v["sis"] + rng.integers(-2, 3), c["pa"][0] - 6, c["pa"][0] + 6))
            v["dia"] = int(np.clip(v["dia"] + rng.integers(-2, 3), c["pa"][1] - 5, c["pa"][1] + 5))
            v["fr"] = int(np.clip(v["fr"] + rng.integers(-1, 2), c["fr"] - 2, c["fr"] + 2))
            self.lb_spo2.config(text=f"{v['spo2']}")
            self.lb_pa.config(text=f"{v['sis']}/{v['dia']}")
            self.lb_fr.config(text=f"{v['fr']}")
        v["temp"] = float(np.clip(v["temp"] + rng.normal(0, 0.03), c["temp"] - 0.2, c["temp"] + 0.2))
        self.lb_temp.config(text=f"{v['temp']:.1f}")

    def _atualizar_banner(self, agora):
        nome = NOMES_TELA[nu.CONDICOES[self.cond_idx]["codigo"]].upper()
        if self.gravando:
            resta = max(0, int(np.ceil(self.fim_gravacao - agora)))
            texto, cor = f"●  GRAVANDO AMOSTRA PARA ANÁLISE — {resta} s   (alarme sonoro pausado)", "#1565c0"
        elif self.nivel_alarme == 0:
            texto, cor = nome, "#37474f"
        elif self.mudo:
            texto, cor = f"{self.nome_alarme.upper()}   —   ALARME SILENCIADO", "#424242"
        else:
            prefixo = {1: "ALARME", 2: "ALARME CRÍTICO", 3: "EMERGÊNCIA"}[self.nivel_alarme]
            cor = {1: "#e65100", 2: "#c62828", 3: "#b71c1c"}[self.nivel_alarme]
            if self.nivel_alarme >= 2 and int(agora * 3) % 2 == 0:
                cor = "#000000"
            texto = f"{prefixo}: {self.nome_alarme.upper()}"
        if self.banner.cget("text") != texto or self.banner.cget("bg") != cor:
            self.banner.config(text=texto, bg=cor)

    # acoes
    def _definir_condicao(self, idx, enviar):
        self.cond_idx = idx
        c = nu.CONDICOES[idx]
        self.gerador.definir(idx)
        baixo, alto = self.gerador.faixa
        margem = 0.15 * (alto - baixo)
        self.ax_ecg.set_ylim(baixo - margem, alto + margem)
        origem = "sintético" if self.gerador.registro == "sintético" else f"MIT-BIH registro {self.gerador.registro}"
        self.txt_ecg.set_text(f"ECG  II  —  {origem}")
        self.buf_ecg[:] = np.nan
        self.buf_pl[:] = np.nan
        self.canvas.draw_idle()
        self.nivel_alarme = 0
        self.nome_alarme = NOMES_TELA[c["codigo"]]
        self.mudo = False
        self.picos_r.clear()
        self._sortear_vitais()
        self.prox_vitais = 0.0
        if self.gravando:
            self._cancelar_gravacao("condição alterada durante a gravação")
        if enviar and self.conexao:
            self.conexao.enviar(f"COND:{idx}")
        for k, b in enumerate(self.bts_cond):
            b.config(bg="#2e7d32" if k == idx else BORDA)

    def _mudo(self):
        if self.conexao:
            self.conexao.enviar("MUTE")
        else:
            self.mudo = not self.mudo

    def _analisar(self):
        if self.gravando:
            return
        self._iniciar_gravacao(enviar=True)

    def _iniciar_gravacao(self, enviar):
        self.gravando = True
        self.amostra = []
        self.fim_gravacao = time.perf_counter() + DUR_GRAVACAO
        if enviar and self.conexao:
            self.conexao.enviar("REC")

    def _cancelar_gravacao(self, motivo):
        self.gravando = False
        self.amostra = []
        self.lb_analise.config(text=f"Gravação cancelada: {motivo}.")

    def _finalizar_gravacao(self):
        self.gravando = False
        sinal = np.array(self.amostra)
        self.amostra = []
        if len(sinal) < 5 * nu.FS:
            self.lb_analise.config(text="Amostra curta demais, tente de novo.")
            return
        res = self.analisador.analisar(sinal)
        self._aplicar_resultado(res, "botão Analisar (15 s)", sinal)

    def _carregar_amostra(self):
        caminho = filedialog.askopenfilename(title="Escolha a amostra de ECG (125 Hz)",
                                             filetypes=[("CSV ou TXT", "*.csv *.txt"), ("Todos", "*.*")])
        if not caminho:
            return
        try:
            sinal = nu.ler_amostra(caminho)
        except Exception as e:
            messagebox.showerror("Amostra inválida", str(e))
            return
        res = self.analisador.analisar(sinal)
        self._aplicar_resultado(res, f"arquivo: {os.path.basename(caminho)}", sinal)
        self._mostrar_sinal(sinal, f"Amostra carregada — {NOMES_TELA[res['codigo']]}")

    def _aplicar_resultado(self, res, origem, sinal):
        painel = NOMES_TELA[nu.CONDICOES[self.cond_idx]["codigo"]]
        rid = self.registros.salvar(origem, painel, res, sinal)
        resultado = NOMES_TELA[res["codigo"]]
        self.nivel_alarme = res["alarme"]
        self.nome_alarme = f"{resultado} (análise)"
        self.mudo = False
        p = res["pct"]
        confere = ""
        if origem.startswith("botão"):
            confere = "\nConfere com o painel" if res["codigo"] == nu.CONDICOES[self.cond_idx]["codigo"] else "\nDiferente do painel"
        self.lb_analise.config(text=(
            f"#{rid} — {resultado}{confere}\n\n"
            f"FC medida: {res['fc']:.0f} bpm\nVariação RR: {res['cv_rr']:.2f}\n"
            f"Batimentos analisados: {res['batimentos']}\n"
            f"N {p['N']:.0f}%   S {p['S']:.0f}%   V {p['V']:.0f}%\nF {p['F']:.0f}%   Q {p['Q']:.0f}%\n\n"
            f"Origem: {origem}"))
        if self.conexao:
            self.conexao.enviar(f"RESULT:{res['codigo']}:{res['alarme']}")
        self._atualizar_historico()

    # log
    def _atualizar_historico(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.registros.listar():
            rid, data, origem, painel, resultado, fc, bat, pv, arquivo = r
            self.tree.insert("", "end", iid=str(rid), values=(rid, data, origem, painel, resultado,
                                                             f"{fc:.0f}", bat, f"{pv:.0f}"))

    def _ver_registro(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Histórico", "Selecione um registro.")
            return
        rid = int(sel[0])
        linha = next(r for r in self.registros.listar() if r[0] == rid)
        try:
            sinal = np.loadtxt(linha[8])
        except Exception as e:
            messagebox.showerror("Histórico", f"Não foi possível abrir o sinal:\n{e}")
            return
        self._mostrar_sinal(sinal, f"Registro #{rid} — {linha[4]} ({linha[1]})")

    def _mostrar_sinal(self, sinal, titulo):
        jan = tk.Toplevel(self.root)
        jan.title(titulo)
        jan.configure(bg=PRETO)
        fig = Figure(figsize=(11, 3.6), facecolor=PRETO)
        ax = fig.add_subplot(111)
        t = np.arange(len(sinal)) / nu.FS
        s = np.asarray(sinal, float)
        s = nu.filtrar(s)
        picos = nu.detectar_picos_r(s)
        ax.plot(t, s, color=COR_ECG, lw=1.1)
        ax.plot(t[picos], s[picos], "o", color=COR_FR, ms=4, label="Picos R detectados")
        ax.set_facecolor(PRETO)
        ax.set_title(titulo, color="white")
        ax.set_xlabel("tempo (s)", color=COR_TXT)
        ax.tick_params(colors=COR_TXT)
        ax.legend(facecolor=PAINEL, labelcolor="white", loc="upper right")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=jan).get_tk_widget().pack(fill="both", expand=True)

    # serial
    def _atualizar_portas(self):
        portas = [p.device for p in list_ports.comports()] if serial else []
        self.cb_porta["values"] = [SIMULACAO] + portas
        if not self.cb_porta.get():
            self.cb_porta.set(portas[-1] if portas else SIMULACAO)

    def _conectar(self):
        if self.conexao:
            self._desconectar("desconectado")
            return
        porta = self.cb_porta.get()
        if porta == SIMULACAO:
            self.lb_status.config(text="Modo simulação: use as teclas ← → M A")
            return
        if serial is None:
            messagebox.showerror("pyserial", "Instale a biblioteca: pip install pyserial")
            return
        try:
            self.conexao = ConexaoSerial(porta, self.fila)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir {porta}.\nFeche o Monitor Serial do Arduino IDE.\n\n{e}")
            return
        self.bt_conectar.config(text="Desconectar")
        self.lb_status.config(text=f"Conectado em {porta} - aguardando o ESP32…")
        self.root.after(1500, self._sincronizar)

    def _sincronizar(self):
        if self.conexao:
            self.conexao.enviar(f"COND:{self.cond_idx}")
            self.conexao.enviar("PING")

    def _desconectar(self, motivo):
        if self.conexao:
            self.conexao.fechar()
            self.conexao = None
        self.bt_conectar.config(text="Conectar")
        self.lb_status.config(text=f"Simulação ({motivo}): use as teclas ← → M A")

    def _processar_serial(self):
        while True:
            try:
                linha = self.fila.get_nowait()
            except queue.Empty:
                return
            if linha == "__ERRO__":
                self._desconectar("conexão perdida")
            elif linha.startswith("COND:"):
                partes = linha.split(":")
                if len(partes) > 1 and partes[1].isdigit() and int(partes[1]) < len(nu.CONDICOES):
                    self._definir_condicao(int(partes[1]), enviar=False)
            elif linha.startswith("MUTE:"):
                self.mudo = not linha.endswith("0")
            elif linha.startswith("MODE:") and self.conexao:
                modo = "som sincronizado com a tela" if linha.endswith("PC") else "ESP32 gerando o som sozinho"
                self.lb_status.config(text=f"Conectado em {self.conexao.ser.port} - {modo}")
            elif linha == "PONG" and self.conexao:
                self.lb_status.config(text=f"Conectado em {self.conexao.ser.port} - ESP32 respondendo")
            elif linha == "REC:START":
                if not self.gravando:
                    self._iniciar_gravacao(enviar=False)
            elif linha == "REC:CANCEL":
                if self.gravando:
                    self._cancelar_gravacao("cancelada no painel físico")

    def _fechar(self):
        self._desconectar("encerrando")
        self.root.destroy()


if __name__ == "__main__":
    raiz = tk.Tk()
    Monitor(raiz)
    raiz.mainloop()
