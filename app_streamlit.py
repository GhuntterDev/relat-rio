# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json, os, warnings, re
from typing import Dict, Any, Tuple, List
import altair as alt
import matplotlib.pyplot as plt
from matplotlib import ticker
from io import BytesIO
import zipfile
from datetime import datetime, date, timedelta
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime

# reportlab (instale com pip install reportlab se necessário)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def gerar_relatorio_ranking_setor_pdf(df_vendas, loja_nome, top_n, setores_escolhidos):
    """
    Gera um PDF com tabelas de ranking por setor.
    Cada tabela contém: Ranking, Código, Nome, Valor Atual (vazia)
    
    Args:
        df_vendas: DataFrame com colunas ['setor', 'codigo_base', 'nome', 'valor']
        loja_nome: Nome da loja para o cabeçalho
        top_n: Quantidade de produtos no ranking
        setores_escolhidos: Lista de setores para incluir
    
    Returns:
        bytes: Conteúdo do PDF gerado
    """
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        for setor in setores_escolhidos:
            # Filtra dados do setor
            df_setor = df_vendas[df_vendas["setor"] == setor].copy()
            
            if df_setor.empty:
                continue
            
            # Agrupa por código_base e calcula totais
            ranking = (
                df_setor.groupby(["codigo_base", "nome"], as_index=False)
                .agg(valor_total=("valor", "sum"))
                .sort_values("valor_total", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            
            # Adiciona coluna de ranking
            ranking.insert(0, "Ranking", range(1, len(ranking) + 1))
            
            # Adiciona coluna vazia "Valor Atual"
            ranking["Valor Atual"] = ""
            
            # Cria figura
            fig = plt.figure(figsize=(11, 8.5), dpi=150)
            ax = fig.add_subplot(111)
            ax.axis("off")
            
            # Cabeçalho
            fig.text(0.5, 0.95, f"Relatório de Ranking - Top {top_n}", 
                    ha="center", fontsize=16, fontweight="bold")
            fig.text(0.5, 0.92, f"Loja: {loja_nome}", 
                    ha="center", fontsize=12)
            fig.text(0.5, 0.89, f"Setor: {setor}", 
                    ha="center", fontsize=12, fontweight="bold")
            fig.text(0.5, 0.86, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                    ha="center", fontsize=10, color="#666666")
            
            # Prepara dados da tabela
            table_data = ranking[["Ranking", "codigo_base", "nome", "Valor Atual"]].values.tolist()
            
            # Cria tabela
            tb = Table(ax, bbox=[0.05, 0.05, 0.9, 0.75])
            
            # Cabeçalhos
            headers = ["Ranking", "Código", "Nome do Produto", "Valor Atual"]
            col_widths = [0.12, 0.18, 0.50, 0.20]
            
            for j, (header, width) in enumerate(zip(headers, col_widths)):
                cell = tb.add_cell(-1, j, width, 0.04, text=header, loc="center",
                                 facecolor="#1f2937", edgecolor="#000000")
                cell.get_text().set_color("#ffffff")
                cell.get_text().set_fontsize(10)
                cell.get_text().set_fontweight("bold")
                cell.set_linewidth(1.0)
            
            # Linhas de dados
            for i, row in enumerate(table_data):
                for j, (val, width) in enumerate(zip(row, col_widths)):
                    # Alinhamento: centro para ranking e código, esquerda para nome, centro para valor atual
                    loc = "center" if j in [0, 1, 3] else "left"
                    
                    # Trunca nome se muito longo
                    if j == 2 and len(str(val)) > 60:
                        val = str(val)[:57] + "..."
                    
                    cell = tb.add_cell(i, j, width, 0.035, text=str(val), loc=loc,
                                     facecolor="#ffffff" if i % 2 == 0 else "#f9fafb",
                                     edgecolor="#d1d5db")
                    cell.get_text().set_fontsize(9)
                    cell.set_linewidth(0.5)
            
            ax.add_table(tb)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def gerar_relatorio_todas_lojas_zip(df_vendas, top_n, setores_escolhidos):
    """
    Gera um ZIP contendo um PDF para cada loja.
    Cada PDF contém tabelas de ranking por setor.
    
    Args:
        df_vendas: DataFrame com colunas ['loja', 'setor', 'codigo_base', 'nome', 'valor']
        top_n: Quantidade de produtos no ranking
        setores_escolhidos: Lista de setores para incluir
    
    Returns:
        bytes: Conteúdo do ZIP gerado
    """
    zip_buffer = BytesIO()
    lojas = sorted(df_vendas["loja"].unique())
    
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for loja in lojas:
            df_loja = df_vendas[df_vendas["loja"] == loja]
            
            if df_loja.empty:
                continue
            
            # Gera PDF para esta loja
            pdf_bytes = gerar_relatorio_ranking_setor_pdf(
                df_loja, loja, top_n, setores_escolhidos
            )
            
            # Adiciona ao ZIP
            nome_arquivo = f"ranking_{loja.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            zf.writestr(nome_arquivo, pdf_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# =====================================================
# Configuração
# =====================================================
st.set_page_config(page_title="Painel de Vendas — Histórico", layout="wide")
st.title("📊 Painel de Vendas — Histórico como fonte principal")

st.markdown(
    "Use **CSV/Parquet** de **Histórico** (CSV sem cabeçalho):<br>"
    "- **A:** Data &nbsp;&nbsp;— **B:** Loja &nbsp;&nbsp;— **C:** Código - Nome &nbsp;&nbsp;— **D:** Quantidade &nbsp;&nbsp;— **E:** Valor<br>"
    "Setor é obtido via **produtos.json** (código→setor) e **agrupado** via **setores.json**.<br>"
    "Opcionais: CSV **mapeamento Pai/Filho** (2 colunas) e **códigos-alvo** (1 coluna).<br>"
    "Novo: **CSV de Metas por Loja/Setor** (A:loja, B:setor, C:meta, D:início, E:fim).",
    unsafe_allow_html=True
)

# =====================================================
# Utils
# =====================================================
def df_to_pdf_bytes_reportlab(df, title=None):
    """
    Recebe um pandas.DataFrame (df) e retorna um BytesIO contendo um PDF simples
    com a tabela. Colunas/linhas aparecem em texto — não é imagem/gráfico.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vazio — nada para exportar")

    # prepara buffer e documento
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)

    styles = getSampleStyleSheet()
    elements = []

    # título (opcional)
    if title:
        elements.append(Paragraph(str(title), styles['Heading2']))
        elements.append(Spacer(1, 8))

    # converte df para lista de listas (primeira linha = colunas)
    cols = list(df.columns)
    data = [cols] + df.fillna("").astype(str).values.tolist()

    # tenta definir larguras de coluna simples (distribui toda largura)
    page_width, page_height = A4
    usable_width = page_width - doc.leftMargin - doc.rightMargin
    # larguras proporcionais (ajusta colunas para caber)
    # por padrão dá mais espaço ao nome e menos ao rank
    col_weights = []
    # heurística simples: dar mais peso p/ colunas maiores de texto
    for c in cols:
        if c.lower() in ['nome', 'produto', 'descricao', 'description']:
            col_weights.append(3)
        elif c.lower() in ['setor', 'loja']:
            col_weights.append(1.5)
        else:
            col_weights.append(1)
    total_weight = sum(col_weights)
    col_widths = [usable_width * (w / total_weight) for w in col_weights]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f2f2f2')),
        ('TEXTCOLOR',(0,0),(-1,0), colors.black),
        ('ALIGN',(0,0),(-1, -1),'LEFT'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ])
    table.setStyle(style)
    elements.append(table)

    # build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer


def fmt_currency_br(x: float) -> str:
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def _safe_seek(file):
    try: file.seek(0)
    except Exception: pass

@st.cache_data
def read_metas_csv_robusto(file) -> pd.DataFrame:
    encodings = ["utf-8", "iso-8859-1", "windows-1252", "latin-1"]
    seps = [";", "\t", ",", "|", None]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                _safe_seek(file)
                df = pd.read_csv(
                    file, header=None, dtype=str, encoding=enc,
                    sep=sep, engine="python", on_bad_lines="skip", quoting=3
                )
                df = df.dropna(axis=1, how="all")
                if df.shape[1] < 5:
                    continue
                df = df.iloc[:, :5].rename(columns={
                    df.columns[0]: "loja",
                    df.columns[1]: "setor",
                    df.columns[2]: "meta_raw",
                    df.columns[3]: "inicio_raw",
                    df.columns[4]: "fim_raw",
                })
                meta = pd.to_numeric(
                    df["meta_raw"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                    errors="coerce"
                ).fillna(0.0)
                ini = parse_date_smart(df["inicio_raw"]).dt.date
                fim = parse_date_smart(df["fim_raw"]).dt.date
                out = pd.DataFrame({
                    "loja":  df["loja"].astype(str).str.strip(),
                    "setor": df["setor"].astype(str).str.strip(),
                    "meta":  meta.astype(float),
                    "inicio": ini,
                    "fim":    fim,
                })
                out = out.dropna(subset=["setor", "loja", "inicio", "fim"])
                return out
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Falha ao ler CSV de metas. Último erro: {last_err}")

@st.cache_data
def read_csv_hist_no_header(file):
    encodings = ["utf-8", "iso-8859-1", "windows-1252", "latin-1"]
    seps = [";", "\t", ",", "|", None]
    decimals = [",", "."]
    last_err = None
    for enc in encodings:
        for sep in seps:
            for dec in decimals:
                try:
                    _safe_seek(file)
                    df = pd.read_csv(
                        file, encoding=enc, sep=sep, header=None, dtype=str,
                        engine="python", decimal=dec, on_bad_lines="skip", quoting=3,
                    )
                    df = df.dropna(axis=1, how="all").dropna(how="all")
                    if df.shape[1] >= 5:
                        return df.iloc[:, :5]
                except Exception as e:
                    last_err = e
    raise RuntimeError(f"Falha ao ler histórico sem cabeçalho. Último erro: {last_err}")

def parse_date_smart(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    def _mask(rex: str, fmt: str):
        m = s.str.match(rex, na=False) & out.isna()
        if m.any():
            out.loc[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")

    _mask(r"^\d{1,2}/\d{1,2}/\d{4}$", "%d/%m/%Y")
    _mask(r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y")
    _mask(r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d")
    _mask(r"^\d{1,2}/\d{1,2}/\d{2}$", "%d/%m/%y")
    _mask(r"^\d{1,2}-\d{1,2}-\d{2}$", "%d-%m-%y")
    _mask(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$", "%d/%m/%Y %H:%M")
    _mask(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}$", "%d/%m/%Y %H:%M:%S")
    _mask(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", "%Y-%m-%dT%H:%M")
    _mask(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", "%Y-%m-%dT%H:%M:%S")

    left = out.isna()
    if left.any():
        num = pd.to_numeric(s[left].str.replace(",", ".", regex=False), errors="coerce")
        ok = num.notna() & (num >= 30000) & (num <= 80000)
        if ok.any():
            base = pd.to_datetime("1899-12-30")
            out.loc[left][ok] = base + pd.to_timedelta(num[ok], unit="D")

    left = out.isna()
    if left.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.loc[left] = pd.to_datetime(s[left], errors="coerce", dayfirst=True)

    return out

# --- PyInstaller-friendly ---
def try_read_json_local(filename: str):
    import sys
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS") else (Path(__file__).parent if "__file__" in globals() else Path(os.getcwd()))
    p = base / filename
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), str(p)
    return None, None

def normalize_mapping_json(obj: Any) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k is None or v is None: continue
            mapping[str(k).strip()] = str(v).strip()
    elif isinstance(obj, list):
        from_keys = ["from","orig","source","codigo","cod","key","de"]
        to_keys   = ["to","dest","group","grupo","setor","value","para"]
        for it in obj:
            if not isinstance(it, dict): continue
            src = next((str(it[k]).strip() for k in from_keys if k in it and it[k] is not None), None)
            dst = next((str(it[k]).strip() for k in to_keys   if k in it and it[k] is not None), None)
            if src and dst: mapping[src] = dst
    return mapping

def apply_json_mapping(series: pd.Series, mapping: Dict[str,str]) -> Tuple[pd.Series, int]:
    if not mapping: return series, 0
    m_upper = {str(k).strip().upper(): v for k, v in mapping.items()}
    s_raw = series.astype(str).fillna("").str.strip()
    s_upper = s_raw.str.upper()
    out = s_raw.tolist(); hits = 0
    for i, val in enumerate(s_upper):
        if val in m_upper: out[i] = m_upper[val]; hits += 1
    return pd.Series(out, index=series.index), hits

def search_in_df(df: pd.DataFrame, query: str, cols: List[str]) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q: return df
    terms = [t for t in q.split() if t]
    if not terms: return df
    for c in cols:
        if c not in df.columns: df[c] = ""
    mask = pd.Series(True, index=df.index)
    for t in terms:
        term = pd.Series(False, index=df.index)
        for c in cols:
            term |= df[c].astype(str).str.lower().str.contains(t, na=False, regex=False)
        mask &= term
    return df[mask]

CHART_H = 370
CHART_H_METAS = 360

# Paleta para tema escuro
ALT_BG = "#0f172a00"
ALT_FG = "#e5e7eb"
ALT_GRID = "#1f2937"

def alt_dark_theme(chart: alt.Chart):
    return (
        chart
        .configure(background=ALT_BG)
        .configure_axis(
            labelColor=ALT_FG,
            titleColor=ALT_FG,
            gridColor=ALT_GRID
        )
        .configure_legend(
            labelColor=ALT_FG,
            titleColor=ALT_FG
        )
        .configure_header(
            labelColor=ALT_FG,
            titleColor=ALT_FG
        )
        .configure_view(strokeWidth=0)
    )

def compact_chart(data: pd.DataFrame, x_field: str, y_field: str, x_title: str, height: int = CHART_H):
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y(f"{y_field}:N", sort="-x", title=None),
            tooltip=[y_field, alt.Tooltip(f"{x_field}:Q", title=x_title, format=",.2f")],
        )
        .properties(height=height, width="container")
        .configure_view(strokeWidth=0)
        .configure_axis(labelLimit=160, labelFontSize=11, titleFontSize=12)
    )

def _alt_bar_h_with_labels(df, value_col: str, cat_col: str, title_x: str, *,
                           value_fmt=",.2f", color_field=None, height=CHART_H):
    enc_color = alt.Color(f"{color_field}:N", title=color_field) if color_field else alt.value(None)
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat_col}:N", sort="-x", title=None),
        x=alt.X(f"{value_col}:Q", title=title_x, axis=alt.Axis(format=value_fmt)),
        color=enc_color,
        tooltip=[cat_col] + ([color_field] if color_field else []) + [
            alt.Tooltip(f"{value_col}:Q", title=title_x, format=value_fmt)
        ],
    )
    bars = base.mark_bar()
    labels = base.mark_text(align="left", baseline="middle", dx=6, color=ALT_FG).encode(
        text=alt.Text(f"{value_col}:Q", format=value_fmt)
    )
    return (bars + labels).properties(height=height, width="container").configure_view(strokeWidth=0)

def apply_parent_child(df, map_df, enabled=True):
    if not enabled or map_df is None or map_df.empty:
        df["codigo_base"] = df["codigo"]; return df
    cols_lower = {c.lower(): c for c in map_df.columns}
    pai_col   = cols_lower.get("pai")   or cols_lower.get("parent") or cols_lower.get("codigo_pai")  or list(map_df.columns)[0]
    filho_col = cols_lower.get("filho") or cols_lower.get("child")  or cols_lower.get("codigo_filho") or list(map_df.columns)[1]
    m = map_df[[pai_col, filho_col]].dropna().astype(str).apply(lambda s: s.str.strip())
    child2parent = dict(zip(m[filho_col], m[pai_col]))
    df["codigo_base"] = df["codigo"].map(lambda c: child2parent.get(c, c))
    return df

# --------- SPLIT código - nome ----------
def split_codigo_nome_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Regra: sempre existe pelo menos um traço. Se houver 2+ traços e
    o trecho APÓS o 1º traço começar com preço pt-BR (00,00), ignora o 1º
    e usa o 2º traço para separar.
    """
    s = series.fillna("").astype(str)
    codigos, nomes = [], []
    price_re = re.compile(r'^\s*\d{1,3}(?:\.\d{3})*,\d{2}(?:\b|$)')  # 89,99 | 1.234,56

    for raw in s:
        txt = raw.strip()
        hy = [m.start() for m in re.finditer("-", txt)]
        if not hy:
            # fallback defensivo
            codigos.append(txt); nomes.append(txt); continue

        idx = hy[0]
        if len(hy) > 1:
            right_after_first = txt[idx + 1:]
            if price_re.match(right_after_first):
                idx = hy[1]

        left  = txt[:idx].strip(" -")
        right = txt[idx+1:].strip(" -")

        if '"' in left:
            left = re.sub(r"\D+", "", left)
        left  = left.replace('"', "").strip()
        right = right.replace('"', "").strip()

        codigos.append(left)
        nomes.append(right if right else txt)

    return pd.Series(codigos, index=s.index), pd.Series(nomes, index=s.index)

# =====================================================
# Helpers para calendário de venda (usados em "🎯 Metas")
# =====================================================
def _weekday_ordered_first_open(dows_set: set[int]) -> int:
    # prioridade: Domingo(6), depois Seg(0)..Sáb(5)
    order = [6,0,1,2,3,4,5]
    for d in order:
        if d in dows_set:
            return d
    return 0  # fallback

def alt_x_date(field: str = "data_dt"):
    """
    Eixo X temporal normalizado (um rótulo por dia).
    Usa timeUnit yearmonthdate para não repetir rótulos.
    """
    return alt.X(
        f"yearmonthdate({field}):T",
        title="Dia",
        axis=alt.Axis(format="%d/%m/%Y", labelAngle=90, labelOverlap=True)
    )


def _count_sell_days(start_dt: pd.Timestamp, end_dt: pd.Timestamp, valid_dows: set[int]) -> int:
    if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt or not valid_dows:
        return 0
    total = 0
    cur = start_dt.normalize()
    end_norm = end_dt.normalize()
    while cur <= end_norm:
        span_end = min(end_norm, cur + pd.Timedelta(days=(6 - cur.weekday())))
        delta = (span_end - cur).days + 1
        for i in range(delta):
            d = (cur + pd.Timedelta(days=i)).weekday()
            if d in valid_dows:
                total += 1
        cur = span_end + pd.Timedelta(days=1)
    return int(total)

def _first_open_date_this_week(ref_dt: pd.Timestamp, first_open_dow: int) -> pd.Timestamp:
    base = ref_dt.normalize()
    monday = base - pd.Timedelta(days=base.weekday())  # segunda
    # pandas: Monday=0..Sunday=6; se first_open_dow==6, é domingo
    target = monday + pd.Timedelta(days=first_open_dow if first_open_dow != 6 else 6)
    return target

# =====================================================
# Leitura de Histórico: CSV ou Parquet
# =====================================================
def _coerce_hist_schema(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    cols_norm = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in cols_norm.items()}

    c_data  = next((inv[k] for k in ["data", "date", "dt"] if k in inv), None)
    c_loja  = next((inv[k] for k in ["loja", "store", "filial", "pdv"] if k in inv), None)
    c_codigo= next((inv[k] for k in ["codigo","código","sku","cod"] if k in inv), None)
    c_nome  = next((inv[k] for k in ["nome","descricao","descrição","produto","item"] if k in inv), None)
    c_cn    = next((inv[k] for k in ["codigo_nome","código_nome","cn"] if k in inv), None)
    c_qtd   = next((inv[k] for k in ["quantidade","qtd","qty","unidades"] if k in inv), None)
    c_valor = next((inv[k] for k in ["valor","venda","total","preco","preço","amount"] if k in inv), None)

    if c_cn is not None:
        cn_series = df[c_cn].astype(str)
    elif c_codigo is not None and c_nome is not None:
        cn_series = (df[c_codigo].astype(str).str.strip() + " - " + df[c_nome].astype(str).str.strip())
    elif c_codigo is not None:
        cn_series = df[c_codigo].astype(str)
    elif c_nome is not None:
        cn_series = df[c_nome].astype(str)
    else:
        cn_series = None

    if (c_data is not None) and (c_loja is not None) and (cn_series is not None) and (c_qtd is not None) and (c_valor is not None):
        out = pd.DataFrame({
            "__data":  df[c_data].astype(str),
            "__loja":  df[c_loja].astype(str),
            "__cn":    cn_series.astype(str),
            "__qtd":   df[c_qtd].astype(str),
            "__valor": df[c_valor].astype(str),
        })
        return out

    df = df.dropna(axis=1, how="all").dropna(how="all")
    if df.shape[1] < 5:
        for i in range(5 - df.shape[1]): df[f"_vazio_{i}"] = ""
    df = df.iloc[:, :5]
    df = df.rename(columns={
        df.columns[0]: "__data",
        df.columns[1]: "__loja",
        df.columns[2]: "__cn",
        df.columns[3]: "__qtd",
        df.columns[4]: "__valor",
    })
    for c in ["__data","__loja","__cn","__qtd","__valor"]:
        df[c] = df[c].astype(str)
    return df

@st.cache_data
def read_hist_any(file) -> pd.DataFrame:
    name = getattr(file, "name", "") or ""
    ext = os.path.splitext(name.lower())[1]
    if ext in [".parquet", ".pq"]:
        _safe_seek(file)
        try:
            dfp = pd.read_parquet(file)
        except Exception as e:
            raise RuntimeError(f"Falha ao ler Parquet: {e}")
        df_norm = _coerce_hist_schema(dfp)
        return df_norm[["__data","__loja","__cn","__qtd","__valor"]]
    _safe_seek(file)
    df_csv = read_csv_hist_no_header(file)
    df_csv = df_csv.rename(columns={
        df_csv.columns[0]: "__data",
        df_csv.columns[1]: "__loja",
        df_csv.columns[2]: "__cn",
        df_csv.columns[3]: "__qtd",
        df_csv.columns[4]: "__valor",
    })
    return df_csv[["__data","__loja","__cn","__qtd","__valor"]]

# =========================
# Leitor de CSV de Metas
# =========================
@st.cache_data
def read_metas_csv(file) -> pd.DataFrame:
    def _try_read(sep):
        _safe_seek(file)
        return pd.read_csv(file, header=None, dtype=str, sep=sep, engine="python")
    try:
        df = _try_read(";")
    except Exception:
        try:
            df = _try_read(",")
        except Exception:
            _safe_seek(file)
            df = pd.read_csv(file, header=None, dtype=str, engine="python")
    if df.shape[1] < 5:
        raise RuntimeError("CSV de metas precisa de 5 colunas (A..E).")
    df = df.iloc[:, :5].rename(columns={
        df.columns[0]: "loja",
        df.columns[1]: "setor",
        df.columns[2]: "meta_raw",
        df.columns[3]: "inicio_raw",
        df.columns[4]: "fim_raw",
    })
    df["loja"]  = df["loja"].astype(str).str.strip()
    df["setor"] = df["setor"].astype(str).str.strip()
    df["meta"] = pd.to_numeric(
        df["meta_raw"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)
    inicio = parse_date_smart(df["inicio_raw"])
    fim    = parse_date_smart(df["fim_raw"])
    df["inicio"] = inicio.dt.date
    df["fim"]    = fim.dt.date
    bad = df["inicio"].isna() | df["fim"].isna()
    df = df[~bad].copy()
    return df[["loja","setor","meta","inicio","fim"]]

# =====================================================
# Uploads
# =====================================================
up1, up2, up3 = st.columns([2, 1.2, 1.2])
with up1:
    hist_file = st.file_uploader("Histórico (CSV sem cabeçalho ou Parquet)", type=["csv", "parquet", "pq"])
with up2:
    codigos_file = st.file_uploader("CSV de Códigos (alvo/destaque) — 1 coluna", type=["csv"])
with up3:
    map_file = st.file_uploader("CSV de Mapeamento Pai/Filho", type=["csv"])

metas_file = st.file_uploader("CSV de Metas por Loja/Setor (A:loja, B:setor, C:meta, D:início, E:fim)", type=["csv"])

metas_df = None
if metas_file is not None:
    try:
        metas_df = read_metas_csv_robusto(metas_file)
    except Exception as e:
        st.error(f"Erro ao ler CSV de metas: {e}")

if hist_file is None:
    st.info("Faça upload do arquivo de histórico (CSV/Parquet) para começar.")
    st.stop()

prod_map_local, produtos_json_path = try_read_json_local("produtos.json")
prod_map: Dict[str, str] = normalize_mapping_json(prod_map_local) if prod_map_local else {}

loja_map = {}
setor_map = {}
loja_json_local, loja_json_path = try_read_json_local("agrupamentos.json")
if loja_json_local: loja_map = normalize_mapping_json(loja_json_local)
setor_json_local, setor_json_path = try_read_json_local("setores.json")
if setor_json_local: setor_map = normalize_mapping_json(setor_json_local)

# =====================================================
# Leitura + normalização do histórico
# =====================================================
try:
    raw_hist = read_hist_any(hist_file)
except Exception as e:
    st.error(f"Erro ao ler o histórico (CSV/Parquet): {e}")
    st.stop()

raw_hist = raw_hist.rename(columns={
    raw_hist.columns[0]: "__data",
    raw_hist.columns[1]: "__loja",
    raw_hist.columns[2]: "__cn",
    raw_hist.columns[3]: "__qtd",
    raw_hist.columns[4]: "__valor",
})

dt = parse_date_smart(raw_hist["__data"])
raw_hist["data"] = dt.dt.date

codigo_s, nome_s = split_codigo_nome_series(raw_hist["__cn"])

qtd = pd.to_numeric(
    raw_hist["__qtd"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    errors="coerce"
).fillna(0.0)
val = pd.to_numeric(
    raw_hist["__valor"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    errors="coerce"
).fillna(0.0)

vendas = pd.DataFrame({
    "data": raw_hist["data"],
    "loja": raw_hist["__loja"].astype(str).str.strip(),
    "codigo": codigo_s.astype(str).str.strip(),
    "nome": nome_s.astype(str).str.strip(),
    "quantidade": qtd,
    "valor": val,
})

if loja_map:
    vendas["loja"], _ = apply_json_mapping(vendas["loja"], loja_map)

map_df = None
if map_file is not None:
    try:
        _safe_seek(map_file)
        map_df = pd.read_csv(map_file, sep=";", decimal=",", dtype=str, encoding="utf-8")
    except Exception:
        _safe_seek(map_file)
        map_df = pd.read_csv(map_file, dtype=str)
agrupar_pai_filho = st.sidebar.checkbox("Agrupar Pai/Filho", value=True)
vendas = apply_parent_child(vendas, map_df, enabled=agrupar_pai_filho)

def setor_lookup(cod_base: str, cod: str) -> str:
    if not prod_map: return "Sem Setor"
    s = prod_map.get(str(cod_base))
    if s: return s
    s = prod_map.get(str(cod))
    if s: return s
    return "Sem Setor"

vendas["setor"] = [setor_lookup(cb, c) for cb, c in zip(vendas["codigo_base"].astype(str), vendas["codigo"].astype(str))]
if setor_map:
    vendas["setor"], _ = apply_json_mapping(vendas["setor"], setor_map)

RJ_SET = {"MDC CARIOCA","MDC BONSUCESSO","MDC MADUREIRA","MDC SANTA CRUZ","MDC NILÓPOLIS","MDC MESQUITA"}
def resolve_regiao(loja: str) -> str:
    if loja == "RJ": return "RJ"
    if loja == "SP": return "SP"
    return "RJ" if loja in RJ_SET else "SP"
vendas["regiao"] = vendas["loja"].astype(str).map(resolve_regiao)

vendas["loja_orig"] = raw_hist["__loja"].astype(str).str.strip()
vendas["setor_orig"] = vendas["setor"]

codigos_alvo = None
if codigos_file is not None:
    try:
        _safe_seek(codigos_file)
        df_cod = pd.read_csv(codigos_file, dtype=str, sep=";", decimal=",", encoding="utf-8")
    except Exception:
        _safe_seek(codigos_file)
        df_cod = pd.read_csv(codigos_file, dtype=str)
    codigos_alvo = df_cod.iloc[:,0].astype(str).str.strip().tolist()

# =====================================================
# Filtros principais
# =====================================================
st.sidebar.header("Filtros")

min_d, max_d = vendas["data"].min(), vendas["data"].max()
if pd.isna(pd.Timestamp(min_d)) or pd.isna(pd.Timestamp(max_d)):
    st.error("O histórico não possui datas válidas."); st.stop()

modo_data = st.sidebar.radio("Modo de data", ["Período", "Data única"], index=0, horizontal=True)
if modo_data == "Período":
    periodo = st.sidebar.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(periodo, (list, tuple)) and len(periodo) == 2:
        d_ini, d_fim = periodo
    else:
        d_ini, d_fim = (min_d, max_d)
else:
    d_unica = st.sidebar.date_input("Data", value=max_d, min_value=min_d, max_value=max_d)
    d_ini = d_fim = d_unica

df = vendas[(vendas["data"] >= d_ini) & (vendas["data"] <= d_fim)].copy()

regioes = ["Todas"] + sorted(df["regiao"].unique().tolist())
lojas = ["Todas"] + sorted(df["loja"].unique().tolist())
setores = ["Todos"] + sorted(df["setor"].unique().tolist())

reg_sel  = st.sidebar.selectbox("Região", regioes, index=0)
loja_sel = st.sidebar.selectbox("Loja (agrupada)", lojas, index=0)
setor_sel= st.sidebar.selectbox("Setor (agrupado)", setores, index=0)

if reg_sel != "Todas":  df = df[df["regiao"] == reg_sel]
if loja_sel != "Todas": df = df[df["loja"]   == loja_sel]
if setor_sel != "Todos": df = df[df["setor"] == setor_sel]

if df.empty:
    st.warning("Nenhum dado após aplicar os filtros.")
    st.stop()

# =====================================================
# Agregações
# =====================================================
agg_geral = (
    df.groupby(["codigo_base"], as_index=False)
      .agg(valor_total=("valor","sum"),
           quantidade_total=("quantidade","sum"),
           nome=("nome","first"))
)
agg_geral["rank_geral"] = agg_geral["valor_total"].rank(method="dense", ascending=False).astype(int)
agg_geral = agg_geral.sort_values(["rank_geral","codigo_base"])

rank_loja = (
    df.groupby(["loja","codigo_base"], as_index=False)
      .agg(valor_total=("valor","sum"),
           quantidade_total=("quantidade","sum"),
           nome=("nome","first"))
)
rank_loja["rank_loja"] = rank_loja.groupby("loja")["valor_total"].rank(method="dense", ascending=False).astype(int)
rank_loja = rank_loja.sort_values(["loja","rank_loja","codigo_base"])

rank_setor = (
    df.groupby(["loja","setor","codigo_base"], as_index=False)
      .agg(valor_total=("valor","sum"),
           quantidade_total=("quantidade","sum"),
           nome=("nome","first"))
)
rank_setor["rank_setor"] = rank_setor.groupby(["loja","setor"])["valor_total"].rank(method="dense", ascending=False).astype(int)
rank_setor = rank_setor.sort_values(["loja","setor","rank_setor","codigo_base"])

# =====================================================
# Navegação
# =====================================================
SECOES = [
    "📈 Visão Geral & Gráficos",
    "🏆 Ranking Geral",
    "🏪 Ranking por Loja",
    "🗂️ Ranking por Setor",
    "🎯 Destaques",
    "🎯 Metas",
    "📆 Comparar Dias",
    "📄 Relatório",
    "🗓️ Histórico",
]
if "nav_secao" not in st.session_state:
    st.session_state["nav_secao"] = SECOES[0]
def _on_nav_change(): st.session_state["nav_secao"] = st.session_state["_nav_radio"]
st.radio("Seção", options=SECOES, index=SECOES.index(st.session_state["nav_secao"]), horizontal=True, key="_nav_radio", on_change=_on_nav_change)
secao = st.session_state["nav_secao"]

# =====================================================
# Abas
# =====================================================
if secao == "📈 Visão Geral & Gráficos":
    st.subheader("Visão Geral (com os filtros aplicados)")
    total_venda = df["valor"].sum()
    total_qtd   = df["quantidade"].sum()
    total_skus  = df["codigo_base"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Venda Total", fmt_currency_br(total_venda))
    c2.metric("Quantidade", f"{int(total_qtd):,}".replace(",", "."))
    c3.metric("SKUs Únicos", f"{total_skus:,}".replace(",", "."))

    st.markdown("##### Totais por Loja (Top 10)")
    loja_tot = (df.groupby("loja", as_index=False)
                  .agg(valor_total=("valor","sum"))
                  .sort_values("valor_total", ascending=False)
                  .head(10))
    if not loja_tot.empty and np.isfinite(loja_tot["valor_total"].to_numpy()).any():
        st.altair_chart(compact_chart(loja_tot, "valor_total", "loja", "Vendas (R$)"))
    else:
        st.info("Sem dados para exibir o gráfico de lojas com os filtros atuais.")

    st.markdown("##### Totais por Setor (Top 10)")
    setor_tot = (df.groupby("setor", as_index=False)
                   .agg(valor_total=("valor","sum"))
                   .sort_values("valor_total", ascending=False)
                   .head(10))
    if not setor_tot.empty and np.isfinite(setor_tot["valor_total"].to_numpy()).any():
        st.altair_chart(compact_chart(setor_tot, "valor_total", "setor", "Vendas (R$)"))
    else:
        st.info("Sem dados para exibir o gráfico de setores com os filtros atuais.")

elif secao == "🏆 Ranking Geral":
    st.subheader("Ranking Geral (por código base)")
    q = st.text_input("🔎 Pesquisar", key="s1", placeholder="código, nome…")
    view = search_in_df(agg_geral, q, ["codigo_base","nome"]).sort_values("rank_geral")
    st.data_editor(
        view[["rank_geral","codigo_base","nome","valor_total","quantidade_total"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank_geral": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "valor_total": st.column_config.NumberColumn("Valor (R$)", format="%.2f"),
            "quantidade_total": st.column_config.NumberColumn("Qtd", format="%.0f"),
        }, disabled=True,
    )

elif secao == "🏪 Ranking por Loja":
    st.subheader("Ranking por Loja")
    loja_v = st.selectbox("Loja", sorted(df["loja"].unique()), index=0, key="sl1")
    base = rank_loja[rank_loja["loja"] == loja_v].copy().sort_values("rank_loja")
    q = st.text_input("🔎 Pesquisar", key="s2", placeholder="código, nome…")
    base = search_in_df(base, q, ["codigo_base","nome"])
    st.data_editor(
        base[["rank_loja","codigo_base","nome","valor_total","quantidade_total"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank_loja": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "valor_total": st.column_config.NumberColumn("Valor (R$)", format="%.2f"),
            "quantidade_total": st.column_config.NumberColumn("Qtd", format="%.0f"),
        }, disabled=True,
    )



elif secao == "🗂️ Ranking por Setor":
    st.subheader("Ranking por Setor")
    setor_v = st.selectbox("Setor", sorted(df["setor"].unique()), index=0, key="ss1")
    base = rank_setor[rank_setor["setor"] == setor_v].copy().sort_values("rank_setor")
    q = st.text_input("🔎 Pesquisar", key="s3", placeholder="código, nome…")
    base = search_in_df(base, q, ["codigo_base","nome"])
    st.data_editor(
        base[["rank_setor","codigo_base","nome","valor_total","quantidade_total"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank_setor": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "valor_total": st.column_config.NumberColumn("Valor (R$)", format="%.2f"),
            "quantidade_total": st.column_config.NumberColumn("Qtd", format="%.0f"),
        }, disabled=True,
    )

    # Localizar a seção "🗂️ Ranking por Setor" e adicionar após os rankings:

    # Após a linha que contém: for setor_agrup in setores_agrupados_ordenados:
    # E após todo o loop de exibição dos rankings, adicionar:

    st.markdown("---")
    st.subheader("📄 Exportar Relatório de Ranking")

    col_export1, col_export2 = st.columns([1, 3])

    with col_export1:
        top_n_export = st.number_input(
            "Top N produtos por setor",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Quantidade de produtos a incluir no ranking de cada setor"
        )

    with col_export2:
        # Obtem a lista de setores únicos presentes no DF de 'rank_setor'
        # Adiciona 'Todas' se loja_sel for 'Todas' para o caso de querer exportar para todas as lojas
        setores_disponiveis = sorted(list(set(rank_setor["setor"].values)))

        # inicializa estado (mantém compatibilidade)
        if 'setores_export' not in st.session_state:
            st.session_state['setores_export'] = setores_disponiveis[:5]  # default: 5 primeiros (ou ajuste)

        # botões de seleção rápida
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("✓ Selecionar Todos"):
                st.session_state['setores_export'] = setores_disponiveis
        with col_btn2:
            if st.button("✗ Limpar Seleção"):
                st.session_state['setores_export'] = []
        with col_btn3:
            if st.button("⇄ Inverter Seleção"):
                atual = st.session_state.get('setores_export', [])
                st.session_state['setores_export'] = [s for s in setores_disponiveis if s not in atual]

        # Filtra o default para conter apenas valores válidos (resolve o erro do Streamlit)
        default_setores = st.session_state.get('setores_export', setores_disponiveis[:5])
        if not isinstance(default_setores, list):
            default_setores = [default_setores]
        default_setores_validos = [s for s in default_setores if s in setores_disponiveis]

        setores_para_export = st.multiselect(
            "Selecione os setores para exportar",
            options=setores_disponiveis,
            default=default_setores_validos,
            key='setores_export'
        )

    if setores_para_export:
        col_btn_export1, col_btn_export2 = st.columns(2)

        with col_btn_export1:
            if loja_sel != "Todas":
                # Exportar apenas a loja selecionada
                if st.button("📥 Exportar Loja Atual", use_container_width=True):
                    with st.spinner(f"Gerando relatório para {loja_sel}..."):
                        try:
                            from reportlab.platypus import PageBreak

                            def gerar_relatorio_ranking_setor_tabela_bytes_por_setor(rank_setor_df, loja_sel, top_n, setores_escolhidos):
                                """
                                Gera um PDF (BytesIO) com uma página (ou mais) por setor.
                                Cada setor: título + tabela com colunas [rank_setor, codigo_base, nome, loja, setor].
                                Retorna: BytesIO (ponteiro no início).
                                """
                                # valida
                                if rank_setor_df is None or rank_setor_df.empty:
                                    raise RuntimeError("DataFrame vazio — nada para exportar")

                                # colunas desejadas na ordem
                                cols_needed = ["rank_setor", "codigo_base", "nome", "loja", "setor"]

                                # garante lista de setores (mantém ordem solicitada)
                                if not setores_escolhidos:
                                    setores = sorted(rank_setor_df["setor"].dropna().unique().tolist())
                                else:
                                    setores = [s for s in setores_escolhidos if s in rank_setor_df["setor"].unique()]

                                if not setores:
                                    raise RuntimeError("Nenhum setor válido para exportar.")

                                # prepara doc
                                pdf_buffer = BytesIO()
                                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
                                styles = getSampleStyleSheet()
                                elements = []

                                for i, setor in enumerate(setores):
                                    df_s = rank_setor_df[rank_setor_df["setor"] == setor].copy()
                                    if df_s.empty:
                                        continue

                                    # recria rank_setor se não existir
                                    if "rank_setor" not in df_s.columns:
                                        if "valor_total" in df_s.columns:
                                            df_s["rank_setor"] = df_s.groupby("setor")["valor_total"].rank(method="dense", ascending=False).astype(int)
                                        else:
                                            df_s = df_s.reset_index(drop=True)
                                            df_s.insert(0, "rank_setor", range(1, len(df_s) + 1))

                                    # selecionar top_n por setor (se top_n for None ou <=0, ignora)
                                    if top_n and int(top_n) > 0:
                                        df_s = df_s.sort_values("rank_setor").head(int(top_n)).copy()
                                    else:
                                        df_s = df_s.sort_values("rank_setor").copy()

                                    # garantir as colunas na ordem pedida
                                    for c in cols_needed:
                                        if c not in df_s.columns:
                                            df_s[c] = ""
                                    df_page = df_s[cols_needed].fillna("").astype(str)

                                    # título do setor
                                    titulo = f"{setor} — {loja_sel} (Itens: {len(df_page)})"
                                    elements.append(Paragraph(titulo, styles['Heading3']))
                                    elements.append(Spacer(1, 6))

                                    # monta a tabela (com larguras proporcionais)
                                    page_width, page_height = A4
                                    usable_width = page_width - doc.leftMargin - doc.rightMargin
                                    # pesos simples: nome maior, código e rank menores
                                    weights = [0.6, 1.0, 3.0, 1.2, 1.2]  # ajusta se quiser
                                    total_w = sum(weights)
                                    col_widths = [usable_width * (w / total_w) for w in weights]

                                    data = [cols_needed] + df_page.values.tolist()
                                    table = Table(data, colWidths=col_widths, repeatRows=1)
                                    table_style = TableStyle([
                                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
                                        ('TEXTCOLOR',(0,0),(-1,0), colors.black),
                                        ('ALIGN',(0,0),(-1,-1),'LEFT'),
                                        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                        ('GRID', (0,0), (-1,-1), 0.4, colors.grey),
                                        ('FONTSIZE', (0,0), (-1,-1), 8),
                                        ('BOTTOMPADDING', (0,0), (-1,0), 6),
                                    ])
                                    table.setStyle(table_style)
                                    elements.append(table)

                                    # se não for o último setor, adiciona quebra de página
                                    if i != len(setores) - 1:
                                        elements.append(PageBreak())

                                # build
                                doc.build(elements)
                                pdf_buffer.seek(0)
                                return pdf_buffer  # BytesIO

                            # gera o PDF (a função interna já monta páginas por setor)
                            pdf_buffer = gerar_relatorio_ranking_setor_tabela_bytes_por_setor(
                                rank_setor_df=rank_setor[rank_setor["loja"] == loja_sel],  # filtra por loja atual
                                loja_sel=loja_sel,
                                top_n=top_n_export,
                                setores_escolhidos=setores_para_export
                            )

                            # normalize bytes (aceita BytesIO, bytes, ou getvalue())
                            if hasattr(pdf_buffer, "getvalue"):
                                pdf_bytes = pdf_buffer.getvalue()
                            else:
                                pdf_bytes = pdf_buffer if isinstance(pdf_buffer, (bytes, bytearray)) else bytes(pdf_buffer)

                            nome_arquivo = f"ranking_{loja_sel.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                            st.download_button(
                                label=f"⬇️ Download {nome_arquivo}",
                                data=pdf_bytes,
                                file_name=nome_arquivo,
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success(f"Relatório gerado com sucesso!")

                        except Exception as e:
                            st.error(f"Erro ao gerar relatório: {str(e)}")

        with col_btn_export2:
            if loja_sel == "Todas":
                # Exportar todas as lojas em arquivos separados
                if st.button("📥 Exportar Todas as Lojas (ZIP)", use_container_width=True):
                    with st.spinner("Gerando relatórios para todas as lojas..."):
                        try:
                            # Chama a função que gera o ZIP
                            # Definição temporária da função gerar_relatorio_ranking_setor
                            def gerar_relatorio_ranking_setor(rank_setor_df, loja_sel, top_n, setores_escolhidos):
                                # Esta função deve gerar um arquivo ZIP contendo relatórios de ranking por setor/loja.
                                # Implemente a lógica conforme necessário.
                                import zipfile
                                from io import BytesIO
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                    # Exemplo: para cada loja, gera um arquivo CSV por setor
                                    lojas = sorted(rank_setor_df["loja"].unique())
                                    for loja in lojas if loja_sel == "Todas" else [loja_sel]:
                                        for setor in setores_escolhidos:
                                            df = rank_setor_df[(rank_setor_df["loja"] == loja) & (rank_setor_df["setor"] == setor)]
                                            df_top = df.sort_values("valor_total", ascending=False).head(top_n)
                                            csv_bytes = df_top.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                                            zf.writestr(f"ranking_{loja}_{setor}.csv", csv_bytes)
                                zip_buffer.seek(0)
                                return zip_buffer

                            zip_buffer = gerar_relatorio_ranking_setor(
                                rank_setor_df=rank_setor, # DF completo com todas as lojas
                                loja_sel="Todas",         # Indica modo "Todas as lojas"
                                top_n=top_n_export,
                                setores_escolhidos=setores_para_export
                            )
                            zip_buffer.seek(0)
                            nome_zip = f"rankings_todas_lojas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

                            st.download_button(
                                label=f"⬇️ Download {nome_zip}",
                                data=zip_buffer,
                                file_name=nome_zip,
                                mime="application/zip",
                                use_container_width=True
                            )
                            st.success(f"Relatórios gerados com sucesso! Várias lojas incluídas.")
                        except Exception as e:
                            st.error(f"Erro ao gerar relatórios: {str(e)}")
    else:
        st.warning("Selecione pelo menos um setor para exportar.")

elif secao == "🎯 Destaques":
    st.subheader("Destaques (Códigos-alvo)")
    if not codigos_alvo:
        st.info("Envie um CSV de códigos para ver os destaques.")
    else:
        alvo = pd.DataFrame({"codigo_original": codigos_alvo})
        if agrupar_pai_filho and map_df is not None and not map_df.empty:
            cols_lower = {c.lower(): c for c in map_df.columns}
            pai_col   = cols_lower.get("pai") or list(map_df.columns)[0]
            filho_col = cols_lower.get("filho") or list(map_df.columns)[1]
            m = map_df[[pai_col, filho_col]].dropna().astype(str).apply(lambda s: s.str.strip())
            c2p = dict(zip(m[filho_col], m[pai_col]))
            alvo["codigo_base"] = alvo["codigo_original"].map(lambda c: c2p.get(c, c))
        else:
            alvo["codigo_base"] = alvo["codigo_original"]

        resumo = alvo.merge(
            agg_geral[["codigo_base","nome","valor_total","quantidade_total","rank_geral"]],
            on="codigo_base", how="left"
        ).fillna({"valor_total":0.0,"quantidade_total":0.0})
        resumo["rank_venda"] = resumo["valor_total"].rank(method="dense", ascending=False).astype(int)
        resumo["rank_qtd"]   = resumo["quantidade_total"].rank(method="dense", ascending=False).astype(int)

        q = st.text_input("🔎 Pesquisar", key="s4", placeholder="código, nome…")
        resumo = search_in_df(resumo, q, ["codigo_original","codigo_base","nome"]).sort_values(["valor_total","quantidade_total"], ascending=[False,False])

        st.data_editor(
            resumo[["rank_venda","rank_qtd","codigo_original","codigo_base","nome","valor_total","quantidade_total"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "rank_venda": st.column_config.NumberColumn("Posição (Valor)", format="%d", width="small"),
                "rank_qtd":   st.column_config.NumberColumn("Posição (Qtd)",   format="%d", width="small"),
                "valor_total": st.column_config.NumberColumn("Valor (R$)", format="%.2f"),
                "quantidade_total": st.column_config.NumberColumn("Qtd", format="%.0f"),
            }, disabled=True,
        )

        st.divider()
        st.markdown("### 📈 Evolução histórica dos destaques")
        cod_bases_foco = resumo["codigo_base"].dropna().astype(str).unique().tolist()
        base_high = df[df["codigo_base"].astype(str).isin(cod_bases_foco)].copy()
        if base_high.empty:
            st.warning("Os códigos do arquivo de destaques não aparecem no período/filtros atuais.")
        else:
            evo_high = (base_high.assign(data_dt=pd.to_datetime(base_high["data"]))
                                   .groupby("data_dt", as_index=False)["valor"].sum()
                                   .sort_values("data_dt"))
            st.altair_chart(
                alt.Chart(evo_high).mark_line(point=True).encode(
                    x=alt.X("data_dt:T", title="Dia", axis=alt.Axis(format="%d/%m/%Y", labelAngle=90)),
                    y=alt.Y("valor:Q",   title="Vendas (R$)"),
                    tooltip=[alt.Tooltip("data_dt:T", title="Dia", format="%d/%m/%Y"),
                             alt.Tooltip("valor:Q", title="Vendas", format=",.2f")]
                ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0)
            )

            st.markdown("#### Itens de destaque (linhas por item)")
            top_k_dest = st.slider("Top itens (destaques)", 3, 20, 8, 1, key="top_k_dest")
            top_items = (base_high.groupby(["codigo_base","nome"], as_index=False)["valor"].sum()
                                   .sort_values("valor", ascending=False).head(top_k_dest))
            base_high_top = base_high.merge(top_items[["codigo_base"]], on="codigo_base", how="inner")
            if not base_high_top.empty:
                evo_it = (base_high_top.assign(data_dt=pd.to_datetime(base_high_top["data"]))
                                       .groupby(["data_dt","nome"], as_index=False)["valor"].sum()
                                       .sort_values("data_dt"))
                st.altair_chart(
                    alt.Chart(evo_it).mark_line(point=True).encode(
                        x=alt.X("data_dt:T", title="Dia", axis=alt.Axis(format="%d/%m/%Y", labelAngle=90)),
                        y=alt.Y("valor:Q", title="Vendas (R$)"),
                        color=alt.Color("nome:N", title="Item"),
                        tooltip=[alt.Tooltip("data_dt:T", title="Dia", format="%d/%m/%Y"),
                                 "nome",
                                 alt.Tooltip("valor:Q", title="Vendas", format=",.2f")],
                    ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0)
                )

    

    
elif secao == "🎯 Metas":

    def build_metas_tabela(df_hist, metas_df, loja_sel, *, valid_dows=None):
        import numpy as np
        import pandas as pd
        if valid_dows is None:
            valid_dows = set(range(7))   # 0=Seg ... 6=Dom

        r = metas_df.copy()

        # filtro por loja
        if loja_sel and loja_sel != "Todas" and "loja" in r.columns:
            r = r[r["loja"].astype(str) == str(loja_sel)]

        # numéricos seguros
        for c in ["meta","vendas_periodo","faltante_rs"]:
            if c in r.columns:
                r[c] = pd.to_numeric(r[c], errors="coerce")
        # meta
        if "meta" in r.columns:
            r["meta"] = pd.to_numeric(r["meta"], errors="coerce").fillna(0.0)
        else:
            r["meta"] = 0.0

        # vendas_periodo
        if "vendas_periodo" in r.columns:
            r["vendas_periodo"] = pd.to_numeric(r["vendas_periodo"], errors="coerce").fillna(0.0)
        else:
            r["vendas_periodo"] = 0.0


        r["inicio"] = pd.to_datetime(r.get("inicio"), errors="coerce")
        r["fim"]    = pd.to_datetime(r.get("fim"), errors="coerce")

        # reconstrução de vendas_periodo a partir do histórico da tela (df)
        base_hist = _resolve_df(["vendas", "_vendas_full", "raw_vendas"], fallback=df_hist).copy()

        if loja_sel and loja_sel != "Todas" and "loja" in base_hist.columns:
            base_hist = base_hist[base_hist["loja"].astype(str) == str(loja_sel)]

        if "data" in base_hist.columns and "data_dt" not in base_hist.columns:
            base_hist["data_dt"] = pd.to_datetime(base_hist["data"])
        elif "data_dt" in base_hist.columns:
            base_hist["data_dt"] = pd.to_datetime(base_hist["data_dt"])
        else:
            base_hist["data_dt"] = pd.NaT

        def realizado_ate_hoje(ini, fim, setor):
            if pd.isna(ini) or pd.isna(fim): return 0.0
            ini = pd.to_datetime(ini)
            end = min(pd.to_datetime(fim), pd.to_datetime("today").normalize())
            s_left  = base_hist["setor"].astype(str).str.strip()
            s_right = str(setor).strip()
            m = (base_hist["data_dt"] >= ini) & (base_hist["data_dt"] <= end) & (s_left == s_right)
            return float(base_hist.loc[m, "valor"].sum())
        
        def _valid_dows_por_loja(loja: str) -> set[int]:
            # 0=Seg ... 6=Dom
            lj = (str(loja) or "").strip().upper()
            # Ajuste aqui as lojas que abrem aos domingos:
            if lj in {"MDC MESQUITA"}:
                return {0,1,2,3,4,5,6}  # inclui domingo
            return {0,1,2,3,4,5}        # fecha aos domingos
        
        def _peso_dia(loja: str, dt: pd.Timestamp) -> float:
            """
            Peso por dia restante:
            - Sábado na MDC CARIOCA vale 0.3, MAS se o sábado for o dia ATUAL, vale 1.0
            - Domingo só vale 1.0 para lojas que abrem domingo; caso contrário 0.0
            - Demais dias valem 1.0
            """
            lj = (str(loja) or "").strip().upper()
            d  = pd.to_datetime(dt).normalize()
            dow = int(d.weekday())  # 0=Seg ... 6=Dom
            hoje = pd.to_datetime("today").normalize()

            # domingo
            if dow == 6:
                return 1.0 if (lj == "MDC MESQUITA") else 0.0

            # sábado
            if dow == 5:
                # EXCEÇÃO: se o sábado é o DIA ATUAL, conte como 1.0 p/ MDC CARIOCA
                if lj == "MDC CARIOCA" and d == hoje:
                    return 1.0
                return 0.3 if lj == "MDC CARIOCA" else 1.0

            # seg-sex
            return 1.0




        def dias_decorridos_setor(ini, fim, loja):
            import pandas as pd
            # define 'hoje' no escopo local (normalizado para 00:00)
            hoje = pd.to_datetime("today").normalize()

            if pd.isna(ini):
                return 0

            ini = pd.to_datetime(ini).normalize()
            fim = pd.to_datetime(fim).normalize() if pd.notna(fim) else None

            # inclui HOJE no intervalo
            end = min(fim, hoje) if fim is not None else hoje
            if end < ini:
                return 0
            v = _valid_dows_por_loja(loja)
            total = 0
            cur = ini
            while cur <= end:
                if cur.weekday() in v:
                    total += 1
                cur += pd.Timedelta(days=1)
            return int(total)


        def dias_restantes_setor(ini, fim, loja):
            hoje = pd.to_datetime("today").normalize()
            if pd.isna(ini) or pd.isna(fim):
                return 0
            ini = pd.to_datetime(ini); fim = pd.to_datetime(fim)
            if fim < hoje:
                return 0
            start = max(hoje, ini)  # inclui HOJE
            v = _valid_dows_por_loja(loja)
            total = 0
            cur = start
            while cur <= fim:
                if cur.weekday() in v:
                    total += 1
                cur += pd.Timedelta(days=1)
            return int(total)



        # --- calcula por linha (respeitando setor) ---
        r["vendas_periodo"] = [
            realizado_ate_hoje(ini, fim, setor)
            for ini, fim, setor in zip(r["inicio"], r["fim"], r["setor"])
        ]

        r["dias_decorridos"] = [
            dias_decorridos_setor(ini, fim, loja)
            for ini, fim, loja in zip(r["inicio"], r["fim"], r.get("loja", "" if len(r)==0 else r["loja"]))
        ]

        r["dias_restantes"] = [
            dias_restantes_setor(ini, fim, loja)
            for ini, fim, loja in zip(r["inicio"], r["fim"], r.get("loja", "" if len(r)==0 else r["loja"]))
        ]

        # --- soma de pesos dos dias restantes (de amanhã até o fim) por loja/meta ---
        from datetime import datetime

        def _soma_pesos_restantes(ini, fim, loja):
            if pd.isna(ini) or pd.isna(fim): return 0.0
            ini = pd.to_datetime(ini); fim = pd.to_datetime(fim)
            start = max(pd.to_datetime("today").normalize(), ini)  # inclui HOJE
            if start > fim: return 0.0

            cur = start
            total = 0.0
            while cur <= fim:
                w = _peso_dia(loja, cur)  # sem fracionar HOJE
                total += w
                cur += pd.Timedelta(days=1)
            return float(total)



        r["peso_restante"] = [
            _soma_pesos_restantes(ini, fim, loja)
            for ini, fim, loja in zip(r["inicio"], r["fim"], r.get("loja", "" if len(r)==0 else r["loja"]))
        ]

        hoje = pd.to_datetime("today").normalize()

        # faltante e ritmos
        r["faltante_rs"] = r["meta"] - r["vendas_periodo"]

        # ritmo atual diário (média dos dias DECORRIDOS)
        r["ritmo_atual_dia"] = np.where(r["dias_decorridos"] > 0,
                                        r["vendas_periodo"] / r["dias_decorridos"], 0.0)

        # >>> necessário ponderado pelos PESOS dos dias restantes <<<
        r["necessario_dia"] = np.where(r["peso_restante"] > 0,
                                    r["faltante_rs"] / r["peso_restante"], 0.0)

        r["gap_dia"] = r["ritmo_atual_dia"] - r["necessario_dia"]


        encerrado = (pd.to_datetime(r["fim"]) < hoje) | (r["dias_restantes"] <= 0)
        r["status_ritmo"] = np.where(
            encerrado,
            np.where(r["faltante_rs"] <= 0, "✅ OK", "⚠️ Abaixo"),
            np.where((r["necessario_dia"] <= 0) | (r["gap_dia"] >= 0), "✅ OK", "⚠️ Abaixo")
        )

        cols = [
            "loja","setor",
            "dias_restantes","dias_decorridos",
            "vendas_periodo","faltante_rs",
            "ritmo_atual_dia","necessario_dia","gap_dia","status_ritmo",
        ]
        keep = [c for c in cols if c in r.columns]
        return r[keep].sort_values(["gap_dia","faltante_rs"], ascending=[False, False])

    
    def _resolve_df(candidate_keys, fallback=None):
        # tenta pegar do session_state
        for k in candidate_keys:
            df = st.session_state.get(k, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        # tenta pegar de variáveis globais
        for k in candidate_keys:
            df = globals().get(k, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        # fallback explícito (se passado)
        if isinstance(fallback, pd.DataFrame) and not fallback.empty:
            return fallback
        return None

    # =========================
    # Helpers (corrigidos)
    # =========================
    def _fmt_pct(x):
        try:
            return f"{float(x):.1%}"
        except Exception:
            return "-"

    def _ax_table_from_df(ax, df, col_labels, col_formats=None, row_height=0.22, font_size=9, col_align=None):
        ax.axis("off")
        tb = Table(ax, bbox=[0, 0, 1, 1])

        n_rows, n_cols = len(df), len(col_labels)
        widths = [1.2] * n_cols
        if col_align is None:
            col_align = ["left"] * n_cols

        # Header
        for j, lab in enumerate(col_labels):
            cell = tb.add_cell(-1, j, widths[j]/sum(widths), row_height,
                            text=str(lab), loc="center", facecolor="#111827")
            cell.get_text().set_color("#e5e7eb")
            cell.get_text().set_fontsize(font_size)
            cell.get_text().set_fontweight("bold")

        # Linhas
        for i in range(n_rows):
            for j, c in enumerate(df.columns[:n_cols]):
                val = df.iloc[i, j]
                if col_formats and col_formats[j]:
                    val = col_formats[j](val)
                cell = tb.add_cell(i, j, widths[j]/sum(widths), row_height,
                                text=str(val), loc=col_align[j])
                cell.get_text().set_color("#e5e7eb")
                cell.get_text().set_fontsize(font_size)

        for j in range(n_cols):
            tb._cells[(-1, j)].visible_edges = "open"

        ax.add_table(tb)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    # ============================================
    # Geração do PRINT ÚNICO (PNG + PDF) — ajustado
    # ============================================
    def gerar_print_metas_png_pdf(evo_src, metas_src, loja_sel, setor_sel, fmt_currency_br, lojas_escolhidas=None):
        """
        Retorna:
        - (png_bytes, pdf_bytes) para 0 ou 1 loja
        - (zip_bytes, None)      para 2+ lojas  (ZIP contendo PNG+PDF por loja)

        Observação: datas no eixo X em dd/mm/aaaa; série "Necessário/dia" NÃO é exibida;
        pontos diários anotados com o valor; KPIs em layout 2x2.
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        from matplotlib.table import Table
        from io import BytesIO
        from datetime import datetime
        import zipfile

        # -------------------- helpers --------------------
        def _fmt_pct(x):
            try:
                return f"{float(x):.1%}"
            except Exception:
                return "-"

        def _ax_table_from_df(ax, df, col_labels, col_formats=None,
                            row_height=0.07, font_size=10,
                            col_align=None, text_color="#000000"):
            ax.axis("off")
            tb = Table(ax, bbox=[0, 0, 1, 1])
            n_rows, n_cols = len(df), len(col_labels)
            widths = [1.5 if i == 0 else 1.0 for i in range(n_cols)]
            if col_align is None:
                col_align = ["left", "center", "right", "right", "right", "right", "center"][:n_cols]

            # header
            for j, lab in enumerate(col_labels):
                cell = tb.add_cell(-1, j, widths[j]/sum(widths), row_height,
                                text=str(lab), loc="center",
                                facecolor="#111827", edgecolor="black")
                cell.get_text().set_color("#ffffff")
                cell.get_text().set_fontsize(font_size)
                cell.get_text().set_fontweight("bold")
                cell.set_linewidth(0.5)

            # linhas
            for i in range(n_rows):
                for j, c in enumerate(df.columns[:n_cols]):
                    val = df.iloc[i, j]
                    if col_formats and col_formats[j]:
                        val = col_formats[j](val)
                    cell = tb.add_cell(i, j, widths[j]/sum(widths), row_height - 0.005,
                                    text=str(val), loc=col_align[j], edgecolor="black")
                    cell.get_text().set_color(text_color)
                    cell.get_text().set_fontsize(font_size)
                    cell.set_linewidth(0.25)

            tb.auto_set_font_size(False)
            tb.set_fontsize(font_size)
            tb.auto_set_column_width(col=list(range(n_cols)))
            ax.add_table(tb)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return ax

        def _build_one(loja_scope, evo_df, metas_df) -> tuple[bytes, bytes]:
            """Gera UMA página (PNG+PDF) para a loja_scope (ou contexto atual)."""
            # -------------------- KPIs --------------------
            meta_total = float(metas_df["meta"].sum()) if "meta" in metas_df.columns else 0.0

            if "vendas_periodo" in metas_df.columns:
                realizado_total = float(metas_df["vendas_periodo"].sum())
            elif isinstance(evo_df, pd.DataFrame) and not evo_df.empty and "data_dt" in evo_df.columns:
                ultimo_dia_venda = evo_df["data_dt"].max()
                realizado_total = float(evo_df.loc[evo_df["data_dt"] <= ultimo_dia_venda, "valor"].sum())
            else:
                realizado_total = 0.0

            ating_total = (realizado_total / meta_total) if meta_total else 0.0

            if "atingimento_pct" in metas_df.columns:
                ating_col = pd.to_numeric(metas_df["atingimento_pct"], errors="coerce").fillna(0)
                if ating_col.max() > 2:  # se veio em %
                    ating_col = ating_col / 100.0
                setores_bateram = int((ating_col >= 1.0).sum())
            else:
                setores_bateram = 0

            # -------------------- Série diária (apenas vendas) --------------------
            have_series = isinstance(evo_df, pd.DataFrame) and {"data_dt", "valor"}.issubset(evo_df.columns)
            if have_series and not evo_df.empty:
                per_dia = evo_df.groupby("data_dt", as_index=False)["valor"].sum().sort_values("data_dt")
                titulo_serie = (f"Vendas — {loja_scope}"
                                if (loja_scope and loja_scope != "Todas") else "Vendas — período")
                plot_df = per_dia.copy()
                plot_df["serie"] = titulo_serie
            else:
                plot_df = pd.DataFrame(columns=["data_dt", "valor", "serie"])
                titulo_serie = "Vendas (sem série diária)"

            # -------------------- Top atingimentos / urgências --------------------
            base_best = (metas_df.sort_values("atingimento_pct", ascending=False).head(12).copy()
                        if "atingimento_pct" in metas_df.columns else metas_df.head(0).copy())
            base_urg  = (metas_df.sort_values("faltante_rs", ascending=False).head(12).copy()
                        if "faltante_rs" in metas_df.columns else metas_df.head(0).copy())

            # -------------------- Tabela (ordem por Δ dia desc) --------------------
            # 0) Reaproveita a mesma tabela da UI se existir (garante números idênticos)
            ui_key = f"_ritmo_view_{loja_scope}"
            ui_tbl = st.session_state.get(ui_key)

            if isinstance(ui_tbl, pd.DataFrame) and not ui_tbl.empty:
                cols_print = ["setor","dias_restantes","faltante_rs","ritmo_atual_dia","necessario_dia","gap_dia","status_ritmo"]
                cols_print = [c for c in cols_print if c in ui_tbl.columns]
                tbl = ui_tbl[cols_print].copy().head(12).reset_index(drop=True)
                tabela_tipo = "loja"
            else:
                # Fallback: monta com a mesma função usada na interface
                tbl_full = build_metas_tabela(
                    df_hist=df,
                    metas_df=metas_work,
                    loja_sel=loja_scope,
                    valid_dows={0,1,2,3,4,5},
                )

                cols_print = ["setor","dias_restantes","faltante_rs","ritmo_atual_dia","necessario_dia","gap_dia","status_ritmo"]
                cols_print = [c for c in cols_print if c in tbl_full.columns]
                tbl = tbl_full[cols_print].head(12).reset_index(drop=True)
                tabela_tipo = "loja"
            # -------------------- FIGURA --------------------
            fig = plt.figure(figsize=(12, 18), dpi=180)
            gs = GridSpec(5, 4, figure=fig, wspace=0.08, hspace=0.26,
                        top=0.990, left=0.06, right=0.99, bottom=0.28)

            # --- HEADER (KPIs 2x2) ---
            ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
            ax0.text(0.0, 0.70, "Metas — resumo", fontsize=15, fontweight="bold", va="top")

            left_x, right_x = 0.00, 0.52
            row1_y, row2_y = 0.50, 0.20
            label_fs, value_fs = 12, 20

            def kpi(ax, x, y, label, value):
                ax.text(x, y, label, fontsize=label_fs, va="top")
                ax.text(x, y - 0.12, value, fontsize=value_fs, fontweight="bold", va="top")

            kpi(ax0, left_x,  row1_y, "Meta total",          fmt_currency_br(meta_total))
            kpi(ax0, right_x, row1_y, "Realizado",           fmt_currency_br(realizado_total))
            kpi(ax0, left_x,  row2_y, "Atingimento",         _fmt_pct(ating_total))
            kpi(ax0, right_x, row2_y, "Setores que bateram", str(setores_bateram))

            # --- Série diária ---
            ax1 = fig.add_subplot(gs[1, :])
            ax1.set_title(titulo_serie, fontsize=12, color="#000000")
            if not plot_df.empty:
                a = plot_df.sort_values("data_dt")
                for serie, g in a.groupby("serie"):
                    ax1.plot(g["data_dt"], g["valor"], marker="o", linewidth=2, label=serie)
                    # rótulo no topo do ponto
                    for x, y in zip(g["data_dt"], g["valor"]):
                        ax1.annotate(fmt_currency_br(y), (x, y),
                                    xytext=(0, 6), textcoords='offset points',
                                    ha='center', va='bottom', fontsize=8)
                # padding vertical p/ “centralizar” a linha
                ymin = float(a["valor"].min()) if len(a) else 0.0
                ymax = float(a["valor"].max()) if len(a) else 1.0
                rng  = max(1.0, ymax - ymin)
                ax1.set_ylim(ymin - rng*0.35, ymax + rng*0.35)
                ax1.legend(loc="upper left", frameon=True, framealpha=0.8)
                ax1.set_ylabel("R$ por dia")
                # 1 rótulo por ponto: um tick por data
                xticks = pd.to_datetime(a["data_dt"].drop_duplicates().tolist())
                ax1.set_xticks(xticks)
                ax1.set_xticklabels([d.strftime("%d/%m/%Y") for d in xticks], rotation=0, ha="center")
                from matplotlib.ticker import NullLocator
                ax1.xaxis.set_minor_locator(NullLocator())

            else:
                ax1.text(0.5, 0.5, "Sem dados de série diária disponíveis",
                        ha="center", va="center", fontsize=10)
            # eixo X em dd/mm/aaaa e rótulos retos
            ax1.grid(True, alpha=0.2)

            # --- Melhores atingimentos ---
            ax2 = fig.add_subplot(gs[2, :])
            ax2.set_title("Melhores atingimentos (%)", fontsize=11)
            if not base_best.empty and "atingimento_pct" in base_best.columns:
                y = np.arange(len(base_best))[::-1]
                vals = base_best["atingimento_pct"].astype(float).to_numpy()
                vals = np.where(vals > 2, vals / 100.0, vals)  # garante 0–1

                # CORES por faixa: <100% = azul, >=100% = verde, >=110% = amarelo
                colors = np.where(
                    vals >= 1.10, "#facc15",               # amarelo
                    np.where(vals >= 1.00, "#22c55e", "#60a5fa")  # verde / azul
                )

                ax2.barh(y, vals, height=0.58, color=colors)  # <- usa as cores
                labels = (base_best["setor"].astype(str)
                        if (loja_scope and loja_scope != "Todas")
                        else (base_best.get("loja", base_best["setor"]).astype(str) + " / " + base_best["setor"].astype(str)))
                ax2.set_yticks(y, labels)
                for i, v in enumerate(vals):
                    ax2.text(v + 0.01, y[i], _fmt_pct(v), va="center", fontsize=9)
                vmax = float(vals.max()) if len(vals) else 1.0
                ax2.set_xlim(0, max(1.1, vmax*1.15))
                ax2.grid(True, axis="x", alpha=0.15)
            else:
                ax2.text(0.5, 0.5, "Sem dados de atingimentos disponíveis",
                        ha="center", va="center", fontsize=10)

            # --- Maiores urgências ---
            ax3 = fig.add_subplot(gs[3, :])
            ax3.set_title("Maiores urgências (R$ faltante)", fontsize=11)
            if not base_urg.empty and "faltante_rs" in base_urg.columns:
                y = np.arange(len(base_urg))[::-1]
                ax3.barh(y, base_urg["faltante_rs"].values, height=0.58)
                ax3.barh(y, base_urg["faltante_rs"].values, height=0.58, color="#60a5fa")  # azul
                labels = (base_urg["setor"].astype(str)
                        if (loja_scope and loja_scope != "Todas")
                        else (base_urg.get("loja", base_urg["setor"]).astype(str) + " / " + base_urg["setor"].astype(str)))
                ax3.set_yticks(y, labels)
                for i, v in enumerate(base_urg["faltante_rs"].values):
                    ax3.text(float(v) * 1.01, y[i], fmt_currency_br(v), va="center", fontsize=9)
                ax3.grid(True, axis="x", alpha=0.15)
            else:
                ax3.text(0.5, 0.5, "Sem dados de urgências disponíveis",
                        ha="center", va="center", fontsize=10)

            # --- Tabela ---
            ax4 = fig.add_subplot(gs[4, :]); ax4.axis("off")
            if (tabela_tipo == "loja") and not tbl.empty:
                cols = ["setor", "dias_restantes", "faltante_rs", "ritmo_atual_dia",
                        "necessario_dia", "gap_dia", "status_ritmo"]
                _ax_table_from_df(
                    ax4, tbl[cols],
                    ["Setor", "Dias", "Faltante (R$)", "Ritmo atual", "Necessário", "Δ (dia)", "Status"],
                    col_formats=[None,
                                lambda v: f"{int(v):d}",
                                fmt_currency_br,
                                fmt_currency_br,
                                fmt_currency_br,
                                fmt_currency_br,
                                None],
                    col_align=["center", "center", "center", "center", "center", "center", "center"],
                    text_color="#000000",
                )
            else:
                ax4.text(0.5, 0.5, "Sem dados disponíveis para a tabela",
                        ha="center", va="center", fontsize=12, color="#000000")

            fig.suptitle(f"Metas — {datetime.now().strftime('%d/%m/%Y')}", y=0.995, fontsize=12)

            # export
            png_buf, pdf_buf = BytesIO(), BytesIO()
            fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.6)
            fig.savefig(pdf_buf, format="pdf", dpi=180, bbox_inches="tight", pad_inches=0.6)
            plt.close(fig)
            png_buf.seek(0); pdf_buf.seek(0)
            return png_buf.getvalue(), pdf_buf.getvalue()

        # =================== modo multi-loja (ZIP) ===================
        if lojas_escolhidas and len(lojas_escolhidas) > 1:
            zip_mem = BytesIO()
            with zipfile.ZipFile(zip_mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for loja in lojas_escolhidas:
                    # recortes por loja
                    metas_lj = metas_src[metas_src.get("loja", "").astype(str) == str(loja)] if "loja" in metas_src.columns else metas_src.copy()
                    evo_lj = (evo_src[evo_src.get("loja", "").astype(str) == str(loja)]
                            if isinstance(evo_src, pd.DataFrame) and "loja" in evo_src.columns else evo_src)
                    if metas_lj is None or metas_lj.empty:
                        continue
                    p_bytes, d_bytes = _build_one(loja, evo_lj, metas_lj)
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    base = f"metas_{loja}_{ts}"
                    zf.writestr(f"{base}.png", p_bytes)
                    zf.writestr(f"{base}.pdf", d_bytes)
            zip_mem.seek(0)
            return zip_mem.getvalue(), None

        # =================== escopo único ===================
        # recortes (se 1 loja foi passada explicitamente)
        target_loja = None
        metas_sc = metas_src.copy()
        evo_sc = evo_src.copy() if isinstance(evo_src, pd.DataFrame) else evo_src
        if lojas_escolhidas and len(lojas_escolhidas) == 1:
            target_loja = lojas_escolhidas[0]
            if "loja" in metas_sc.columns:
                metas_sc = metas_sc[metas_sc["loja"].astype(str) == str(target_loja)]
            if isinstance(evo_sc, pd.DataFrame) and "loja" in evo_sc.columns:
                evo_sc = evo_sc[evo_sc["loja"].astype(str) == str(target_loja)]

        return _build_one(target_loja or loja_sel, evo_sc, metas_sc)
    
    def gerar_print_setores_topitens(
        vendas_df: pd.DataFrame,
        setores_escolhidos: list,
        top_n: int,
        fmt_currency_br,
        loja_sel: str | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        todas_as_lojas: bool = False,
        lojas_ord: list | None = None,
    ):
        """
        Gera páginas Top Itens por SETOR:
        • 1 setor  -> (png_bytes, pdf_bytes)
        • N setores -> (zip_bytes, None) com PNG+PDF por setor
        • todas_as_lojas=True -> (zip_bytes, None) com 1 PDF por loja (multi-página: 1 página por setor)

        vendas_df pode ter: data OU data_dt ; codigo OU codigo_base.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.table import Table
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime
        from io import BytesIO
        import zipfile

        # -------- schema flex --------
        base = vendas_df.copy()
        if "data_dt" not in base.columns:
            if "data" in base.columns:
                base["data_dt"] = pd.to_datetime(base["data"])
            else:
                raise AssertionError("vendas_df precisa de 'data' ou 'data_dt'.")
        else:
            base["data_dt"] = pd.to_datetime(base["data_dt"])

        if "codigo_base" not in base.columns:
            if "codigo" in base.columns:
                base["codigo_base"] = base["codigo"].astype(str)
            else:
                raise AssertionError("vendas_df precisa de 'codigo' ou 'codigo_base'.")

        if "nome" not in base.columns:
            # tenta alguns nomes comuns
            for alt in ["produto", "descricao", "descrição", "item", "nome"]:
                if alt in base.columns:
                    base["nome"] = base[alt].astype(str)
                    break
            if "nome" not in base.columns:
                base["nome"] = base["codigo_base"].astype(str)

        req = {"setor", "quantidade", "valor"}
        miss = req - set(base.columns)
        if miss:
            raise AssertionError(f"vendas_df faltando colunas: {sorted(miss)}")

        # recorte por loja (se aplicável) – só para o modo "loja atual"
        # recorte por loja (apenas no modo loja atual)
        if (not todas_as_lojas) and loja_sel and "loja" in base.columns and loja_sel != "Todas":
            base = base[base["loja"] == loja_sel]


        # período da “semana”
        if end_date is None:
            end_date = pd.to_datetime(base["data_dt"].max())
        else:
            end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = end_date - pd.Timedelta(days=6)  # últimos 7 dias
        else:
            start_date = pd.to_datetime(start_date)
            start_date = start_date.normalize()
            end_date   = end_date.normalize()       # << ADICIONE ESTA LINHA

        # ---------------- figure builder (1 página) ----------------
        def _build_fig(df_setor: pd.DataFrame, setor: str):
            # KPI
            vendas_semana = float(df_setor["valor"].sum())
            qtd_semana    = int(df_setor["quantidade"].sum())
            skus_semana   = int(df_setor["codigo_base"].nunique())

            # Top N
            agr = (df_setor.groupby(["codigo_base","nome"], as_index=False)
                            .agg(qtd=("quantidade","sum"), vendas=("valor","sum"))
                            .sort_values("vendas", ascending=False))
            total = float(agr["vendas"].sum()) if len(agr) else 0.0
            if total > 0:
                agr["share"] = agr["vendas"] / total
            else:
                agr["share"] = 0.0
            agr["acum"] = agr["share"].cumsum()
            top = agr.head(max(1, int(top_n))).reset_index(drop=True)
            top.insert(0, "rank", np.arange(1, len(top) + 1))

            # FIG
            plt.close("all")
            fig = plt.figure(figsize=(12, 14), dpi=180)
            gs = GridSpec(5, 4, figure=fig, wspace=0.08, hspace=0.28,
                        top=0.990, left=0.06, right=0.99, bottom=0.24)

            # Header/KPIs
            ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
            ax0.text(0.00, 0.92, f"📦 Top itens — {setor}", fontsize=15, fontweight="bold", va="top")
            ax0.text(0.00, 0.70, "Vendas na semana", fontsize=11, va="top")
            ax0.text(0.00, 0.55, fmt_currency_br(vendas_semana), fontsize=22, fontweight="bold", va="top")
            ax0.text(0.52, 0.70, "Qtd total", fontsize=11, va="top")
            ax0.text(0.52, 0.55, f"{qtd_semana:,}".replace(",", "."), fontsize=18, fontweight="bold", va="top")
            ax0.text(0.52, 0.38, "SKUs", fontsize=11, va="top")
            ax0.text(0.52, 0.25, f"{skus_semana:,}".replace(",", "."), fontsize=18, fontweight="bold", va="top")
            ax0.text(0.00, 0.25, f"Período: {start_date.strftime('%d/%m/%Y')} — {end_date.strftime('%d/%m/%Y')}",
                    fontsize=10, va="top", color="#6b7280")

            # Gráfico
            ax1 = fig.add_subplot(gs[1:3, :])
            if not top.empty:
                y = np.arange(len(top))[::-1]
                ax1.barh(y, top["vendas"].values, height=0.58, color="#60a5fa")
                ax1.set_yticks(y, [str(n)[:55] for n in top["nome"].astype(str)])
                for i, v in enumerate(top["vendas"].values):
                    ax1.text(float(v) * 1.01, y[i], fmt_currency_br(v), va="center", fontsize=9)
                ax1.set_xlabel("Vendas (R$)")
                ax1.set_title(f"Top {len(top)} itens — {setor}", fontsize=12, color="#9CA3AF")
                ax1.grid(True, axis="x", alpha=0.15)
            else:
                ax1.text(0.5, 0.5, "Sem vendas no período.", ha="center", va="center", fontsize=12)

            # Tabela
            ax2 = fig.add_subplot(gs[3:, :]); ax2.axis("off")
            tbl = top[["rank","codigo_base","nome","qtd","vendas","share","acum"]].copy()
            col_formats = [
                lambda v: f"{int(v):d}",
                str, str,
                lambda v: f"{int(v):d}",
                fmt_currency_br,
                lambda v: f"{float(v):.2%}",
                lambda v: f"{float(v):.2%}",
            ]
            # centralizado
            tb = Table(ax2, bbox=[0, 0, 1, 1])
            headers = ["Rank","Código","Produto","Qtd","Vendas (R$)","Share","Acum."]
            widths = [1.0, 1.2, 3.2, 1.0, 1.3, 1.0, 1.0]
            for j, lab in enumerate(headers):
                cell = tb.add_cell(-1, j, widths[j]/sum(widths), 0.09, text=lab, loc="center",
                                facecolor="#111827", edgecolor="black")
                cell.get_text().set_color("#ffffff"); cell.get_text().set_fontsize(9); cell.get_text().set_fontweight("bold")
            for i in range(len(tbl)):
                for j, c in enumerate(tbl.columns):
                    val = tbl.iloc[i, j]
                    if col_formats[j]: val = col_formats[j](val)
                    cell = tb.add_cell(i, j, widths[j]/sum(widths), 0.085, text=str(val), loc="center", edgecolor="black")
                    cell.get_text().set_fontsize(9)
            ax2.add_table(tb); ax2.set_xlim(0,1); ax2.set_ylim(0,1)

            fig.suptitle(f"{setor}", y=0.994, fontsize=12)
            return fig

        # ------------ seleção de setores ------------
        setores = [s for s in (setores_escolhidos or []) if pd.notna(s)]
        if not setores: raise ValueError("Passe ao menos 1 setor em `setores_escolhidos`.")

        # ------------ modos de saída ------------
        if todas_as_lojas:
            if "loja" not in base.columns:
                raise AssertionError("Para 'todas_as_lojas', vendas_df precisa da coluna 'loja'.")
            lojas_lista = lojas_ord or sorted(base["loja"].dropna().unique().tolist())
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            zip_mem = BytesIO()
            with zipfile.ZipFile(zip_mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for loja in lojas_lista:
                    b_loja = base[base["loja"] == loja]
                    if b_loja.empty: continue
                    pdf_buf = BytesIO()
                    with PdfPages(pdf_buf) as pdf:
                        for setor in setores:
                            df_s = b_loja[(b_loja["setor"] == setor) &
                                        (b_loja["data_dt"] >= start_date) &
                                        (b_loja["data_dt"] <= end_date)]
                            fig = _build_fig(df_s, f"{setor} — {loja}")
                            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.5)
                            plt.close(fig)
                    pdf_buf.seek(0)
                    zf.writestr(f"topitens_{loja}_{ts}.pdf", pdf_buf.getvalue())
            zip_mem.seek(0)
            return zip_mem.getvalue(), None

        # --- loja atual ---
        if len(setores) == 1:
            df_s = base[(base["setor"] == setores[0]) &
                        (base["data_dt"] >= start_date) &
                        (base["data_dt"] <= end_date)]
            fig = _build_fig(df_s, setores[0])
            png_buf, pdf_buf = BytesIO(), BytesIO()
            fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.5)
            fig.savefig(pdf_buf, format="pdf", dpi=180, bbox_inches="tight", pad_inches=0.5)
            plt.close(fig)
            png_buf.seek(0); pdf_buf.seek(0)
            return png_buf.getvalue(), pdf_buf.getvalue()

        # vários setores (loja atual) -> ZIP com PNG+PDF por setor
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        zip_mem = BytesIO()
        with zipfile.ZipFile(zip_mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for setor in setores:
                df_s = base[(base["setor"] == setor) &
                            (base["data_dt"] >= start_date) &
                            (base["data_dt"] <= end_date)]
                fig = _build_fig(df_s, setor)
                png_buf, pdf_buf = BytesIO(), BytesIO()
                fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.5)
                fig.savefig(pdf_buf, format="pdf", dpi=180, bbox_inches="tight", pad_inches=0.5)
                plt.close(fig)
                png_buf.seek(0); pdf_buf.seek(0)
                zf.writestr(f"topitens_{setor}_{ts}.png", png_buf.getvalue())
                zf.writestr(f"topitens_{setor}_{ts}.pdf", pdf_buf.getvalue())
        zip_mem.seek(0)
        return zip_mem.getvalue(), None




    # ============================================
    # UI da seção (exibe seus gráficos normais + exportação)
    # ============================================
    st.subheader("🎯 Metas por Loja e Setor")

    # -----------------------------
    # 0) Verificações
    # -----------------------------
    if metas_df is None or metas_df.empty:
        st.info("Envie o CSV de metas (A: loja, B: setor, C: meta, D: início, E: fim).")
        st.stop()

    # normaliza metas e aplica os mesmos mapeamentos/filtros (exceto período)
    metas_work = metas_df.copy()
    if loja_map:
        metas_work["loja"], _ = apply_json_mapping(metas_work["loja"], loja_map)
    if setor_map:
        metas_work["setor"], _ = apply_json_mapping(metas_work["setor"], setor_map)
    metas_work["regiao"] = metas_work["loja"].astype(str).map(resolve_regiao)

    st.session_state["metas_unfiltered"] = metas_work.copy()

    # filtros (mesmos do sidebar)
    if reg_sel != "Todas":
        metas_work = metas_work[metas_work["regiao"] == reg_sel]
    if loja_sel != "Todas":
        metas_work = metas_work[metas_work["loja"] == loja_sel]
    if setor_sel != "Todos":
        metas_work = metas_work[metas_work["setor"] == setor_sel]

    if metas_work.empty:
        st.warning("Sem metas para os filtros selecionados.")
        st.stop()

    # chaves normalizadas para juntar com vendas
    metas_work = metas_work.reset_index(drop=True).copy()
    metas_work["meta_id"] = metas_work.index
    metas_work["loja_norm"]  = metas_work["loja"].astype(str).str.strip().str.upper()
    metas_work["setor_norm"] = metas_work["setor"].astype(str).str.strip().str.upper()

    # base de vendas "crua" (sem recorte de período do sidebar; apenas reg/loja/setor)
    vendas_base = vendas.copy()
    if reg_sel != "Todas":
        vendas_base = vendas_base[vendas_base["regiao"] == reg_sel]
    if loja_sel != "Todas":
        vendas_base = vendas_base[vendas_base["loja"] == loja_sel]
    if setor_sel != "Todos":
        vendas_base = vendas_base[vendas_base["setor"] == setor_sel]

    vendas_base["loja_norm"]  = vendas_base["loja"].astype(str).str.strip().str.upper()
    vendas_base["setor_norm"] = vendas_base["setor"].astype(str).str.strip().str.upper()

    # junta e soma vendas dentro do intervalo de cada meta
    merged = vendas_base.merge(
        metas_work[["meta_id","loja_norm","setor_norm","inicio","fim","meta","loja","setor"]],
        on=["loja_norm","setor_norm"], how="right"
    )
    merged["data_dt"]   = pd.to_datetime(merged["data"])
    merged["inicio_dt"] = pd.to_datetime(merged["inicio"])
    merged["fim_dt"]    = pd.to_datetime(merged["fim"])
    in_range = (merged["data_dt"] >= merged["inicio_dt"]) & (merged["data_dt"] <= merged["fim_dt"])
    merged_in = merged[in_range].copy()

    vendas_por_meta = (merged_in.groupby("meta_id", as_index=False)["valor"].sum()
                                .rename(columns={"valor": "vendas_periodo"}))
    metas_work = metas_work.merge(vendas_por_meta, on="meta_id", how="left")
    metas_work["vendas_periodo"] = metas_work["vendas_periodo"].fillna(0.0)

    # ==========================================================
    # Helpers de calendário/abertura e pesos por dia
    # ==========================================================
    def _loja_abre_domingo(loja_nome: str) -> bool:
        return str(loja_nome).strip().upper() == "MDC MESQUITA"

    def _peso_dia(loja_nome: str, dia: pd.Timestamp) -> float:
        """
        Peso 0 para dias fechados.
        - Mesquita: abre domingo a domingo → todos os dias peso >= 0
        - Outras: fecham domingo → peso 0 aos domingos
        - MDC CARIOCA: sábado com peso 0,4 (demais dias úteis = 1,0)
        """
        nome = str(loja_nome).strip().upper()
        dow = int(pd.Timestamp(dia).weekday())  # 0=Seg ... 5=Sáb, 6=Dom
        if dow == 6 and not _loja_abre_domingo(nome):
            return 0.0
        if nome == "MDC CARIOCA" and dow == 5:
            return 0.3
        return 1.0

    # índice auxiliar por loja para "primeiro dia com venda"
    vendas_por_loja_dia = (
        vendas_base.groupby(["loja_norm", pd.to_datetime(vendas_base["data"])])
                   .agg(venda_dia=("valor", "sum")).reset_index()
                   .rename(columns={"data": "data_dt"})
    )

    hoje_dt = pd.to_datetime(datetime.now().date())

    def _primeiro_dia_com_venda(loja_norm: str, d_ini: pd.Timestamp, d_fim: pd.Timestamp) -> pd.Timestamp:
        rng = pd.date_range(d_ini, d_fim, freq="D")
        base = vendas_por_loja_dia[vendas_por_loja_dia["loja_norm"] == loja_norm]
        if base.empty:
            return d_ini
        dias_com_venda = set(pd.to_datetime(base["data_dt"]).dt.normalize().tolist())
        for d in rng:
            if pd.Timestamp(d.normalize()) in dias_com_venda:
                return pd.Timestamp(d)
        return d_ini

    def _dias_inclusivos(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return int((b - a).days) + 1 if (pd.notna(a) and pd.notna(b) and b >= a) else 0

    # ==========================================================
    # 1) Cálculo de dias + pesos restantes (para “necessário por dia”)
    # ==========================================================
    metas_work["inicio_dt"] = pd.to_datetime(metas_work["inicio"])
    metas_work["fim_dt"]    = pd.to_datetime(metas_work["fim"])

    def _calc_days_and_weights(row):
        ini = row["inicio_dt"]; fim = row["fim_dt"]; lj_norm = str(row["loja_norm"])
        lj_nome = str(row["loja"])
        if pd.isna(ini) or pd.isna(fim) or fim < ini:
            return pd.Series({"dias_totais":0, "dias_passados":0, "dias_restantes":0, "peso_restante":0.0})

        ini_eff = _primeiro_dia_com_venda(lj_norm, ini, fim)
        dias_tot = _dias_inclusivos(ini, fim)

        ref_end = min(fim, hoje_dt)
        dias_pass = _dias_inclusivos(ini_eff, ref_end) if ref_end >= ini_eff else 0
        ref_start = max(hoje_dt, ini_eff)
        dias_rest = _dias_inclusivos(ref_start, fim) if fim >= ref_start else 0

        if dias_rest <= 0:
            peso_rest = 0.0
        else:
            rng = pd.date_range(ref_start, fim, freq="D")
            pesos = [_peso_dia(lj_nome, d) for d in rng]
            peso_rest = float(np.sum(pesos))

        return pd.Series({
            "dias_totais": dias_tot,
            "dias_passados": dias_pass,
            "dias_restantes": dias_rest,
            "peso_restante": peso_rest
        })

    calc = metas_work.apply(_calc_days_and_weights, axis=1)
    metas_work[["dias_totais","dias_passados","dias_restantes","peso_restante"]] = calc[["dias_totais","dias_passados","dias_restantes","peso_restante"]]

    # métricas derivadas
    metas_work["atingimento_pct"] = np.where(metas_work["meta"] > 0, metas_work["vendas_periodo"] / metas_work["meta"], 0.0)
    metas_work["faltante_rs"]     = (metas_work["meta"] - metas_work["vendas_periodo"]).clip(lower=0)
    metas_work["ritmo_atual_dia"] = np.where(metas_work["dias_passados"] > 0,
                                             metas_work["vendas_periodo"]/metas_work["dias_passados"], 0.0)

    # necessário por dia baseado em PESOS dos dias restantes
    metas_work["ritmo_necess_rest_dia"] = np.where(
        metas_work["dias_restantes"].fillna(0).astype(int) <= 1,
        metas_work["faltante_rs"],  # 0 ou 1 dia → necessário do dia = faltante
        np.where(
            metas_work["peso_restante"] > 0.0,
            metas_work["faltante_rs"] / metas_work["peso_restante"],  # >1 dia → usa pesos
            np.nan
        )
    )

    metas_work["proj_final"] = metas_work["ritmo_atual_dia"] * metas_work["dias_totais"]
    metas_work["status"] = np.where(metas_work["atingimento_pct"] >= 1.0, "Atingida", "Pendente")

    # -----------------------------
    # 2) KPIs (escopo atual)
    # -----------------------------
    meta_total      = float(metas_work["meta"].sum())
    realizado_total = float(metas_work["vendas_periodo"].sum())
    ating_total     = (realizado_total/meta_total) if meta_total > 0 else 0.0
    setores_bateram = int((metas_work["atingimento_pct"] >= 1.0).sum())

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Meta total", fmt_currency_br(meta_total))
    kc2.metric("Realizado", fmt_currency_br(realizado_total))
    kc3.metric("Atingimento total", f"{ating_total:.1%}")
    kc4.metric("Setores que bateram", f"{setores_bateram:,}".replace(",", "."))

    st.divider()

    # ==========================================================
    # 3) Evolução diária + linha “Necessário/dia” com pesos
    # ==========================================================
    CHART_H_METAS = 320

    if 'merged_in' in locals() and merged_in is not None and not merged_in.empty:
        evo_src = merged_in.copy()
        if "data_dt" not in evo_src.columns:
            evo_src["data_dt"] = pd.to_datetime(evo_src["data"])

        def _pick(colbase: str) -> str:
            if colbase in evo_src.columns: return colbase
            if f"{colbase}_y" in evo_src.columns: return f"{colbase}_y"
            if f"{colbase}_x" in evo_src.columns: return f"{colbase}_x"
            evo_src[colbase] = ""
            return colbase

        loja_col  = _pick("loja")
        setor_col = _pick("setor")

        evo_daily = (
            evo_src.groupby(["data_dt", loja_col, setor_col], as_index=False)["valor"].sum()
                   .rename(columns={loja_col:"loja", setor_col:"setor"})
                   .sort_values("data_dt")
        )

        def _serie_com_necessario(evo_diario_scope: pd.DataFrame,
                                  metas_scope: pd.DataFrame,
                                  titulo_vendas: str,
                                  loja_nome: str | None) -> pd.DataFrame | None:
            if evo_diario_scope.empty or metas_scope.empty:
                return None

            start = pd.to_datetime(metas_scope["inicio"]).min()
            end   = pd.to_datetime(metas_scope["fim"]).max()
            cal = pd.DataFrame({"data_dt": pd.date_range(start, end, freq="D")})

            vendas_dia = (evo_diario_scope.groupby("data_dt", as_index=False)["valor"].sum()
                                         .merge(cal, on="data_dt", how="right")
                                         .fillna({"valor": 0.0})
                                         .sort_values("data_dt"))

            meta_tot = float(metas_scope["meta"].sum())
            acum_ate_ontem = vendas_dia["valor"].shift(1, fill_value=0).cumsum()

            if loja_nome:
                # calendário completo do período da meta
                days_ser = pd.to_datetime(vendas_dia["data_dt"]).dt.normalize()
                days = days_ser.tolist()
                hoje_dt = pd.to_datetime("today").normalize()

                i0 = next((k for k, d in enumerate(days) if d >= hoje_dt), len(days))  # índice de HOJE

                # vendas diárias como vetor numérico
                valores_np = vendas_dia["valor"].fillna(0.0).astype(float).to_numpy()

                # ---- 1) PLANO FIXO PARA TODO O PERÍODO (com pesos por dia) ----
                pesos_all = np.array([_peso_dia(loja_nome, d) for d in days], dtype=float)
                soma_all = float(pesos_all.sum())

                # série que vamos devolver
                necessario = np.zeros(len(days), dtype=float)

                if soma_all > 0.0:
                    base_full = meta_tot / soma_all
                    # plano fixo (não muda mais para dias passados)
                    necessario = base_full * pesos_all
                else:
                    necessario[:] = 0.0  # nada a distribuir

                # ---- 2) PARTIDA = HOJE (não fecha o dia) ----
                # calendário completo do período da meta (já existe acima):
                # days_ser = pd.to_datetime(vendas_dia["data_dt"]).dt.normalize()
                # days = days_ser.tolist()
                # hoje_dt = pd.to_datetime("today").normalize()
                # pesos_all = np.array([_peso_dia(loja_nome, d) for d in days], dtype=float)

                # índice do primeiro dia >= HOJE
                i0 = next((k for k, d in enumerate(days) if d >= hoje_dt), len(days))

                # faltante considerando o realizado ATÉ HOJE (inclui vendas de hoje)
                realizado_ate_hoje = float(vendas_dia.loc[days_ser <= hoje_dt, "valor"].sum())
                restante = max(meta_tot - realizado_ate_hoje, 0.0)

                # pesos dos dias a partir de HOJE (sem fracionar o dia corrente)
                pesos_rest = pesos_all[i0:].astype(float)
                soma_rest = float(pesos_rest.sum())

                if soma_rest > 0.0:
                    base_rest = restante / soma_rest
                    necessario[i0:] = base_rest * pesos_rest
                else:
                    necessario[i0:] = 0.0


            else:
                # (fallback quando a UI está com "Todas" as lojas)
                days_ser = pd.to_datetime(vendas_dia["data_dt"]).dt.normalize()
                days = days_ser.tolist()
                hoje_dt = pd.to_datetime("today").normalize()

                # mantemos o comportamento simples (média nos dias restantes),
                # pois não há um peso único quando misturamos lojas.
                i0 = next((k for k, d in enumerate(days) if d >= hoje_dt), len(days))
                necessario = np.full(len(days), np.nan, dtype=float)
                if i0 < len(days):
                    n_rest = len(days) - i0
                    if n_rest > 0:
                        # faltante até ontem (somando vendas até i0-1)
                        realizado_ate_ontem = float(vendas_dia["valor"].fillna(0.0).astype(float).to_numpy()[:i0].sum())
                        base = max(meta_tot - realizado_ate_ontem, 0.0) / n_rest
                        necessario[i0:] = base



            linha_nec = vendas_dia[["data_dt"]].copy()
            linha_nec["valor"] = necessario
            linha_nec["serie"] = "Necessário/dia"

            serie_plot = vendas_dia.rename(columns={"valor":"valor"}).assign(serie=titulo_vendas)
            return pd.concat([serie_plot, linha_nec], ignore_index=True)

        # Sem filtro → por LOJA (só vendas; sem “necessário” agregado aqui)
        if (loja_sel == "Todas") and (setor_sel == "Todos"):
            per_loja = evo_daily.groupby(["data_dt","loja"], as_index=False)["valor"].sum()
            chart = alt.Chart(per_loja).mark_line(point=True).encode(
                x=alt_x_date("data_dt"),
                y=alt.Y("valor:Q", title="Vendas (R$)"),
                color=alt.Color("loja:N", title="Loja", legend=alt.Legend(orient="right")),
                tooltip=[
                    "loja",
                    alt.Tooltip("yearmonthdate(data_dt):T", title="Dia", format="%d/%m/%Y"),
                    alt.Tooltip("valor:Q", title="Vendas", format=",.2f")
                ],
            )
            st.altair_chart(chart.properties(height=CHART_H_METAS, width="container").configure_view(strokeWidth=0), use_container_width=True)

        # Filtro de LOJA → série única + necessário com pesos da loja
        elif (loja_sel != "Todas") and (setor_sel == "Todos"):
            serie_loja = evo_daily[evo_daily["loja"] == loja_sel].groupby("data_dt", as_index=False)["valor"].sum()
            metas_scope = metas_work[metas_work["loja"] == loja_sel]
            plot_df = _serie_com_necessario(serie_loja, metas_scope, f"Vendas — {loja_sel}", loja_sel)
            if plot_df is not None and not plot_df.empty:
                chart = alt.Chart(plot_df).mark_line(point=True).encode(
                    x=alt_x_date("data_dt"),
                    y=alt.Y("valor:Q", title="R$ por dia"),
                    color=alt.Color("serie:N", legend=alt.Legend(orient="right")),
                    tooltip=[
                        alt.Tooltip("yearmonthdate(data_dt):T", title="Dia", format="%d/%m/%Y"),
                        alt.Tooltip("valor:Q", title="R$ por dia", format=",.2f"),
                        "serie"
                    ],
                )
                st.altair_chart(chart.properties(height=CHART_H_METAS, width="container").configure_view(strokeWidth=0), use_container_width=True)
            else:
                st.info("Sem dados de vendas para a loja selecionada no período das metas.")

        # Filtro de SETOR
        else:
            base = evo_daily[evo_daily["setor"] == setor_sel]
            if loja_sel != "Todas":
                base = base[base["loja"] == loja_sel]
                metas_scope = metas_work[(metas_work["setor"] == setor_sel) & (metas_work["loja"] == loja_sel)]
                titulo = f"Vendas — {loja_sel} / {setor_sel}"
                loja_ctx = loja_sel
            else:
                metas_scope = metas_work[metas_work["setor"] == setor_sel]
                titulo = f"Vendas — {setor_sel}"
                loja_ctx = None  # sem loja definida → fallback uniforme

            serie = base.groupby("data_dt", as_index=False)["valor"].sum()
            plot_df = _serie_com_necessario(serie, metas_scope, titulo, loja_ctx)
            if plot_df is not None and not plot_df.empty:
                chart = alt.Chart(plot_df).mark_line(point=True).encode(
                    x=alt_x_date("data_dt"),
                    y=alt.Y("valor:Q", title="R$ por dia"),
                    color=alt.Color("serie:N", legend=alt.Legend(orient="right")),
                    tooltip=[
                        alt.Tooltip("yearmonthdate(data_dt):T", title="Dia", format="%d/%m/%Y"),
                        alt.Tooltip("valor:Q", title="R$ por dia", format=",.2f"),
                        "serie"
                    ],
                )
                st.altair_chart(chart.properties(height=CHART_H_METAS, width="container").configure_view(strokeWidth=0), use_container_width=True)
            else:
                st.info("Sem dados para o filtro selecionado no período das metas.")
    else:
        st.info("Sem vendas dentro dos intervalos das metas para gerar a série diária.")

    st.divider()

    # ==========================================================
    # 4) Melhores atingimentos (%) e Maiores urgências (R$)
    #    —> AGORA, SÓ QUANDO NÃO HÁ SETOR SELECIONADO
    # ==========================================================
    if setor_sel == "Todos":
        top_k_m = st.slider("Top K (gráficos)", 3, 20, 8, 1, key="metas_topk")

        # a) Melhores atingimentos
        st.markdown("🥇 **Melhores atingimentos (%)**")
        base_best = metas_work.sort_values("atingimento_pct", ascending=False).head(top_k_m).copy()
        if not base_best.empty:
            base_best["faixa_cor"] = np.select(
                [base_best["atingimento_pct"] >= 1.10, base_best["atingimento_pct"] >= 1.00],
                [">=110%", ">=100%"], default="<100%"
            )
            legend_faixa = alt.Legend(title=None, orient="bottom", direction="horizontal", columns=3)
            bars = (alt.Chart(base_best).mark_bar().encode(
                x=alt.X("atingimento_pct:Q", title="Atingimento (%)", axis=alt.Axis(format=".0%")),
                y=alt.Y("setor:N", sort="-x", title=None),
                color=alt.Color("faixa_cor:N",
                                scale=alt.Scale(domain=["<100%",">=100%",">=110%"],
                                                range=["#60a5fa","#22c55e","#facc15"]),
                                legend=legend_faixa),
                tooltip=["loja","setor",
                         alt.Tooltip("vendas_periodo:Q", title="Vendas", format=",.2f"),
                         alt.Tooltip("meta:Q", title="Meta", format=",.2f"),
                         alt.Tooltip("atingimento_pct:Q", title="Ating.", format=".1%")]
            ))
            chart_best = bars
            if loja_sel != "Todas":
                labels = alt.Chart(base_best).mark_text(align="left", baseline="middle", dx=6).encode(
                    y=alt.Y("setor:N", sort="-x", title=None),
                    x=alt.X("atingimento_pct:Q"),
                    text=alt.Text("atingimento_pct:Q", format=".1%"),
                    color=alt.value(ALT_FG)
                )
                chart_best = alt.layer(bars, labels)
            st.altair_chart(chart_best.properties(height=CHART_H, width="container").configure_view(strokeWidth=0), use_container_width=True)

        # b) Maiores urgências (R$ faltante)
        st.markdown("🆘 **Maiores urgências (R$ faltante)**")
        base_urg = metas_work.sort_values("faltante_rs", ascending=False).head(top_k_m).copy()
        if not base_urg.empty:
            enc_color = alt.Color("loja:N", title="Loja") if loja_sel == "Todas" else alt.value("#60a5fa")
            bars_urg = (
                alt.Chart(base_urg)
                .mark_bar()
                .encode(
                    x=alt.X("faltante_rs:Q", title="R$ faltante", axis=alt.Axis(format=",.2f")),
                    y=alt.Y("setor:N", sort="-x", title=None),
                    color=enc_color,
                    tooltip=[
                        "loja", "setor",
                        alt.Tooltip("faltante_rs:Q", title="Faltante", format=",.2f"),
                        alt.Tooltip("vendas_periodo:Q", title="Vendas",   format=",.2f"),
                        alt.Tooltip("meta:Q",           title="Meta",     format=",.2f"),
                    ],
                )
            )
            if loja_sel != "Todas":
                labels_urg = (
                    alt.Chart(base_urg)
                    .mark_text(align="left", baseline="middle", dx=8, fontWeight="bold")
                    .encode(
                        y=alt.Y("setor:N", sort="-x", title=None),
                        x=alt.X("faltante_rs:Q"),
                        text=alt.Text("faltante_rs:Q", format=",.2f"),
                        color=alt.value(ALT_FG),
                    )
                )
                chart_urg = alt.layer(bars_urg, labels_urg)
            else:
                chart_urg = bars_urg

            st.altair_chart(
                alt_dark_theme(chart_urg.properties(height=CHART_H, width="container")),
                use_container_width=True
            )
    # (Se setor_sel != "Todos", não renderiza a seção 4)

    # -----------------------------
    # 5) Ritmo atual × necessário (apenas quando uma LOJA escolhida)
    # -----------------------------
    if loja_sel != "Todas":
        st.markdown("#### Ritmo atual × necessário (por setor)")
        # usa o helper único (mesma lógica do print)
        valid_dows_cfg = {0,1,2,3,4,5}   # seg–sáb; troque se vende domingo
        view = build_metas_tabela(df_hist=df, metas_df=metas_work, loja_sel=loja_sel, valid_dows=valid_dows_cfg)


        st.dataframe(
            view, hide_index=True, use_container_width=True,
            column_config={
                "dias_restantes":   st.column_config.NumberColumn("Dias rest.", format="%d"),
                "faltante_rs":      st.column_config.NumberColumn("Faltante (R$)", format="%.2f"),
                "ritmo_atual_dia":  st.column_config.NumberColumn("Ritmo atual (R$/dia)", format="%.2f"),
                "necessario_dia":   st.column_config.NumberColumn("Necessário (R$/dia)",  format="%.2f"),
                "gap_dia":          st.column_config.NumberColumn("Diferença (R$/dia)",   format="%.2f"),
                "status_ritmo":     st.column_config.TextColumn("Status"),
            },
        )
        # guarde a visão da UI para reutilizar no print
        st.session_state[f"_ritmo_view_{loja_sel}"] = view.copy()


    # -----------------------------
    # 6) Pareto por setor (quando um setor foi escolhido)
    # -----------------------------
    if setor_sel != "Todos":
        st.divider()
        st.markdown("### Produtos mais impactantes para a meta (Pareto)")
        st.caption("Top itens (com % e % acumulado)")

        metas_setor = metas_work[metas_work["setor"] == setor_sel].copy()
        if not metas_setor.empty:
            base_merge = vendas_base.merge(
                metas_setor[["meta_id", "loja_norm", "setor_norm", "inicio", "fim"]],
                on=["loja_norm", "setor_norm"],
                how="inner"
            )
            base_merge["data_dt"] = pd.to_datetime(base_merge["data"])
            base_merge["inicio_dt"] = pd.to_datetime(base_merge["inicio"])
            base_merge["fim_dt"] = pd.to_datetime(base_merge["fim"])
            in_win = (base_merge["data_dt"] >= base_merge["inicio_dt"]) & (base_merge["data_dt"] <= base_merge["fim_dt"])
            vendas_setor_periodo = base_merge[in_win].copy()

            if not vendas_setor_periodo.empty:
                itens = (
                    vendas_setor_periodo
                    .groupby(["codigo_base", "nome"], as_index=False)
                    .agg(venda=("valor", "sum"), qtd=("quantidade", "sum"))
                    .sort_values("venda", ascending=False)
                )
                total_setor = float(itens["venda"].sum())
                itens["share"] = np.where(total_setor > 0, itens["venda"] / total_setor, 0.0)
                itens["cumshare"] = itens["share"].cumsum()

                top_n = st.slider("Top itens para exibir", 5, 50, 20, 1, key="pareto_topn")
                base_itens = itens.head(top_n).copy()
                base_itens["rank"] = np.arange(1, len(base_itens) + 1)

                base_itens["share_txt"] = (base_itens["share"] * 100).map(lambda v: f"{v:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                base_itens["cumshare_txt"] = (base_itens["cumshare"] * 100).map(lambda v: f"{v:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))

                def _width_from_col(series, min_px=80, max_px=520):
                    lens = series.astype(str).map(len)
                    ch = int(np.nanpercentile(lens, 95)) if len(lens) else 10
                    return int(np.clip(40 + ch * 8, min_px, max_px))

                w_cod = _width_from_col(base_itens["codigo_base"], min_px=100, max_px=240)
                w_prod = _width_from_col(base_itens["nome"], min_px=220, max_px=520)

                ROW_H, HDR_H = 34, 38
                TABLE_H = HDR_H + ROW_H * max(8, len(base_itens))
                CHART_H_SYNC = max(200, TABLE_H - 6)

                col_tbl, col_chart = st.columns([1.45, 1.0])

                with col_tbl:
                    st.data_editor(
                        base_itens[["rank", "codigo_base", "nome", "qtd", "venda", "share_txt", "cumshare_txt"]]
                            .rename(columns={
                                "rank": "Rank",
                                "codigo_base": "Código",
                                "nome": "Produto",
                                "qtd": "Qtd",
                                "venda": "Vendas (R$)",
                                "share_txt": "Share",
                                "cumshare_txt": "Acum."
                            }),
                        hide_index=True,
                        use_container_width=True,
                        height=TABLE_H,
                        disabled=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", format="%d", width=70),
                            "Código": st.column_config.TextColumn("Código", width=w_cod),
                            "Produto": st.column_config.TextColumn("Produto", width=w_prod),
                            "Qtd": st.column_config.NumberColumn("Qtd", format="%d", width=90),
                            "Vendas (R$)": st.column_config.NumberColumn("Vendas (R$)", format="%.2f", width=120),
                            "Share": st.column_config.TextColumn("Share", width=90),
                            "Acum.": st.column_config.TextColumn("Acum.", width=90),
                        },
                    )

                with col_chart:
                    st.markdown(f"**Participação (Top {len(base_itens)})**")
                    st.altair_chart(
                        alt.Chart(base_itens).mark_bar().encode(
                            x=alt.X("venda:Q", title="Vendas (R$)", axis=alt.Axis(format=",.2f")),
                            y=alt.Y("nome:N", sort="-x", title=None),
                            tooltip=[
                                alt.Tooltip("codigo_base:N", title="Código"),
                                alt.Tooltip("nome:N", title="Produto"),
                                alt.Tooltip("qtd:Q", title="Qtd"),
                                alt.Tooltip("venda:Q", title="Vendas", format=",.2f"),
                                alt.Tooltip("share:Q", title="Share", format=".2%"),
                                alt.Tooltip("cumshare:Q", title="Acum.", format=".2%")
                            ],
                        ).properties(height=CHART_H_SYNC, width="container")
                        .configure_view(strokeWidth=0),
                        use_container_width=True
                    )

        st.divider()
    # snapshot não filtrado da base de vendas para exportações
    if "vendas_unfiltered" not in st.session_state:
        st.session_state["vendas_unfiltered"] = vendas.copy()

    # -----------------------------
    # 7) Tabela detalhada
    # -----------------------------
    st.markdown("### Tabela detalhada")
    st.dataframe(metas_work.sort_values(["loja", "setor"]).reset_index(drop=True), use_container_width=True)
    st.divider()

    # --- bases para os prints (Metas e Setor) ---
    from datetime import datetime
    import pandas as pd

    metas_src = st.session_state.get("metas_work", globals().get("metas_work"))
    evo_src   = st.session_state.get("evo_daily",  globals().get("evo_daily"))

    if not isinstance(metas_src, pd.DataFrame) or metas_src.empty:
        st.error("`metas_work` não encontrado ou vazio. Carregue-o antes desta seção.")
        st.stop()
    if not isinstance(evo_src, pd.DataFrame) or evo_src.empty:
        st.error("`evo_daily` não encontrado ou vazio. Carregue-o antes desta seção.")
        st.stop()

    # -----------------------------
    # 8) Exportar prints
    # -----------------------------
    if setor_sel == "Todos":
        # ------ PRINT DE METAS (normal) ------
        st.markdown("#### 📸 Exportar ‘print’ da aba Metas")

        # seletor de lojas (opcional)
        met_base_lojas = st.session_state.get("metas_unfiltered", metas_src)
        todas_lojas = sorted(
            met_base_lojas["loja"].dropna().astype(str).unique().tolist()
        ) if "loja" in met_base_lojas.columns else []
        default_lojas = [loja_sel] if (loja_sel != "Todas" and loja_sel in todas_lojas) else []
        lojas_para_print = st.multiselect(
            "Lojas para incluir no print (deixe vazio para usar o escopo atual)",
            options=todas_lojas,
            default=default_lojas,
            key="ms_lojas_print_metas",
        )

        # buffers
        if "metas_print_png" not in st.session_state:
            st.session_state["metas_print_png"] = None
            st.session_state["metas_print_pdf"] = None
            st.session_state["metas_print_zip"] = None

        btn_label = "Gerar ZIP (todas as lojas)" if (lojas_para_print and len(lojas_para_print) > 1) else "Gerar print único (PNG + PDF)"
        if st.button(btn_label, type="primary", key="btn_gerar_print_metas"):
            data1, data2 = gerar_print_metas_png_pdf(
                evo_src,
                metas_src,
                loja_sel,
                setor_sel,
                fmt_currency_br,
                lojas_escolhidas=(lojas_para_print or None),
            )
            if (lojas_para_print and len(lojas_para_print) > 1):
                st.session_state["metas_print_zip"] = data1
                st.session_state["metas_print_png"] = None
                st.session_state["metas_print_pdf"] = None
            else:
                st.session_state["metas_print_zip"] = None
                st.session_state["metas_print_png"] = data1
                st.session_state["metas_print_pdf"] = data2

        # botões de download
        zip_bytes = st.session_state.get("metas_print_zip")
        png_bytes = st.session_state.get("metas_print_png")
        pdf_bytes = st.session_state.get("metas_print_pdf")

        if zip_bytes:
            st.download_button(
                "⬇️ Baixar ZIP (todas as lojas)",
                data=zip_bytes,
                file_name=f"metas_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                key="dl_zip_metas",
            )
        elif (png_bytes is not None) and (pdf_bytes is not None):
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Baixar imagem (PNG)",
                    data=png_bytes,
                    file_name=f"metas_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    key="dl_png_metas",
                )
            with c2:
                st.download_button(
                    "⬇️ Baixar PDF (1 página)",
                    data=pdf_bytes,
                    file_name=f"metas_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf_metas",
                )

    else:
        # ------ PRINT POR SETOR (Top itens) ------
        st.markdown("#### 📸 Exportar `print` por **Setor** (Top itens)")

        # lista de setores com meta (não filtra por setor_sel; lista todos com meta)
        # Monta a lista de setores a partir de UMA BASE NÃO FILTRADA de metas
        # (tenta várias chaves comuns no session_state; cai para metas_src só se não achar outra)
        # Seleciona a primeira base disponível que seja DataFrame e NÃO esteja vazia
        _candidates = [
            st.session_state.get("metas_unfiltered", None),
            st.session_state.get("metas_full", None),
            st.session_state.get("metas_raw", None),
            st.session_state.get("metas_df", None),
            st.session_state.get("metas_source", None),
            globals().get("metas_unfiltered", None),
            globals().get("metas_full", None),
            globals().get("metas_raw", None),
            globals().get("metas_df", None),
            globals().get("metas_source", None),
            metas_src,  # fallback (já validado acima)
        ]
        _met_base = next(
            (c for c in _candidates if isinstance(c, pd.DataFrame) and not c.empty),
            metas_src,
        )
        _met = _met_base.copy()


        _met = _met_base.copy()

        # lista todos os setores QUE TÊM META (>0) — sem filtrar por setor_sel, nem por loja
        if "meta" in _met.columns and "setor" in _met.columns:
            setores_com_meta = sorted(
                _met.loc[_met["meta"].fillna(0) > 0, "setor"].dropna().astype(str).unique().tolist()
            )
        else:
            setores_com_meta = []

        # se não achou uma base não filtrada, cai para o escopo atual (evita "No results")
        if not setores_com_meta and "meta" in metas_src.columns and "setor" in metas_src.columns:
            setores_com_meta = sorted(
                metas_src.loc[metas_src["meta"].fillna(0) > 0, "setor"].dropna().astype(str).unique().tolist()
            )

        default_setores = [setor_sel] if (setor_sel != "Todos" and setor_sel in setores_com_meta) else []
        # estado inicial do multiselect de setores
        if "ms_setores_meta" not in st.session_state:
            st.session_state["ms_setores_meta"] = default_setores


        top_n = st.number_input("Top itens para exibir", min_value=5, max_value=50, value=20, step=1, key="topN_setor")
        # botões auxiliares para seleção
        c_all, c_none, c_inv = st.columns(3)

        with c_all:
            if st.button("Selecionar todos", key="btn_sel_all_setores"):
                st.session_state["ms_setores_meta"] = list(setores_com_meta)

        with c_none:
            if st.button("Limpar", key="btn_clear_setores"):
                st.session_state["ms_setores_meta"] = []

        with c_inv:
            if st.button("Inverter seleção", key="btn_inv_setores"):
                atual = set(st.session_state.get("ms_setores_meta", []))
                st.session_state["ms_setores_meta"] = [s for s in setores_com_meta if s not in atual]


        setores_para_print = st.multiselect(
            "Setores (apenas os que possuem meta)",
            options=setores_com_meta,
            default=st.session_state.get("ms_setores_meta", default_setores),
            key="ms_setores_meta",
        )



        # buffers
        if "setores_print_zip" not in st.session_state:
            st.session_state["setores_print_zip"] = None
            st.session_state["setores_print_png"] = None
            st.session_state["setores_print_pdf"] = None

        c1, c2 = st.columns(2)

        # (1) Loja atual
        with c1:
            if st.button("Gerar (Loja atual)", type="primary", key="btn_setor_loja"):
                d1, d2 = gerar_print_setores_topitens(
                    vendas_df=vendas,                       # DF de itens
                    setores_escolhidos=setores_para_print,
                    top_n=int(top_n),
                    fmt_currency_br=fmt_currency_br,
                    loja_sel=(loja_sel if loja_sel != "Todas" else None),
                    todas_as_lojas=False,
                )
                if d2 is None:
                    st.session_state["setores_print_zip"] = d1
                    st.session_state["setores_print_png"] = None
                    st.session_state["setores_print_pdf"] = None
                else:
                    st.session_state["setores_print_zip"] = None
                    st.session_state["setores_print_png"] = d1
                    st.session_state["setores_print_pdf"] = d2

        # (2) Todas as lojas (1 PDF por loja, várias páginas — 1 por setor)
        with c2:
            if st.button("Gerar para **todas as lojas** (1 PDF por loja)", key="btn_setor_todas"):
                import pandas as pd

                # 2a) pegar a base NÃO FILTRADA de vendas
                vendas_base = st.session_state.get("vendas_unfiltered")
                if not isinstance(vendas_base, pd.DataFrame) or vendas_base.empty:
                    vendas_base = vendas  # fallback (pode estar filtrada)

                # 2b) montar a lista de lojas a partir de uma base de VENDAS NÃO FILTRADA
                lojas_all = sorted(
                    vendas_base["loja"].astype(str).str.strip().dropna().unique().tolist()
                )


                # 2c) chamar a função em modo TODAS AS LOJAS (ignora loja_sel)
                d1, _ = gerar_print_setores_topitens(
                    vendas_df=vendas_base,                 # <<< base NÃO filtrada
                    setores_escolhidos=setores_para_print,
                    top_n=int(top_n),
                    fmt_currency_br=fmt_currency_br,
                    loja_sel=None,                         # <<< NÃO passe a loja atual
                    start_date=d_ini,                      # << adicionar
                    end_date=d_fim,                        # << adicionar
                    todas_as_lojas=True,                   # <<< ativa o modo multi-loja
                    lojas_ord=lojas_all,
                )
                st.session_state["setores_print_zip"] = d1
                st.session_state["setores_print_png"] = None
                st.session_state["setores_print_pdf"] = None



        # downloads (prioriza ZIP)
        zip_b = st.session_state.get("setores_print_zip")
        png_b = st.session_state.get("setores_print_png")
        pdf_b = st.session_state.get("setores_print_pdf")

        if zip_b:
            st.download_button(
                "⬇️ Baixar ZIP",
                data=zip_b,
                file_name=f"topitens_setores_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                key="dl_zip_setor",
            )
        elif (png_b is not None) and (pdf_b is not None):
            cc1, cc2 = st.columns(2)
            with cc1:
                st.download_button(
                    "⬇️ PNG",
                    data=png_b,
                    file_name=f"topitens_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    key="dl_png_setor",
                )
            with cc2:
                st.download_button(
                    "⬇️ PDF",
                    data=pdf_b,
                    file_name=f"topitens_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf_setor",
                )
    # ----- fim do bloco da elif "🎯 Metas" -----


elif secao == "📆 Comparar Dias":
    st.subheader("Comparar Dias (A × B)")
    mask = pd.Series(True, index=vendas.index)
    if reg_sel != "Todas":  mask &= vendas["regiao"] == reg_sel
    if loja_sel != "Todas": mask &= vendas["loja"]   == loja_sel
    if setor_sel != "Todos": mask &= vendas["setor"] == setor_sel

    base_cmp = vendas[mask].copy()
    if base_cmp.empty:
        st.warning("Sem dados para os filtros de Região/Loja/Setor atuais.")
        st.stop()

    unq_dates = sorted(base_cmp["data"].dropna().unique().tolist())
    if len(unq_dates) < 2:
        st.warning("Precisamos de pelo menos duas datas para comparar."); st.stop()

    dB_default = unq_dates[-1]
    dA_default = unq_dates[-2] if len(unq_dates) >= 2 else unq_dates[0]

    c1, c2, c3 = st.columns([1.2,1.2,1.0])
    with c1:
        dA = st.date_input("Dia A", value=dA_default, min_value=unq_dates[0], max_value=unq_dates[-1], key="cmp_dA")
    with c2:
        dB = st.date_input("Dia B", value=dB_default, min_value=unq_dates[0], max_value=unq_dates[-1], key="cmp_dB")
    with c3:
        top_k = st.slider("Top (gráficos)", 5, 20, 10, 1, key="cmp_topk")

    dfA = base_cmp[base_cmp["data"] == dA].copy()
    dfB = base_cmp[base_cmp["data"] == dB].copy()
    if dfA.empty or dfB.empty:
        st.warning("Uma das datas não possui dados com os filtros aplicados."); st.stop()

    def kpis(df_: pd.DataFrame) -> dict:
        val = df_["valor"].sum()
        qtd = df_["quantidade"].sum()
        skus = df_["codigo_base"].nunique()
        return {"vendas": val, "unidades": qtd, "skus": skus, "preco_medio": (val/qtd if qtd else 0.0), "ticket_aprox": (val/skus if skus else 0.0)}

    kA, kB = kpis(dfA), kpis(dfB)

    def _fmt_num_int(n):
        try: return f"{int(round(n)):,}".replace(",", ".")
        except Exception: return str(n)
    def _fmt_num_2(n):
        try: return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception: return str(n)

    def kpi_card(title: str, value_str: str, delta: float, *, money: bool=False, help_text: str|None=None):
        if delta is None or pd.isna(delta): delta = 0.0
        arrow = "▲" if delta >= 0 else "▼"
        color = "#22c55e" if delta >= 0 else "#ef4444"
        bg    = "rgba(34,197,94,.15)" if delta >= 0 else "rgba(239,68,68,.15)"
        delta_txt = f"{arrow} {fmt_currency_br(abs(delta))}" if money else (f"{arrow} {_fmt_num_int(abs(delta))}" if abs(delta) >= 1 else f"{arrow} {_fmt_num_2(abs(delta))}")
        help_attr = f'title="{help_text}"' if help_text else ""
        html = f"""
        <div style="padding:6px 0;">
            <div style="color:#9aa0a6;font-size:13px;margin-bottom:2px;">{title}</div>
            <div style="font-size:32px;font-weight:600;line-height:1.1;">{value_str}</div>
            <span {help_attr}
                  style="display:inline-block;margin-top:8px;padding:3px 10px;border-radius:999px;
                         font-size:12px;background:{bg};color:{color};">
                {delta_txt}
            </span>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    delta_v  = float(kB["vendas"]       - kA["vendas"])
    delta_q  = float(kB["unidades"]     - kA["unidades"])
    delta_s  = float(kB["skus"]         - kA["skus"])
    delta_pm = float(kB["preco_medio"]  - kA["preco_medio"])
    delta_t  = float(kB["ticket_aprox"] - kA["ticket_aprox"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card(f"Vendas — {dA.strftime('%d/%m/%Y')}", fmt_currency_br(kA["vendas"]), delta_v, money=True,
                 help_text=f"Δ vs {dB.strftime('%d/%m/%Y')}: {fmt_currency_br(delta_v)}")
    with c2:
        kpi_card("Unidades", _fmt_num_int(kA["unidades"]), delta_q, money=False,
                 help_text=f"Δ: {_fmt_num_int(delta_q)} un")
    with c3:
        kpi_card("SKUs", _fmt_num_int(kA["skus"]), delta_s, money=False,
                 help_text=f"Δ: {_fmt_num_int(delta_s)} SKUs")
    with c4:
        kpi_card("Preço Médio", fmt_currency_br(kA["preco_medio"]), delta_pm, money=True,
                 help_text=f"Δ: {fmt_currency_br(delta_pm)}")
    with c5:
        kpi_card("Ticket aprox.", fmt_currency_br(kA["ticket_aprox"]), delta_t, money=True,
                 help_text=f"Δ: {fmt_currency_br(delta_t)}")

    def side_by_side(dfA_, dfB_, key_col: str, val_col: str, label: str):
        a = dfA_.groupby(key_col, as_index=False)[val_col].sum().rename(columns={val_col:"valor", key_col:"chave"})
        a["dia"] = dA.strftime("%d/%m")
        b = dfB_.groupby(key_col, as_index=False)[val_col].sum().rename(columns={val_col:"valor", key_col:"chave"})
        b["dia"] = dB.strftime("%d/%m")
        both = pd.concat([a, b], ignore_index=True)
        top_keys = (both.groupby("chave")["valor"].sum().sort_values(ascending=False).head(top_k).index.tolist())
        both = both[both["chave"].isin(top_keys)]
        return (alt.Chart(both).mark_bar().encode(
            y=alt.Y("chave:N", sort="-x", title=None),
            x=alt.X("valor:Q", title=label),
            color=alt.Color("dia:N", legend=alt.Legend(title="Dia")),
            tooltip=["chave", "dia", alt.Tooltip("valor:Q", title=label, format=",.2f")],
        ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0))

    st.markdown("#### Setores — comparação")
    st.altair_chart(side_by_side(dfA, dfB, "setor", "valor", "Vendas (R$)"))
    st.markdown("#### Lojas — comparação")
    st.altair_chart(side_by_side(dfA, dfB, "loja", "valor", "Vendas (R$)"))
    st.markdown("#### Itens — comparação")
    dfA["item"] = dfA["nome"].fillna(dfA["codigo"])
    dfB["item"] = dfB["nome"].fillna(dfB["codigo"])
    st.altair_chart(side_by_side(
        dfA.rename(columns={"item": "chave"}),
        dfB.rename(columns={"item": "chave"}),
        "chave", "valor", "Vendas (R$)"
    ))

elif secao == "📄 Relatório":
    st.subheader("Relatório (gráficos + prints)")
    top_k = st.slider("Tamanho dos 'Top' (k)", 5, 20, 10, 1)

    left, right = st.columns([2.2, 1.2])

    # KPIs sempre (usando df já filtrado por período/região/loja/setor)
    with right:
        st.markdown("### Indicadores")
        total_valor = float(df["valor"].sum())
        total_qtd   = float(df["quantidade"].sum())
        skus_total  = int(df["codigo_base"].nunique())
        c1, c2 = st.columns(2)
        c1.metric("Vendas (Total)", fmt_currency_br(total_valor))
        c2.metric("Unidades", f"{int(total_qtd):,}".replace(",", "."))
        c1.metric("Preço Médio", fmt_currency_br(total_valor/total_qtd if total_qtd else 0))
        c2.metric("Ticket Médio (aprox.)", fmt_currency_br(total_valor/skus_total if skus_total else 0))

    # Quando NENHUM setor específico está selecionado: manter seu relatório por setor
    if setor_sel == "Todos":
        loja_titulo = loja_sel if loja_sel != "Todas" else "todas as lojas"

        por_setor = (
            df.groupby("setor", as_index=False)
              .agg(valor_total=("valor","sum"),
                   quantidade_total=("quantidade","sum"),
                   skus=("codigo_base","nunique"))
        )
        por_setor["preco_medio"]       = por_setor["valor_total"] / por_setor["quantidade_total"].replace(0, np.nan)
        por_setor["ticket_medio_aprox"]= por_setor["valor_total"] / por_setor["skus"].replace(0, np.nan)

        with left:
            base = por_setor.sort_values("valor_total", ascending=False)
            st.caption(f"Top {top_k} Vendas por Setor — {loja_titulo.upper()} (mostrando {min(top_k,len(base))} de {len(base)})")
            st.altair_chart(compact_chart(base.head(top_k), "valor_total", "setor", "Vendas (R$)"))

            base = por_setor.dropna(subset=["preco_medio"]).sort_values("preco_medio", ascending=False)
            st.caption(f"Top {top_k} Preço Médio por Setor")
            st.altair_chart(compact_chart(base.head(top_k), "preco_medio", "setor", "Preço Médio (R$)"))

            base = por_setor.dropna(subset=["ticket_medio_aprox"]).sort_values("ticket_medio_aprox", ascending=False)
            st.caption(f"Top {top_k} Ticket Médio (aprox.) por Setor")
            st.altair_chart(compact_chart(base.head(top_k), "ticket_medio_aprox", "setor", "Ticket Médio (R$)"))

        # ===== prints por loja (igual você já tinha) =====
        st.divider()
        st.markdown("#### 📸 Gerar prints deste relatório por loja")
        lojas_all = sorted(df["loja"].unique().tolist())
        col_sel_all, col_ms = st.columns([0.22, 0.78])
        select_all = col_sel_all.checkbox("Selecionar todas", value=False, key="select_all_lojas")
        lojas_escolhidas = col_ms.multiselect("Lojas", lojas_all, default=(lojas_all if select_all else []), key="multi_lojas")

        def gerar_print_loja(df_total: pd.DataFrame, loja: str, k: int) -> bytes:
            df_loja = df_total[df_total["loja"] == loja].copy()
            por_setor_lj = (
                df_loja.groupby("setor", as_index=False)
                       .agg(valor_total=("valor","sum"),
                            quantidade_total=("quantidade","sum"),
                            skus=("codigo_base","nunique"))
            )
            por_setor_lj["preco_medio"]       = por_setor_lj["valor_total"]/por_setor_lj["quantidade_total"].replace(0, np.nan)
            por_setor_lj["ticket_medio_aprox"]= por_setor_lj["valor_total"]/por_setor_lj["skus"].replace(0, np.nan)

            por_item = (
                df_loja.groupby(["codigo_base"], as_index=False)
                       .agg(valor_total=("valor","sum"),
                            quantidade_total=("quantidade","sum"),
                            nome=("nome","first"))
            )
            top5 = por_item.sort_values("valor_total", ascending=False).head(5)
            bot5 = por_item.sort_values("valor_total", ascending=True).head(5)

            total_valor = df_loja["valor"].sum()
            total_qtd   = df_loja["quantidade"].sum()
            skus_total  = df_loja["codigo_base"].nunique()
            preco_medio = total_valor/total_qtd if total_qtd else 0
            ticket_aprox= total_valor/skus_total if skus_total else 0

            plt.close("all")
            fig = plt.figure(figsize=(16, 9), dpi=160)
            gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[2,2,1], height_ratios=[1,1,1], wspace=0.35, hspace=0.45)

            ax1 = fig.add_subplot(gs[0,:2])
            d1 = por_setor_lj.sort_values("valor_total", ascending=False).head(k)
            ax1.barh(d1["setor"], d1["valor_total"]); ax1.invert_yaxis()
            ax1.set_title(f"Top {min(k,len(d1))} Vendas por Setor — {loja}", fontsize=11)
            ax1.set_xlabel("Vendas (R$)")
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}".replace(",", ".")))

            ax2 = fig.add_subplot(gs[1,:2])
            d2 = por_setor_lj.dropna(subset=["preco_medio"]).sort_values("preco_medio", ascending=False).head(k)
            ax2.barh(d2["setor"], d2["preco_medio"]); ax2.invert_yaxis()
            ax2.set_title("Top Preço Médio por Setor", fontsize=11)
            ax2.set_xlabel("Preço Médio (R$)")
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.2f}".replace(".", ",").replace(",", ".")))

            ax3 = fig.add_subplot(gs[2,:2])
            d3 = por_setor_lj.dropna(subset=["ticket_medio_aprox"]).sort_values("ticket_medio_aprox", ascending=False).head(k)
            ax3.barh(d3["setor"], d3["ticket_medio_aprox"]); ax3.invert_yaxis()
            ax3.set_title("Top Ticket Médio (aprox.)", fontsize=11)
            ax3.set_xlabel("Ticket Médio (R$)")
            ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.2f}".replace(".", ",").replace(",", ".")))

            ax4 = fig.add_subplot(gs[:,2]); ax4.axis("off")
            x0, y = 0.0, 1.0
            def t(txt, size=14, bold=False, dy=0.0):
                nonlocal y
                y += dy
                ax4.text(x0, y, txt, fontsize=size, fontweight="bold" if bold else "normal", va="top")
                y -= 0.06

            t("Indicadores", 18, True, dy=0.02)
            ax4.text(x0, y, "Vendas (Total)", fontsize=12)
            ax4.text(x0, y-0.06, fmt_currency_br(total_valor), fontsize=20, fontweight="bold"); y -= 0.14
            ax4.text(x0, y, "Unidades", fontsize=12)
            ax4.text(x0, y-0.06, f"{int(total_qtd):,}".replace(",", "."), fontsize=20, fontweight="bold"); y -= 0.14
            ax4.text(x0, y, "Preço Médio", fontsize=12)
            ax4.text(x0, y-0.06, fmt_currency_br(preco_medio), fontsize=20, fontweight="bold"); y -= 0.14
            ax4.text(x0, y, "Ticket Médio (aprox.)", fontsize=12)
            ax4.text(x0, y-0.06, fmt_currency_br(ticket_aprox), fontsize=20, fontweight="bold"); y -= 0.10

            y -= 0.02
            ax4.text(x0, y, f"Top 5 ({loja})", fontsize=12, fontweight="bold"); y -= 0.06
            if top5.empty:
                ax4.text(x0, y, "- sem dados", fontsize=10); y -= 0.04
            else:
                for _, r in top5.iterrows():
                    ax4.text(x0, y, f"• {r['nome']} ({fmt_currency_br(r['valor_total'])} / {int(r['quantidade_total'])} un)", fontsize=9); y -= 0.04

            y -= 0.02
            ax4.text(x0, y, f"Bottom 5 ({loja})", fontsize=12, fontweight="bold"); y -= 0.06
            if bot5.empty:
                ax4.text(x0, y, "- sem dados", fontsize=10); y -= 0.04
            else:
                for _, r in bot5.iterrows():
                    ax4.text(x0, y, f"• {r['nome']} ({fmt_currency_br(r['valor_total'])} / {int(r['quantidade_total'])} un)", fontsize=9); y -= 0.04

            fig.suptitle(f"Relatório — {loja}  •  {datetime.now().strftime('%d/%m/%Y')}", fontsize=12, y=0.98)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        previews = []
        if st.button("Gerar prints (PNG em ZIP)", type="primary", disabled=not lojas_escolhidas):
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for loja in lojas_escolhidas:
                    img = gerar_print_loja(df, loja, top_k)
                    zf.writestr(f"relatorio_{loja.replace(' ', '_')}.png", img)
                    previews.append((loja, img))
            zip_buf.seek(0)
            st.download_button(
                "⬇️ Baixar ZIP com prints",
                data=zip_buf.getvalue(),
                file_name=f"prints_relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip"
            )
        if previews:
            st.markdown("##### Pré-visualizações")
            for loja, img in previews:
                st.markdown(f"**{loja}**")
                st.image(img, use_column_width=True)

    # Quando um SETOR está selecionado: mostrar relatório por itens (nunca vazio)
    else:
        with left:
            st.caption(f"Top {top_k} Itens — {loja_sel if loja_sel!='Todas' else 'todas as lojas'} • {setor_sel}")
            base_itens = (df.groupby(["codigo_base","nome"], as_index=False)
                            .agg(quantidade=("quantidade","sum"),
                                 venda=("valor","sum"))
                            .sort_values("venda", ascending=False)
                            .head(top_k))
            base_itens["rank"] = np.arange(1, len(base_itens)+1)
            st.dataframe(
                base_itens[["rank","codigo_base","nome","quantidade","venda"]]
                    .rename(columns={"rank":"Rank","codigo_base":"Código","nome":"Produto",
                                     "quantidade":"Qtd","venda":"Vendas (R$)"}),
                use_container_width=True
            )
            st.altair_chart(
                alt.Chart(base_itens).mark_bar().encode(
                    x=alt.X("venda:Q", title="Vendas (R$)", axis=alt.Axis(format=",.2f")),
                    y=alt.Y("nome:N", sort="-x", title=None),
                    tooltip=["codigo_base","nome",
                             alt.Tooltip("quantidade:Q", title="Qtd", format=",.0f"),
                             alt.Tooltip("venda:Q", title="Vendas", format=",.2f")]
                ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0),
                use_container_width=True
            )

elif secao == "🗓️ Histórico":
    st.subheader("Detalhamento do Histórico (dados filtrados)")
    top_k_h = st.slider("Top (gráficos)", 5, 20, 10, 1, key="top_k_hist")

    c1, c2, c3 = st.columns(3)
    c1.metric("Vendas no período", fmt_currency_br(df["valor"].sum()))
    c2.metric("Unidades", f"{int(df['quantidade'].sum()):,}".replace(",", "."))
    c3.metric("Itens únicos", f"{df['codigo'].nunique():,}".replace(",", "."))

    # Evolução total (ordenada por data real)
    evo = (df.assign(data_dt=pd.to_datetime(df["data"]))
                .groupby("data_dt", as_index=False)["valor"].sum()
                .sort_values("data_dt"))
    st.altair_chart(
        alt.Chart(evo).mark_line(point=True).encode(
            x=alt.X("data_dt:T", title="Dia", axis=alt.Axis(format="%d/%m/%Y", labelAngle=90)),
            y=alt.Y("valor:Q",   title="Vendas (R$)"),
            tooltip=[alt.Tooltip("data_dt:T", title="Dia", format="%d/%m/%Y"),
                        alt.Tooltip("valor:Q",   title="Vendas", format=",.2f")]
        ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0)
    )

    # Evolução por Loja (cada loja = uma linha)
    st.markdown("#### Lojas — linhas por dia")
    evo_loja = (
        df.assign(data_dt=pd.to_datetime(df["data"]))
            .groupby(["data_dt","loja"], as_index=False)["valor"].sum()
            .sort_values("data_dt")
    )
    st.altair_chart(
        alt.Chart(evo_loja).mark_line(point=True).encode(
            x=alt.X("data_dt:T", title="Dia", axis=alt.Axis(format="%d/%m/%Y", labelAngle=90)),
            y=alt.Y("valor:Q", title="Vendas (R$)"),
            color=alt.Color("loja:N", title="Loja"),
            tooltip=["loja",
                        alt.Tooltip("data_dt:T", title="Dia", format="%d/%m/%Y"),
                        alt.Tooltip("valor:Q",   title="Vendas", format=",.2f")],
        ).properties(height=CHART_H, width="container").configure_view(strokeWidth=0)
    )

    # Top Lojas
    top_lojas = df.groupby("loja", as_index=False)["valor"].sum().sort_values("valor", ascending=False).head(top_k_h)
    st.altair_chart(compact_chart(top_lojas, "valor", "loja", "Vendas (R$)"))

    # Top Itens
    df["item_label"] = df["nome"].fillna(df["codigo"])
    top_itens = (df.groupby(["codigo","item_label"], as_index=False)["valor"].sum()
                    .rename(columns={"item_label":"Item"})
                    .sort_values("valor", ascending=False).head(top_k_h))
    st.altair_chart(compact_chart(top_itens, "valor", "Item", "Vendas (R$)"))

    # Insights rápidos
    with st.expander("🧠 Insights rápidos", expanded=False):
        if not evo.empty:
            best_row = evo.loc[evo["valor"].idxmax()]
            worst_row = evo.loc[evo["valor"].idxmin()]
            c1, c2 = st.columns(2)
            c1.metric("Melhor dia", best_row["data_dt"].strftime("%d/%m/%Y"), fmt_currency_br(best_row["valor"]))
            c2.metric("Pior dia",   worst_row["data_dt"].strftime("%d/%m/%Y"), fmt_currency_br(worst_row["valor"]))

        evo2 = evo.copy()
        evo2["dow"] = evo2["data_dt"].dt.dayofweek
        nomes = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        evo2["dow_name"] = evo2["dow"].map({i:n for i,n in enumerate(nomes)})
        by_dow = (evo2.groupby("dow_name", as_index=False)["valor"].sum()
                        .sort_values("valor", ascending=False))
        st.caption("Vendas por dia da semana (soma no período)")
        st.altair_chart(compact_chart(by_dow, "valor", "dow_name", "Vendas (R$)"))

        itens = df.groupby("codigo", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
        if not itens.empty:
            itens["share"] = itens["valor"] / itens["valor"].sum()
            itens["cumshare"] = itens["share"].cumsum()
            n80 = int((itens["cumshare"] <= 0.80).sum())
            total_itens = int(itens.shape[0])
            st.write(f"**{n80} itens ({(n80/total_itens):.0%})** respondem por **≈80%** da venda do período.")

    st.markdown("##### Tabela detalhada")
    cols_show = ["data","loja","setor","codigo","nome","quantidade","valor"]
    st.dataframe(
        df[cols_show].sort_values(["data","valor"], ascending=[False,False]).reset_index(drop=True),
        use_container_width=True
    )
    csv_hist = df[cols_show].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv_hist,
        file_name=f"historico_filtrado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# =====================================================
# Diagnóstico / fontes (fora do if/elif principal)
# =====================================================
diag = st.expander("ℹ️ Diagnóstico / fontes", expanded=False)
with diag:
    st.write("**produtos.json:**", produtos_json_path if prod_map_local else "não encontrado (ou usando upload)")
    st.write("**agrupamentos.json (lojas):**", loja_json_path if loja_json_local else "não encontrado")
    st.write("**setores.json:**", setor_json_path if setor_json_local else "não encontrado")
    st.write(f"**Lojas únicas (originais):** {vendas['loja_orig'].nunique()} | **após mapeamento:** {vendas['loja'].nunique()}")
