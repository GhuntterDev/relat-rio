# app_streamlit_avaliacao.py
# Streamlit: App de avaliação de colaboradores (multi-lojas, multi-arquivos)
# - Lê CSVs com layout por posição: A=Data, B=Setor, C=Colaborador, D=Velocidade, F=Atendimento, H=Qualidade, J=Ajuda
# - Vazios contam como 0 nas notas
# - Suporta múltiplos arquivos (consolida e infere "Loja" do nome do arquivo)
# - Filtros de período/loja/setor
# - Resumos por pessoa, por setor, por loja e geral
# - Downloads de resumos em CSV
# - Gráficos (ranking e evolução temporal)

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# CONFIGURAÇÕES
# ==========================
st.set_page_config(page_title="Avaliação de Colaboradores", layout="wide")

# Mapeamento de colunas por letra
NOTAS_COLS = {
    "Velocidade": "D",
    "Atendimento": "F",
    "Qualidade": "H",
    "Ajuda": "J",
}
COL_DATA = "A"       # Data e hora (dd/mm/aaaa ou dd/mm/aaaa hh:mm)
COL_SETOR = "B"      # Setor
COL_NOME = "C"       # Nome do colaborador

# ==========================
# HELPERS
# ==========================

def infer_loja_from_filename(name: str) -> str:
    """Tenta extrair o nome da loja do nome do arquivo.
    Ex.: 'Avaliação Colaborador Loja  - Carioca (respostas) - Resumo.csv' -> 'Carioca'"""
    if not name:
        return "Desconhecida"
    m = re.search(r"-\s*([^-()]+?)\s*(?:\(|-|\.)", name)
    if m:
        loja = m.group(1).strip()
        return loja
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", name)
    return tokens[-1] if tokens else "Desconhecida"


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BOM e tira espaços de nomes de colunas."""
    df.columns = [c.encode("utf-8").decode("utf-8-sig").strip() if isinstance(c, str) else c for c in df.columns]
    return df


def get_col_by_letter(df: pd.DataFrame, letter: str) -> str:
    """Retorna o nome da coluna correspondente à letra (A,B,C...)."""
    idx = ord(letter.upper()) - ord("A")
    if not (0 <= idx < len(df.columns)):
        raise ValueError(f"Letra de coluna inválida: {letter}")
    return df.columns[idx]


def parse_datetime_ptbr(series: pd.Series) -> pd.Series:
    """Converte datas em dd/mm/aaaa (com ou sem hora)."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def cast_notas_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Converte notas para numérico com segurança (vazio -> 0.0)."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
            continue
        obj = df[c]
        # se, por algum motivo, vier um DataFrame (colunas duplicadas), pega a primeira
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        df[c] = pd.to_numeric(obj, errors="coerce").fillna(0.0)
    return df


def build_summary(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    nota_cols = ["Velocidade", "Atendimento", "Qualidade", "Ajuda"]
    work = df.copy()
    work["Avaliações"] = 1
    if by_cols:  # agrupado
        means = work.groupby(by_cols, dropna=False)[nota_cols].mean(numeric_only=True)
        counts = work.groupby(by_cols, dropna=False)["Avaliações"].sum()
        out = means.join(counts).reset_index()
    else:  # geral sem groupby
        out = pd.DataFrame({
            "Velocidade": [work["Velocidade"].mean()],
            "Atendimento": [work["Atendimento"].mean()],
            "Qualidade": [work["Qualidade"].mean()],
            "Ajuda": [work["Ajuda"].mean()],
            "Avaliações": [work["Avaliações"].sum()],
        })
    out["Média Geral"] = out[["Velocidade", "Atendimento", "Qualidade", "Ajuda"]].mean(axis=1)
    out[["Velocidade", "Atendimento", "Qualidade", "Ajuda", "Média Geral"]] = \
        out[["Velocidade", "Atendimento", "Qualidade", "Ajuda", "Média Geral"]].round(2)
    if by_cols:
        out = out.sort_values("Média Geral", ascending=False)
    return out


def download_button_csv(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# ==========================
# FONTE DE DADOS (Upload ou Google Sheets)
# ==========================
st.sidebar.header("Configurações")
fonte = st.sidebar.radio(
    "Fonte de dados",
    ["Upload CSV", "Google Sheets"],
    help="Escolha 'Google Sheets' para atualizar automaticamente de uma planilha."
)

# Parâmetros Google Sheets (usando Service Account via st.secrets)
# st.secrets["gcp_service_account"] deve conter as credenciais JSON da service account
# e a planilha deve ser compartilhada com o e-mail da service account.
if fonte == "Google Sheets":
    spreadsheet_id = st.sidebar.text_input(
        "Spreadsheet ID (Google Sheets)",
        help="O ID é a parte entre /d/ e /edit da URL."
    )
    ttl_min = st.sidebar.number_input(
        "Intervalo de atualização automática (minutos)",
        min_value=0, value=2, step=1,
        help="0 desativa o auto-refresh."
    )
    st.sidebar.button("↻ Atualizar agora", on_click=lambda: st.cache_data.clear())
else:
    ttl_min = 0

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

# ==========================
# LOAD & PREP
# ==========================
frames = []

if fonte == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Envie um ou mais CSVs (uma loja por arquivo ou várias):",
        type=["csv"],
        accept_multiple_files=True,
        help="Formato esperado por posição: Data(A), Setor(B), Colaborador(C), Velocidade(D), Atendimento(F), Qualidade(H), Ajuda(J).",
    )

    if uploaded_files:
        for f in uploaded_files:
            loja = infer_loja_from_filename(f.name)
            content = f.read()
            df = None
            for sep in [";", ","]:
                try:
                    df_try = pd.read_csv(io.BytesIO(content), sep=sep, dtype=str, encoding="utf-8")
                    if df_try.shape[1] >= 4:
                        df = df_try
                        break
                except Exception:
                    pass
            if df is None or df.empty:
                st.error(f"Não consegui ler o arquivo: {f.name}")
                continue

            df = normalize_colnames(df)

            # mapeia letras -> nomes (por posição)
            try:
                col_data = get_col_by_letter(df, COL_DATA)
                col_setor = get_col_by_letter(df, COL_SETOR)
                col_nome  = get_col_by_letter(df, COL_NOME)
                col_vel = get_col_by_letter(df, NOTAS_COLS["Velocidade"])
                col_atd = get_col_by_letter(df, NOTAS_COLS["Atendimento"])
                col_qlt = get_col_by_letter(df, NOTAS_COLS["Qualidade"])
                col_ajd = get_col_by_letter(df, NOTAS_COLS["Ajuda"])
            except Exception as e:
                st.error(f"Erro ao mapear colunas por letra no arquivo {f.name}: {e}")
                continue

            rec = pd.DataFrame({
                "Data": df[col_data],
                "Setor": df[col_setor],
                "Colaborador": df[col_nome],
                "Velocidade": df[col_vel],
                "Atendimento": df[col_atd],
                "Qualidade": df[col_qlt],
                "Ajuda": df[col_ajd],
            })

            rec["Data"] = parse_datetime_ptbr(rec["Data"])
            rec = cast_notas_safe(rec, ["Velocidade", "Atendimento", "Qualidade", "Ajuda"])

            # ignorar linhas sem nome (NaN, vazio, "nan", "none")
            before_rows = len(rec)
            rec = rec[rec["Colaborador"].notna()].copy()
            rec["Colaborador"] = rec["Colaborador"].astype(str).str.strip()
            rec = rec[(rec["Colaborador"] != "") & (rec["Colaborador"].str.lower() != "nan") & (rec["Colaborador"].str.lower() != "none")]

            rec["Loja"] = loja
            frames.append(rec)

elif fonte == "Google Sheets":
    def _fetch_from_gsheets(spreadsheet_id: str) -> list[pd.DataFrame]:
        import gspread
        # autentica via st.secrets
        creds = st.secrets.get("gcp_service_account", None)
        if creds is None:
            st.error("Credenciais não encontradas em st.secrets['gcp_service_account'].")
            return []
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(spreadsheet_id)
        # lê TODAS as abas; cada título de aba vira a coluna Loja
        dfs = []
        for ws in sh.worksheets():
            title = ws.title
            values = ws.get_all_values()
            if not values:
                continue
            if len(values) >= 2:
                df = pd.DataFrame(values[1:], columns=values[0])
            else:
                df = pd.DataFrame(values)
            dfs.append((title, df))
        return dfs

    @st.cache_data(ttl=lambda: ttl_min * 60 if ttl_min else None, show_spinner=False)
    def load_gsheets(spreadsheet_id: str):
        return _fetch_from_gsheets(spreadsheet_id)

    if spreadsheet_id:
        try:
            sheets = load_gsheets(spreadsheet_id)
            for title, df in sheets:
                # normaliza colunas e usa posição A,B,C,D,F,H,J
                df = normalize_colnames(df)
                try:
                    col_data = get_col_by_letter(df, COL_DATA)
                    col_setor = get_col_by_letter(df, COL_SETOR)
                    col_nome  = get_col_by_letter(df, COL_NOME)
                    col_vel = get_col_by_letter(df, NOTAS_COLS["Velocidade"])
                    col_atd = get_col_by_letter(df, NOTAS_COLS["Atendimento"])
                    col_qlt = get_col_by_letter(df, NOTAS_COLS["Qualidade"])
                    col_ajd = get_col_by_letter(df, NOTAS_COLS["Ajuda"])
                except Exception as e:
                    st.warning(f"Aba '{title}': problema ao mapear colunas por posição — {e}")
                    continue

                rec = pd.DataFrame({
                    "Data": df[col_data],
                    "Setor": df[col_setor],
                    "Colaborador": df[col_nome],
                    "Velocidade": df[col_vel],
                    "Atendimento": df[col_atd],
                    "Qualidade": df[col_qlt],
                    "Ajuda": df[col_ajd],
                })
                rec["Data"] = parse_datetime_ptbr(rec["Data"])
                rec = cast_notas_safe(rec, ["Velocidade", "Atendimento", "Qualidade", "Ajuda"])

                # ignorar linhas sem nome
                rec = rec[rec["Colaborador"].notna()].copy()
                rec["Colaborador"] = rec["Colaborador"].astype(str).str.strip()
                rec = rec[(rec["Colaborador"] != "") & (rec["Colaborador"].str.lower() != "nan") & (rec["Colaborador"].str.lower() != "none")]

                rec["Loja"] = title  # usa o nome da aba como Loja
                frames.append(rec)
        except Exception as e:
            st.error(f"Erro ao carregar Google Sheets: {e}")

if frames:
    data = pd.concat(frames, ignore_index=True)

    # limites de data
    if data["Data"].notna().any():
        date_min = pd.to_datetime(data["Data"].min()).date()
        date_max = pd.to_datetime(data["Data"].max()).date()
    else:
        today = datetime.today().date()
        date_min = date_max = today

    # filtros
    d_ini, d_fim = st.sidebar.date_input(
        "Período",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    lojas_sel = st.sidebar.multiselect(
        "Filtrar por loja",
        options=sorted([x for x in data["Loja"].dropna().unique()]),
        default=sorted([x for x in data["Loja"].dropna().unique()]),
    )
    setores_sel = st.sidebar.multiselect(
        "Filtrar por setor",
        options=sorted([x for x in data["Setor"].dropna().unique()]),
        default=sorted([x for x in data["Setor"].dropna().unique()]),
    )

    # aplica filtros
    mask = pd.Series(True, index=data.index)
    if d_ini and d_fim:
        mask &= data["Data"].between(pd.to_datetime(d_ini), pd.to_datetime(d_fim) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), inclusive="both")
    if lojas_sel:
        mask &= data["Loja"].isin(lojas_sel)
    if setores_sel:
        mask &= data["Setor"].isin(setores_sel)

    df_f = data.loc[mask].copy()

    # ==========================
    # LAYOUT PRINCIPAL
    # ==========================
    st.title("📊 Avaliação de Colaboradores")

    # KPIs topo
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Avaliações (linhas)", int(df_f.shape[0]))
    with col2:
        st.metric("Média Velocidade", round(df_f["Velocidade"].mean() if len(df_f) else 0, 2))
    with col3:
        st.metric("Média Atendimento", round(df_f["Atendimento"].mean() if len(df_f) else 0, 2))
    with col4:
        st.metric("Média Qualidade", round(df_f["Qualidade"].mean() if len(df_f) else 0, 2))
    with col5:
        st.metric("Média Ajuda", round(df_f["Ajuda"].mean() if len(df_f) else 0, 2))

    st.markdown("### 🔎 Prévia dos dados filtrados")
    st.dataframe(df_f.sort_values("Data", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("👤 Resumo por Pessoa")
    resumo_pessoa = build_summary(df_f, by_cols=["Colaborador", "Loja", "Setor"])
    st.dataframe(resumo_pessoa, use_container_width=True)
    download_button_csv(resumo_pessoa, "Baixar CSV (por pessoa)", "resumo_por_pessoa.csv")

    st.markdown("### 🏬 Resumo por Setor")
    resumo_setor = build_summary(df_f, by_cols=["Setor", "Loja"])
    st.dataframe(resumo_setor, use_container_width=True)
    download_button_csv(resumo_setor, "Baixar CSV (por setor)", "resumo_por_setor.csv")

    st.markdown("### 🏪 Resumo por Loja")
    resumo_loja = build_summary(df_f, by_cols=["Loja"])
    st.dataframe(resumo_loja, use_container_width=True)
    download_button_csv(resumo_loja, "Baixar CSV (por loja)", "resumo_por_loja.csv")

    st.markdown("### 🌐 Resumo Geral (todas as lojas)")
    resumo_geral = build_summary(df_f, by_cols=[])
    st.dataframe(resumo_geral, use_container_width=True)
    download_button_csv(resumo_geral, "Baixar CSV (geral)", "resumo_geral.csv")

    # Botão: gerar relatório Excel por loja (ordenado por Média Geral)
    st.markdown("---")
    st.subheader("📄 Exportar relatório por loja (Excel)")

    def gerar_relatorio_excel_por_loja(df_in: pd.DataFrame) -> bytes:
        import io
        import pandas as pd
        # usamos o resumo por pessoa já calculado (df_f -> resumo_pessoa)
        base = build_summary(df_in, by_cols=["Colaborador", "Loja", "Setor"]).copy()
        # garante ordem por média geral desc
        base = base.sort_values("Média Geral", ascending=False)
        # ordena colunas conforme a planilha de referência
        cols = [
            "Colaborador", "Loja", "Setor",
            "Velocidade", "Atendimento", "Qualidade", "Ajuda",
            "Avaliações", "Média Geral",
        ]
        base = base[[c for c in cols if c in base.columns]]

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            workbook  = writer.book
            header_fmt = workbook.add_format({
                "bold": True,
                "bg_color": "#B7B7B7",  # igual ao arquivo exemplo
                "font_color": "#000000",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            })
            num2_fmt   = workbook.add_format({"num_format": "0.00", "border": 1, "align": "center", "valign": "vcenter"})
            int0_fmt   = workbook.add_format({"num_format": "0", "border": 1, "align": "center", "valign": "vcenter"})
            text_fmt   = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

            # uma aba por loja
            for loja, g in base.groupby("Loja", dropna=False):
                sheet = str(loja) if pd.notna(loja) else "(Sem Loja)"
                g = g.copy().sort_values("Média Geral", ascending=False)
                g.to_excel(writer, index=False, sheet_name=sheet)
                ws = writer.sheets[sheet]
                # Larguras e cabeçalho
                widths = {
                    "Colaborador": 36,
                    "Loja": 18,
                    "Setor": 20,
                    "Velocidade": 12,
                    "Atendimento": 12,
                    "Qualidade": 12,
                    "Ajuda": 10,
                    "Avaliações": 12,
                    "Média Geral": 12,
                }
                for col_idx, col_name in enumerate(g.columns, start=1):
                    ws.set_column(col_idx-1, col_idx-1, widths.get(col_name, 14))
                    ws.write(0, col_idx-1, col_name, header_fmt)

                # Formatação por coluna
                for row_idx in range(1, len(g) + 1):
                    ws.set_row(row_idx, 18)
                # aplica formatos numéricos
                col_map = {name: idx for idx, name in enumerate(g.columns, start=0)}
                for nm in ["Velocidade", "Atendimento", "Qualidade", "Ajuda", "Média Geral"]:
                    if nm in col_map:
                        c = col_map[nm]
                        ws.set_column(c, c, widths.get(nm, 12), num2_fmt)
                if "Avaliações" in col_map:
                    c = col_map["Avaliações"]
                    ws.set_column(c, c, widths.get("Avaliações", 10), int0_fmt)
                for nm in ["Colaborador", "Loja", "Setor"]:
                    if nm in col_map:
                        c = col_map[nm]
                        ws.set_column(c, c, widths.get(nm, 18), text_fmt)

                # Filtro e congelar o cabeçalho
                ws.autofilter(0, 0, len(g), len(g.columns)-1)
                ws.freeze_panes(1, 0)

                # Formatação condicional na Média Geral (barra de cor)
                if "Média Geral" in col_map:
                    c = col_map["Média Geral"]
                    ws.conditional_format(1, c, len(g), c, {
                        "type": "3_color_scale",
                        "min_color": "#FCA5A5",  # vermelho claro
                        "mid_color": "#FDE68A",  # amarelo
                        "max_color": "#86EFAC",  # verde claro
                    })

        buf.seek(0)
        return buf.getvalue()

    # Botão para gerar e baixar
    if st.button("⬇️ Gerar relatório Excel (uma aba por loja)"):
        xlsx_bytes = gerar_relatorio_excel_por_loja(df_f)
        st.download_button(
            label="Baixar relatório.xlsx",
            data=xlsx_bytes,
            file_name="relatorio_por_loja.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Gráficos (Altair)
    try:
        import altair as alt

        st.markdown("---")
        st.subheader("📈 Gráficos")
        # Ranking por pessoa (Top 20)
        top_pessoas = resumo_pessoa.head(20)
        if not top_pessoas.empty:
            chart_pessoas = alt.Chart(top_pessoas).mark_bar().encode(
                x=alt.X("Colaborador:N", sort="-y"),
                y=alt.Y("Média Geral:Q"),
                color="Loja:N",
                tooltip=["Colaborador", "Loja", "Setor", "Velocidade", "Atendimento", "Qualidade", "Ajuda", "Média Geral", "Avaliações"],
            ).properties(height=420)
            st.altair_chart(chart_pessoas, use_container_width=True)

        # Ranking por loja
        if not resumo_loja.empty:
            chart_loja = alt.Chart(resumo_loja).mark_bar().encode(
                x=alt.X("Loja:N", sort="-y"),
                y=alt.Y("Média Geral:Q"),
                tooltip=["Loja", "Velocidade", "Atendimento", "Qualidade", "Ajuda", "Média Geral", "Avaliações"],
            ).properties(height=360)
            st.altair_chart(chart_loja, use_container_width=True)

        # Evolução temporal (média diária geral)
        df_time = df_f.copy()
        if not df_time.empty:
            df_time["Data_dia"] = df_time["Data"].dt.date
            notas = ["Velocidade", "Atendimento", "Qualidade", "Ajuda"]
            evol = df_time.groupby("Data_dia")[notas].mean().reset_index().melt("Data_dia", var_name="Métrica", value_name="Média")
            line = alt.Chart(evol).mark_line(point=True).encode(
                x="Data_dia:T",
                y="Média:Q",
                color="Métrica:N",
                tooltip=["Data_dia", "Métrica", "Média"],
            ).properties(height=350)
            st.altair_chart(line, use_container_width=True)
    except Exception as e:
        st.info(f"Gráficos não exibidos (Altair indisponível?): {e}")

else:
    st.title("📊 Avaliação de Colaboradores")
    st.info(
        "Envie um ou mais arquivos CSV na barra lateral para começar. "
        "Formato esperado (por posição): Data(A), Setor(B), Colaborador(C), Velocidade(D), Atendimento(F), Qualidade(H), Ajuda(J)."
    )

# Rodapé de dicas
with st.expander("ℹ️ Dicas e validações"):
    st.markdown(
        """
- **Vazios contam como 0** nas colunas de nota.
- A coluna **Data** é interpretada no formato **dd/mm/aaaa** (com ou sem hora).
- O nome da **Loja** é inferido a partir do **nome do arquivo**. Se preferir, renomeie os arquivos para deixar claro (ex.: "... - Carioca (...).csv").
- Você pode **baixar** os resumos (CSV) logo abaixo de cada tabela.
- Ideias extras: ranking por setor dentro de cada loja, variância/desvio-padrão das notas, e exportar tudo em um único Excel.
        """
    )
