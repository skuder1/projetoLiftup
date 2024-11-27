import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

# Configuração da página
st.set_page_config(layout="wide")

@st.cache_data(ttl=3600)
def load_ticker_list():
    try:
        return pd.read_csv("tickers_ibra.csv", usecols=['ticker', 'company'])
    except Exception as e:
        st.error(f"Erro ao ler arquivo CSV: {str(e)}")
        return None

@st.cache_data(ttl=300, show_spinner="Carregando dados...")
def fetch_stock_data(tickers, start_date, end_date):
    """Função para buscar dados de múltiplos tickers de forma otimizada"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        if isinstance(data, pd.Series):
            return pd.DataFrame({tickers[0].replace('.SA', ''): data})
        
        data.columns = [col.replace('.SA', '') for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None

@st.cache_data
def get_company_name(ticker, ticker_list):
    """Retorna o nome da empresa a partir do ticker"""
    try:
        return ticker_list.loc[ticker_list['ticker'] == ticker, 'company'].iloc[0]
    except:
        return ticker

def calculate_metrics(price_series):
    """Calcula métricas financeiras"""
    try:
        return {
            'current_price': price_series.iloc[-1],
            'daily_return': (price_series.iloc[-1] / price_series.iloc[-2] - 1) * 100,
            'accumulated_return': ((price_series.iloc[-1] / price_series.iloc[0] - 1) * 100)
        }
    except Exception:
        return None

def create_metric_card(ticker, prices, ticker_list, mygrid):
    """Cria card com métricas para um ticker"""
    card = mygrid.container(border=True)
    
    if len(prices[ticker]) < 2:
        st.error(f"Dados insuficientes para {ticker}")
        return
        
    try:
        metrics = calculate_metrics(prices[ticker])
        if not metrics:
            st.error(f"Erro ao calcular métricas para {ticker}")
            return

        header_col1, header_col2 = card.columns([1, 3])
        with header_col1:
            try:
                st.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{ticker}.png', width=100)
            except:
                st.write(ticker)
        
        with header_col2:
            st.markdown(f"### {get_company_name(ticker, ticker_list)}")
            st.caption(ticker)
        
        metric_col1, metric_col2, metric_col3 = card.columns(3)
        
        metric_col1.metric(
            label="Preço Atual",
            value=f"R$ {metrics['current_price']:.2f}",
            help="Último preço disponível"
        )
        
        metric_col2.metric(
            label="Retorno Diário",
            value=f"{metrics['daily_return']:.2f}%",
            help="Variação em relação ao dia anterior"
        )
        
        metric_col3.metric(
            label="Retorno Acumulado",
            value=f"{metrics['accumulated_return']:.2f}%",
            help="Variação desde o início do período"
        )
        
        style_metric_cards(
            background_color='rgba(55, 55, 55, 0.1)',
            border_size_px=1,
            border_radius_px=10,
            box_shadow=True
        )
    except Exception as e:
        st.error(f"Erro ao processar dados para {ticker}: {str(e)}")

def create_price_chart(prices, price_type, ticker_list):
    """Cria gráfico de preços/retornos"""
    if prices.empty:
        return None
        
    data = prices.copy()
    
    if price_type == "Retorno Diário":
        data = data.pct_change()
        title, y_title = 'Retornos Diários', 'Retorno (%)'
    elif price_type == "Retorno Acumulado":
        data = (1 + data.pct_change()).cumprod() - 1
        title, y_title = 'Retornos Acumulados', 'Retorno Acumulado (%)'
    else:
        title, y_title = 'Preços Ajustados', 'Preço'

    fig = go.Figure()
    for column in data.columns:
        company_name = get_company_name(column, ticker_list)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=f"{company_name} ({column})"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title=y_title,
        height=700,
        hovermode='x unified'
    )
    
    return fig

def build_sidebar():
    """Constrói a barra lateral"""
    st.image("images/logo-250-100-transparente.png")
    
    ticker_list = load_ticker_list()
    if ticker_list is None:
        return None, None, None

    formatted_options = [
        f"{row['company']} ({row['ticker']})" 
        for _, row in ticker_list.iterrows()
    ]
    
    selected_companies = st.multiselect(
        label="Selecione as Empresas",
        options=sorted(formatted_options),
        placeholder='Empresas'
    )
    
    if not selected_companies:
        return None, None, None

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("De", value=datetime(2023,6,1), format="DD/MM/YYYY")
    with col2:
        end_date = st.date_input("Até", value="today", format="DD/MM/YYYY")

    tickers = [f"{company.split('(')[-1].strip(')')}.SA" for company in selected_companies]
    
    prices = fetch_stock_data(tickers, start_date, end_date)
    
    if prices is None or prices.empty:
        st.error("Não foi possível obter dados para os tickers selecionados")
        return None, None, None
        
    return tickers, prices, ticker_list

def main():
    """Função principal"""
    with st.sidebar:
            tickers, prices, ticker_list = build_sidebar()

    if tickers and prices is not None:
        st.title("Histórico de Preços")
        
        metrics_container = st.container()
        chart_container = st.container()
        
        with metrics_container:
            st.subheader("Visão Geral dos Ativos")
            mygrid = grid(2, 2, 2, vertical_align="top")
            for ticker in prices.columns:
                create_metric_card(ticker, prices, ticker_list, mygrid)
        
        with chart_container:
            st.markdown("---")
            price_type = st.selectbox(
                "Selecione o Tipo de Gráfico",
                ["Preço Ajustado", "Retorno Diário", "Retorno Acumulado"]
            )
            
            try:
                fig = create_price_chart(prices, price_type, ticker_list)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráfico: {str(e)}")
    else:
        st.warning("Por favor, selecione pelo menos uma empresa na barra lateral.")

if __name__ == "__main__":
    main()