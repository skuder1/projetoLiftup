import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from prophet import Prophet
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime
import requests

st.set_page_config(layout="wide", page_title="LiftUp")

if 'selected_currency' not in st.session_state:
    st.session_state.selected_currency = "BRL"


def load_ticker_list():
    try:
        return pd.read_csv("tickers_ibra.csv", usecols=['ticker', 'company'])
    except Exception as e:
        st.error(f"Erro ao ler arquivo CSV: {str(e)}")
        return None


def get_live_exchange_rates():
    """Busca cotações estáveis usando a Frankfurter API (ECB)."""
    url = "https://api.frankfurter.app/latest"
    params = {
        "from": "USD",
        "to": "BRL,EUR"
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        rates = data.get("rates")
        if not rates or "BRL" not in rates or "EUR" not in rates:
            raise ValueError("Resposta incompleta da API")

        usd_to_brl = rates["BRL"]
        usd_to_eur = rates["EUR"]
        eur_to_brl = usd_to_brl / usd_to_eur

        return {
            "USD_to_BRL": usd_to_brl,
            "EUR_to_BRL": eur_to_brl,
            "BRL_to_USD": 1 / usd_to_brl,
            "BRL_to_EUR": 1 / eur_to_brl,
        }

    except Exception as e:
        st.warning(f"Falha na API ({e}). Usando valores médios.")
        return {
            "USD_to_BRL": 5.3,
            "EUR_to_BRL": 6.15,
            "BRL_to_USD": 1 / 5.3,
            "BRL_to_EUR": 1 / 6.15,
        }

exchange_rates = get_live_exchange_rates()

def convert_currency(value, ticker):
    if st.session_state.selected_currency == "BRL":
        if ticker[-1].isdigit():
            return value
        else:
            return value * exchange_rates["USD_to_BRL"]
    elif st.session_state.selected_currency == "USD":
        if ticker[-1].isdigit():
            return value * exchange_rates["BRL_to_USD"]
        else:
            return value
    elif st.session_state.selected_currency == "EUR":
        if ticker[-1].isdigit():
            return value * exchange_rates["BRL_to_EUR"]
        else:
            return value / (exchange_rates["EUR_to_BRL"] / exchange_rates["USD_to_BRL"])


def fetch_stock_data(tickers, start_date, end_date):
    try:
        formatted_tickers = [
            f"{ticker}.SA" if ticker[-1].isdigit() and not ticker.endswith('.SA')
            else ticker for ticker in tickers
        ]

        data = yf.download(formatted_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

        if isinstance(data, pd.Series):
            return pd.DataFrame({tickers[0].replace('.SA', ''): data})

        data.columns = [col.replace('.SA', '') for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None


def get_company_name(ticker, ticker_list):
    try:
        return ticker_list.loc[ticker_list['ticker'] == ticker, 'company'].iloc[0]
    except:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('longName', ticker)
        except:
            return ticker


def get_currency_symbol():
    if st.session_state.selected_currency == "BRL":
        return "R$"
    elif st.session_state.selected_currency == "USD":
        return "$"
    elif st.session_state.selected_currency == "EUR":
        return "€"


def calculate_metrics(price_series, ticker):
    try:
        current_price = convert_currency(price_series.iloc[-1], ticker)
        return {
            'current_price': current_price,
            'daily_return': (price_series.iloc[-1] / price_series.iloc[-2] - 1) * 100,
            'accumulated_return': ((price_series.iloc[-1] / price_series.iloc[0] - 1) * 100)
        }
    except Exception:
        return None


def create_metric_card(ticker, prices, ticker_list, mygrid):
    card = mygrid.container(border=True)

    if len(prices[ticker]) < 2:
        st.error(f"Dados insuficientes para {ticker}")
        return

    try:
        metrics = calculate_metrics(prices[ticker], ticker)
        if not metrics:
            st.error(f"Erro ao calcular métricas para {ticker}")
            return

        header_col1, header_col2 = card.columns([1, 3])

        with header_col1:
            if ticker[-1].isdigit():  # Ação brasileira
                try:
                    st.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{ticker}.png',
                             width=100)
                except:
                    st.write(ticker)
            else:  # Ação internacional
                try:
                    st.image(
                        f'https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/{ticker}.png',
                        width=100)
                except:
                    st.write(ticker)

        with header_col2:
            st.markdown(f"### {get_company_name(ticker, ticker_list)}")
            st.caption(ticker)

        metric_col1, metric_col2, metric_col3 = card.columns(3)

        currency = get_currency_symbol()

        metric_col1.metric(
            label="Preço Atual",
            value=f"{currency} {metrics['current_price']:.2f}",
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
    if prices.empty:
        return None

    data = prices.copy()

    # Converte os preços para a moeda selecionada
    for column in data.columns:
        data[column] = data[column].apply(lambda x: convert_currency(x, column))

    if price_type == "Retorno Diário":
        data = data.pct_change()
        title, y_title = 'Retornos Diários', 'Retorno (%)'
    elif price_type == "Retorno Acumulado":
        data = (1 + data.pct_change()).cumprod() - 1
        title, y_title = 'Retornos Acumulados', 'Retorno Acumulado (%)'
    else:
        title = 'Preços Ajustados'
        currency = get_currency_symbol()
        y_title = f"Preço ({currency})"

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


def train_prophet_model(data, ticker):
    df = pd.DataFrame({'ds': data.index, 'y': data[ticker]})

    model = Prophet(daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10)
    model.fit(df)

    return model


def make_prediction(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def create_prediction_metrics(forecast_data, current_price, ticker):
    currency = get_currency_symbol()

    last_prediction = convert_currency(forecast_data['yhat'].iloc[-1], ticker)
    current_price = convert_currency(current_price, ticker)
    predicted_return = ((last_prediction / current_price) - 1) * 100

    upper_bound = convert_currency(forecast_data['yhat_upper'].iloc[-1], ticker)
    lower_bound = convert_currency(forecast_data['yhat_lower'].iloc[-1], ticker)
    uncertainty_range = upper_bound - lower_bound
    uncertainty_percentage = (uncertainty_range / last_prediction) * 100

    return {
        'predicted_price': f"{currency} {last_prediction:.2f}",
        'predicted_return': f"{predicted_return:.2f}%",
        'uncertainty': f"±{uncertainty_percentage:.1f}%"
    }

def create_prediction_chart(historical_data, forecast_data, ticker, ticker_list):
    fig = go.Figure()

    company_name = get_company_name(ticker, ticker_list)
    currency = get_currency_symbol()

    historical_values = historical_data[ticker].apply(lambda x: convert_currency(x, ticker))
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_values,
        mode='lines',
        name='Dados Históricos',
        line=dict(color='blue')
    ))

    forecast_values = forecast_data['yhat'].apply(lambda x: convert_currency(x, ticker))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_values,
        mode='lines',
        name='Previsão',
        line=dict(color='#0a84ff', dash='dash')
    ))

    upper_values = forecast_data['yhat_upper'].apply(lambda x: convert_currency(x, ticker))
    lower_values = forecast_data['yhat_lower'].apply(lambda x: convert_currency(x, ticker))

    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=upper_values,
        mode='lines',
        name='Limite Superior',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=lower_values,
        mode='lines',
        name='Margem de Erro',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(10, 132, 255, 0.1)'
    ))

    fig.update_layout(
        title=f'Previsão de Preços - {company_name} ({ticker})',
        xaxis_title='Data',
        yaxis_title=f'Preço ({currency})',
        height=600,
        hovermode='x unified'
    )

    return fig


def prediction_tab(prices, ticker_list):
    st.title("Previsão de Preços")

    if prices is None or prices.empty:
        st.warning("Selecione pelo menos uma empresa para ver as previsões.")
        return

    # Seleção do ativo para previsão
    selected_ticker = st.selectbox(
        "Selecione o Ativo para Previsão",
        options=prices.columns,
        format_func=lambda x: f"{get_company_name(x, ticker_list)} ({x})"
    )

    prediction_days = st.slider(
        "Período de Previsão (dias)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )

    try:
        with st.spinner("Gerando previsões..."):
            model = train_prophet_model(prices, selected_ticker)

            forecast = make_prediction(model, periods=prediction_days)

            current_price = prices[selected_ticker].iloc[-1]
            metrics = create_prediction_metrics(forecast, current_price, selected_ticker)

            # Exibe métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Preço Previsto (Final)",
                    metrics['predicted_price'],
                    metrics['predicted_return']
                )
            with col2:
                st.metric(
                    "Margem de Erro",
                    metrics['uncertainty']
                )
            with col3:
                st.metric(
                    "Dias Previstos",
                    f"{prediction_days} dias"
                )

            # Gera e exibe o gráfico
            fig = create_prediction_chart(prices, forecast, selected_ticker, ticker_list)
            st.plotly_chart(fig, width='stretch')

            # Componentes do modelo
            if st.checkbox("Mostrar Componentes da Previsão"):
                fig_comp = model.plot_components(forecast)
                st.pyplot(fig_comp)

    except Exception as e:
        st.error(f"Erro ao gerar previsões: {str(e)}")


def build_sidebar():
    st.sidebar.image("images/logo.png")

    ticker_list = load_ticker_list()
    if ticker_list is None:
        return None, None, None

    formatted_options = [
        f"{row['company']} ({row['ticker']})"
        for _, row in ticker_list.iterrows()
    ]

    selected_companies = st.sidebar.multiselect(
        label="Selecione as Empresas",
        options=sorted(formatted_options),
        placeholder='Empresas'
    )

    if not selected_companies:
        return None, None, None

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("De", value=datetime(2023, 6, 1), format="DD/MM/YYYY")
    with col2:
        end_date = st.date_input("Até", value="today", format="DD/MM/YYYY")

    tickers = [company.split('(')[-1].strip(')') for company in selected_companies]

    prices = fetch_stock_data(tickers, start_date, end_date)

    if prices is None or prices.empty:
        st.error("Não foi possível obter dados para os tickers selecionados")
        return None, None, None

    st.sidebar.markdown("<div style='min-height: 20vh'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='centered-selectbox'>", unsafe_allow_html=True)
    st.session_state.selected_currency = st.sidebar.selectbox(
    "Moeda",
    options=["BRL", "USD", "EUR"],
    index=0,
    key="currency_selector",
    label_visibility="collapsed" 
)

    return tickers, prices, ticker_list


def main():
    with st.sidebar:
        tickers, prices, ticker_list = build_sidebar()

    # Criação das abas
    tab1, tab2 = st.tabs(["📈 Histórico", "🔮 Previsão"])

    with tab1:
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
                        st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {str(e)}")
        else:
            st.warning("Por favor, selecione pelo menos uma empresa na barra lateral.")

    with tab2:
        prediction_tab(prices, ticker_list)


if __name__ == "__main__":
    main()