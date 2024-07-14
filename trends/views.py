from django.shortcuts import render
from django.http import HttpResponse
from pytrends.request import TrendReq
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import plotly.graph_objs as go
import numpy as np
import google.generativeai as genai
import markdown
import os
import json
import time

# Define the path to the storage file
STORAGE_FILE = 'keyword_data.json'

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def feedback(request):
    return render(request, 'feedback.html')

def faq(request):
    return render(request, 'faq.html')

def trending(request):
    return render(request,'trending.html')


def save_feedback(request):
    if request.method == 'POST':
        # Handle saving feedback data if needed
        return HttpResponse(status=200)
    else:
        return HttpResponse(status=405)  # Method Not Allowed

def save_to_text_file(request):
    if request.method == 'POST':
        try:
            # Ensure directory exists
            save_directory = os.path.join(os.path.dirname(__file__), 'Customer_Data')
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Save data to text file
            data = request.body.decode('utf-8')  # Assuming data is sent as JSON from frontend
            with open(os.path.join(save_directory, 'data.txt'), 'a') as f:
                f.write(data + '\n')

            return HttpResponse(status=200)
        except Exception as e:
            print(e)
            return HttpResponse(status=500)  # Internal Server Error
    else:
        return HttpResponse(status=405)  # Method Not Allowed

def load_data_from_file(keyword):
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            stored_data = json.load(f)
        return stored_data.get(keyword)
    return None

def fetch_google_trends_data(pytrends, keyword, retries=5):
    for i in range(retries):
        try:
            pytrends.build_payload(kw_list=[keyword], timeframe='today 5-y', geo='IN')
            df = pytrends.interest_over_time()
            if not df.empty:
                return df
        except Exception as e:
            if '429' in str(e):
                wait_time = (2 ** i) + (np.random.randint(0, 1000) / 1000)  # Exponential backoff with jitter
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(f"Failed to fetch data for keyword: {keyword} after {retries} retries")

def complex_graph_data_to_text(graph_data):
    description = f"Graph titled '{graph_data['title']}' showing data over {graph_data['x']} with values on {graph_data['y']}. "
    description += "Data points are: " + ", ".join([f"({date}, {value})" for date, value in graph_data['historical_data']])
    description += f". Forecasted data: " + ", ".join([f"({date}, {value:.2f})" for date, value in graph_data['forecasted_data']])
    description += f". Forecast period: {graph_data['forecast_periods']} weeks."
    return description

def predict(request):
    if request.method == 'POST':
        keywords = request.POST.getlist('keywords')
        forecast_periods = int(request.POST.get('forecast_periods', 0))
        csv_file = request.FILES.get('csv_file')
        user_query = request.POST.get('user_query', '')  # Get user query from the form

        historical_data = {}
        forecast_data = {}
        graph_descriptions = []

        # Process uploaded CSV file if provided
        if csv_file:
            df = pd.read_csv(csv_file)
            
            # Get the first column header dynamically
            first_column = df.columns[0]
            df[first_column] = pd.to_datetime(df[first_column], dayfirst=True)
            df.set_index(first_column, inplace=True)
            keyword = df.columns[0]
            df = df.rename(columns={keyword: 'interest'})

            # Fit and forecast with Exponential Smoothing
            model = ExponentialSmoothing(df['interest'], trend='add', seasonal='add', seasonal_periods=52)
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=forecast_periods)

            df['interest'] = np.maximum(df['interest'], 0)
            forecast = np.maximum(forecast, 0)

            df.index = df.index.date
            forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods + 1, freq='W')[1:].date

            historical_data[keyword] = df
            forecast_data[keyword] = pd.Series(forecast, index=forecast_index)

            historical_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(df.index.tolist(), df['interest'].tolist())]
            forecasted_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(forecast_index.tolist(), forecast.tolist())]

            graph_data = {
                'title': keyword,
                'x': 'Date',
                'y': 'Interest',
                'historical_data': historical_data_points,
                'forecasted_data': forecasted_data_points,
                'forecast_periods': forecast_periods
            }
            graph_descriptions.append(complex_graph_data_to_text(graph_data))

        # Process keywords if provided
        if keywords:
            pytrends = TrendReq(timeout=(10, 60))
            for keyword in keywords:
                cached_data = load_data_from_file(keyword)
                if cached_data:
                    historical_dates = [pd.to_datetime(date).date() for date in cached_data['dates']]
                    historical_interest = cached_data['interest']
                    df = pd.DataFrame({'interest': historical_interest}, index=pd.to_datetime(historical_dates))
                else:
                    try:
                        df = fetch_google_trends_data(pytrends, keyword)
                        df = df.drop(['isPartial'], axis=1)
                        df = df.rename(columns={keyword: 'interest'})
                        data_to_save = {
                            'dates': df.index.strftime('%Y-%m-%d').tolist(),
                            'interest': df['interest'].tolist()
                        }
                    except Exception as e:
                        return HttpResponse(f"Error: {str(e)}")

                model = ExponentialSmoothing(df['interest'], trend='add', seasonal='add', seasonal_periods=52)
                fit_model = model.fit()
                forecast = fit_model.forecast(steps=forecast_periods)

                df['interest'] = np.maximum(df['interest'], 0)
                forecast = np.maximum(forecast, 0)

                df.index = df.index.date
                forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods + 1, freq='W')[1:].date

                historical_data[keyword] = df
                forecast_data[keyword] = pd.Series(forecast, index=forecast_index)

                historical_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(df.index.tolist(), df['interest'].tolist())]
                forecasted_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(forecast_index.tolist(), forecast.tolist())]

                graph_data = {
                    'title': keyword,
                    'x': 'Date',
                    'y': 'Interest',
                    'historical_data': historical_data_points,
                    'forecasted_data': forecasted_data_points,
                    'forecast_periods': forecast_periods
                }
                graph_descriptions.append(complex_graph_data_to_text(graph_data))

        combined_graph_description = "\n\n".join(graph_descriptions)

        # Configure and use Google Generative AI
        genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key

        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"You are provided with graph data representing the historical and forecasted interest in the products: {', '.join(keywords)}. Your task is to analyze this data and provide detailed business insights, including trends, future possibilities, and considerations for different demographics. Tell all the basic data and other information to the user as business analyst. In conclusion section, be blunt as to whether you should invest in the product or not with reason. Be practical and explain like an expert. give proper conclusion whether its comparing or providing suggestions regarding the product/products.. REMEMBER TO KEEP MOST OF THE BIASES OR VARIABLE IN MIND ATLEAST WHICH YOU CAN HEL WITH"

        # If the user provided a query, add it to the prompt
        if user_query:
            user_query_section = f"\nAfter conclusion please\nUser Query: {user_query}\n Please provide a specific answer to this query based on the data provided. DONT GIBVE DIPLOMATIC ANSWERS SUGGEST WHAT YOU THINK"
            prompt += user_query_section

        combined_prompt = f"{prompt}\nGraph Data:\n{combined_graph_description}"
        response = model.generate_content(combined_prompt)

        summary_html = markdown.markdown(response.text)

        # Plotting with Plotly
        traces = []
        annotations = []

        for keyword, historical_df in historical_data.items():
            forecast_series = forecast_data[keyword]

            traces.append(go.Scatter(
                x=historical_df.index,
                y=historical_df['interest'],
                mode='lines+markers',
                name=f'{keyword} (Historical)'
            ))
            traces.append(go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                mode='lines+markers',
                name=f'{keyword} (Forecast)',
                line=dict(dash='dash')
            ))

            peak_date = historical_df['interest'].idxmax()
            peak_value = historical_df['interest'].max()

            annotation_y = peak_value + 5 if len(annotations) % 2 == 0 else peak_value - 5

            annotations.append(dict(
                x=peak_date,
                y=annotation_y,
                xref='x',
                yref='y',
                text=f'{keyword} demand peak is here\n{peak_date.strftime("%Y-%m-%d")}',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-80 if len(annotations) % 2 == 0 else 80,
                bgcolor='rgba(255, 0, 0, 0.7)',
                font=dict(size=14, color='white'),
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            ))

        layout = go.Layout(
            title='Result Graph: ',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Interest'),
            template='plotly_dark',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#ffffff'),
            legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top', orientation='h'),
            autosize=True,
            annotations=annotations
        )

        fig = go.Figure(data=traces, layout=layout)
        plot_div = fig.to_html(full_html=False, default_height='100%', default_width='100%')

        keyword_search_links = [(keyword, f"https://www.amazon.in/s?k={keyword}") for keyword in keywords]

        return render(request, 'result.html', {
            'plot_div': plot_div,
            'summary_string': summary_html,
            'keyword_search_links': keyword_search_links
        })

    return render(request, 'index.html')

