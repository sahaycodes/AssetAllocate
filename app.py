from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import re
import plotly.express as px
from datetime import datetime
import math
import os 
import pickle

app = Flask(__name__)

PICKLE_FILE = 'data.pkl'

def save_data_to_pickle(data, file_name):
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {file_name}")
    except Exception as e:
        print(f"Error saving data to pickle: {e}")

def load_data_from_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        print(f"Data successfully loaded from {file_name}")
        return data
    except Exception as e:
        print(f"Error loading data from pickle: {e}")
        return None

def get_data(tickers, start_date='2014-01-01'):
    try:
        data = yf.download(tickers, start=start_date)['Adj Close']
        print("Data successfully fetched from yfinance")
        return data
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return None

def load_or_fetch_data(assets, start_date='2014-01-01'):
    #
    #Load data from a pickle file if available and up-to-date.
    #Fetch from yfinance if the data is missing or needs to be updated.
    #"""
    if os.path.exists(PICKLE_FILE):
        print("Loading data from pickle file...")
        data = load_data_from_pickle(PICKLE_FILE)
        if data is not None:
            # Check if all requested assets are present
            missing_assets = [asset for asset in assets if asset not in data.columns]
            if not missing_assets:
                print("All requested assets are present in the pickle file.")
                return data
            else:
                print(f"Missing assets in pickle file: {missing_assets}")

    print("Fetching data from yfinance...")
    data = get_data(assets, start_date)
    if data is not None:
        save_data_to_pickle(data, PICKLE_FILE)
    return data        

def optimize_portfolio(df,tickers):
    num_of_portfolios = 5000
    sim_df = monteCarlo(df,tickers ,num_of_portfolios)

    sim_df['Volatility'] = sim_df['Volatility'].round(2)
    idx = sim_df.groupby('Volatility')['Returns'].idxmax()
    max_df = sim_df.loc[idx].reset_index(drop=True)
    max_df = max_df.sort_values(by='Volatility').reset_index(drop=True)
    max_df['Weights'] = max_df['Weights'].apply(lambda x: {ticker: weight for ticker, weight in zip(tickers, x)})
    max_df = max_df.to_dict(orient='records')

    # Selecting the portfolio with the highest Sharpe Ratio
    max_returns = sim_df.loc[sim_df['Returns'].idxmax()]
    optimal_weights = max_returns['Weights']
    # Creating DataFrame for weights
    weights_df = pd.DataFrame(optimal_weights, columns=['Weights'])
    weights_df.index = tickers  # Assign tickers as index

    # Expected annual return, volatility, and Sharpe ratio
    mean_returns = df.pct_change(fill_method=None).mean()
    cov_matrix = df.pct_change(fill_method=None).cov() * 252
    annual_return = np.sum(mean_returns * optimal_weights) * 252
    port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (annual_return - 0.02) / port_volatility
    # print("done")
    return weights_df, annual_return, port_volatility, sharpe_ratio, max_df

def monteCarlo(df,assets,num_of_portfolios=10000):
    log_returns = np.log(1 + df.pct_change())

    all_weights = np.zeros((num_of_portfolios, len(assets)))

    ret_arr = np.zeros(num_of_portfolios)
    vol_arr = np.zeros(num_of_portfolios)
    sharpe_arr = np.zeros(num_of_portfolios)

    for i in range(num_of_portfolios):
        monte_weights = np.random.random(len(assets))
        monte_weights /=  np.sum(monte_weights)

        all_weights[i, :] = monte_weights

        portfolio_return = np.sum((log_returns.mean() * monte_weights) * 252)
        portfolio_std_dev = np.sqrt(np.dot(monte_weights.T, np.dot(log_returns.cov() * 252, monte_weights)))

        ret_arr[i] = portfolio_return * 100
        vol_arr[i] = portfolio_std_dev
        sharpe_arr[i] = portfolio_return / portfolio_std_dev

    simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]
    simulations_df = pd.DataFrame(data=simulations_data).T
    simulations_df.columns = ["Returns", "Volatility", "Sharpe Ratio", "Weights"]

    simulations_df = simulations_df.infer_objects()

    return simulations_df


def allocation_strategy(centroids, age, min_age=18, max_age=65):
    underage=0
    overage=0
    if age<18:
        underage = 1
        age = 18
    if age>65:
        overage = 1
        age = 65
    # Define risk categories based on the centroids
    high_risk_centroid = centroids[2]
    medium_risk_centroid = centroids[1]
    low_risk_centroid = centroids[0]

    # Normalize age range
    age_range = max_age - min_age
    normalized_age = (age - min_age) / age_range

    # Calculate allocation weights based on age
    high_risk_weight = 1 - normalized_age
    medium_risk_weight = 0.5 * (1 - np.abs(normalized_age - 0.5))
    low_risk_weight = normalized_age

    # Ensure weights sum up to 1
    total_weights = high_risk_weight + medium_risk_weight + low_risk_weight
    high_risk_weight /= total_weights
    medium_risk_weight /= total_weights
    low_risk_weight /= total_weights

    # Return allocation weights
    return high_risk_weight, medium_risk_weight, low_risk_weight , underage, overage

@app.route('/portfolio', methods=['POST'])
def optimise_port():
    data = request.json
    ticker = str(data['Symbols'])
    ticker = remove_spaces(ticker)
    assets = ticker.split(',')
    df = get_data(assets)
    sim_no = data.get('sim_no', 1000)

    # Monte Carlo Simulation
    num_of_portfolios = sim_no
    sim_df = monteCarlo(df,assets ,num_of_portfolios)
    sim_df['Volatility'] = sim_df['Volatility'].round(2)
    idx = sim_df.groupby('Volatility')['Returns'].idxmax()
    max_df = sim_df.loc[idx].reset_index(drop=True)
    max_df = max_df.sort_values(by='Volatility').reset_index(drop=True)
    max_df['Weights'] = max_df['Weights'].apply(lambda x: {ticker: weight for ticker, weight in zip(assets, x)})
    max_df = max_df.to_dict(orient='records')

    # Selecting the portfolio with the highest Sharpe Ratio
    max_returns = sim_df.loc[sim_df['Returns'].idxmax()]
    optimal_weights = max_returns['Weights']

    weights_df = pd.DataFrame(optimal_weights, columns=['Weights'])
    # Creating DataFrame for weights
    weights_df.index = assets  # Assign tickers as index
    weights_dict = weights_df.to_dict()

    # Expected annual return, volatility, and Sharpe ratio
    mean_returns = df.pct_change(fill_method=None).mean()
    cov_matrix = df.pct_change(fill_method=None).cov() * 252
    annual_return = np.sum(mean_returns * optimal_weights) * 252
    port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (annual_return - 0.02) / port_volatility

    return jsonify({
        "weights_df": weights_dict,
        "annual_return": annual_return * 100,
        "port_volatility": port_volatility,
        "sharpe_ratio": sharpe_ratio,
        "array_of_allocation": max_df
    })





def plot_chart(df, title):
    fig = px.line(df, title=title)
    return fig


def gaussian_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def calculate_probabilities(user_point, centroids):
    distances = np.array([euclidean_distance(user_point, centroid) for centroid in centroids])
    std = np.std(distances)
    probabilities = np.array([gaussian_pdf(dist, 0, std) for dist in distances])
    normalized_probabilities = probabilities / np.sum(probabilities)
    return normalized_probabilities


def remove_spaces(text):
    return re.sub(r'\s+', '', text)


@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    lifestyle_risk = data['lifestyle_risk']
    expected_annual_roi = data['expected_annual_roi']
    age = data['current_age']
    principal_amount = data['principal_amount']
    if data["risk"] == 0:
        return jsonify("error code value 0")

    centroids = np.array([[8.47589795, 0.01583604],
                          [107.31055752, 0.05353579],
                          [15.56913836, 0.01750048]
                         ])

    if lifestyle_risk == 0:
        expected_volatility = centroids[0][1]
    elif lifestyle_risk == 1:
        expected_volatility = centroids[2][1]
    elif lifestyle_risk == 2:
        expected_volatility = centroids[1][1]
    else:
        return jsonify({"error": "Invalid lifestyle risk value"})


    probability_input = np.array([[expected_annual_roi, expected_volatility]])
    probabilities = calculate_probabilities(probability_input, centroids)
    weighted_amounts =  probabilities * principal_amount
    risk_based_weights = pd.DataFrame({'Weight': weighted_amounts})
    high_risk_weight, medium_risk_weight, low_risk_weight, underage, overage = allocation_strategy(centroids, age)

    dataframe = [
        {"weights": low_risk_weight * principal_amount},
        {"weights":high_risk_weight * principal_amount},
        {"weights":medium_risk_weight * principal_amount}
    ]
    age_based_weights = pd.DataFrame(dataframe)
    clusters_data = {
        "Symbols": [
            "GC=F,SI=F",
            "BTC-USD,ETH-USD,SOL-USD, MATIC-USD",
            "HDFCBANK.NS,ICICIBANK.NS,RELIANCE.NS,TATAPOWER.NS,TATAMOTORS.NS,TCS.NS,INFY.NS,HINDUNILVR.NS,LT.NS,APOLLOHOSP.NS"
        ],
        "Underage_Flag": underage,
        "Overage_Flag": overage
    }
    # clusters_data["age_based_weights"]= age_based_weights["weights"]
    # clusters_data['W_risk'] = risk_based_weights['Weight']
    n =math.exp(10-data["risk"])
    clusters_data["Weights"] = (( n * age_based_weights["weights"]) + (1 * np.array(risk_based_weights["Weight"])))/( n + 1)
    clusters_df = pd.DataFrame(clusters_data)
    results = []

    for index, row in clusters_df.iterrows():

        ticker = str(row['Symbols'])
        ticker = remove_spaces(ticker)
        assets = ticker.split(',')
        if index == 0:
            df = get_data(assets,'2005-01-01')
        elif index == 1:
            df = get_data(assets,'2021-06-01')
        elif index == 2:
            df = get_data(assets,'2014-01-01')

        # print(df)
        # print(ticker)
        starting_amount = row['Weights']
        weights_allocated, annual_return, port_volatility, sharpe_ratio, max_df = optimize_portfolio(df,assets)
        # print("1")
        # print(weights_allocated)
        del df
        # print("hi this is max df",max_df)
        # print(weights_df)
        # print(annual_return)
        # print(port_volatility)
        # print("2")
        results.append({
            "Symbols": row['Symbols'],
            "Weights": weights_allocated.to_dict(),
            "Annual Return": annual_return * 100,
            "Volatility": port_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Array_of_allocations": max_df
        })
        # print("3")
    # print(type(results))
    return jsonify({"results": results, "clusters": clusters_df.to_dict(orient='records')})


@app.route('/weights', methods=['POST'])
def weights():
    data = request.json
    lifestyle_risk = data['lifestyle_risk']
    expected_annual_roi = data['expected_annual_roi']
    age = data['current_age']
    principal_amount = data['principal_amount']
    if data["risk"] == 0 or data["risk"]<0 or data["risk"]>10:
        return jsonify("error code value 0")

    centroids = np.array([[8.47589795, 0.01583604],
                          [107.31055752, 0.05353579],
                          [15.56913836, 0.01750048]
                         ])

    if lifestyle_risk == 0:
        expected_volatility = centroids[0][1]
    elif lifestyle_risk == 1:
        expected_volatility = centroids[2][1]
    elif lifestyle_risk == 2:
        expected_volatility = centroids[1][1]
    else:
        return jsonify({"error": "Invalid lifestyle risk value"})


    probability_input = np.array([[expected_annual_roi, expected_volatility]])
    probabilities = calculate_probabilities(probability_input, centroids)
    print(probabilities)
    weighted_amounts =  probabilities * principal_amount
    risk_based_weights = pd.DataFrame({'Weight': weighted_amounts})
    high_risk_weight, medium_risk_weight, low_risk_weight, underage, overage = allocation_strategy(centroids, age)

    dataframe = [
        {"weights": low_risk_weight * principal_amount},
        {"weights":high_risk_weight * principal_amount},
        {"weights":medium_risk_weight * principal_amount}
    ]
    age_based_weights = pd.DataFrame(dataframe)
    clusters_data = {
        "Symbols": [
            "GC=F,SI=F",
            "BTC-USD,ETH-USD,SOL-USD, MATIC-USD",
            "HDFCBANK.NS,ICICIBANK.NS,RELIANCE.NS,TATAPOWER.NS,TATAMOTORS.NS,TCS.NS,INFY.NS,HINDUNILVR.NS,LT.NS,APOLLOHOSP.NS"
        ],
        "Underage_Flag": underage,
        "Overage_Flag": overage
    }
    # clusters_data["age_based_weights"]= age_based_weights["weights"]
    # clusters_data['W_risk'] = risk_based_weights['Weight']
    n =math.exp(10-data["risk"])
    clusters_data["Weights"] = (( n * age_based_weights["weights"]) + (1 * np.array(risk_based_weights["Weight"])))/( n + 1)
    clusters_df = pd.DataFrame(clusters_data)
    return jsonify({"clusters": clusters_df.to_dict(orient='records')})


if __name__ == '__main__':
    app.run(debug=True)