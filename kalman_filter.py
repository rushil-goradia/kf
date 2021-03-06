import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
#from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter

STOCKS = ['BAJFINANCE', 'KOTAKBANK', 'MINDTREE', 'NIFTY_50', 'NIFTY_BANK', 'RELIANCE', 'ULTRACEMCO']
RELATIVE_PATH = "/Users/rushil.goradia/Documents/brokerage_docs/kf/samplestockindextimeseries/"
NUM_STATES = 3

def plot_closing_price(stock_name, stock_data):
    plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.plot(stock_data['Close'][NUM_STATES:])
    #print(stock_data['predict'])
    plt.plot(stock_data['predict'])
    #plt.plot(stock_data['correct'])
    plt.xlabel("Days")
    plt.ylabel("Closing price (Rs.)")
    plt.legend(["actual", "prediction", "corrected_prediction"])
    plt.title(stock_name)
    plt.subplot(3,1,2, sharex=ax1)
    plt.plot(stock_data['residual'])
    plt.ylim([-500,500])
    #plt.plot(stock_data['predict'] - stock_data['Close'][NUM_STATES:])
    plt.legend(["kf_residual", "kf_prediction_error"])
    plt.title("Error in kf prediction")
    plt.xlabel("Days")
    plt.ylabel("Price error (Rs)")
    plt.subplot(3,1,3, sharex=ax1)
    plt.plot(stock_data['kalman_gain'])
    plt.ylabel("Kalman gain")
    plt.xlabel("Days")
    #print(stock_data['residual'])

    


def get_stock_data(stock_name):
    stock_data = {}
    full_path = RELATIVE_PATH + stock_name + ".csv"
    with open(full_path) as csvfile:
        csv_rows = csv.DictReader(csvfile)
        for ind, row in enumerate(csv_rows):
            #print(ind)
            #print(row)
            for key in row.keys():
                if key == '':
                    continue
                if key not in stock_data.keys():
                    stock_data[key] = []
                    #print(row[key])
                value = row[key]
                if key == 'Close':
                    value = float(value)
                stock_data[key] += [value]
                # if ind == 0:
                #     print(stock_data)
    return stock_data

def kf_predict(X, P, A, Q, B, U):
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return (X, P)

def kf_update(X, P, Y, H, R):
    IM = np.dot(H, X)
    #print(np.shape(IM)[1])
    IS = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(P, np.dot(H.T, np.linalg.inv(IS)))
    X = X + np.dot(K, (Y - IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X, P, K, IM, IS, LH)

def gauss_pdf(X, M, S):
    if np.shape(M)[1] == 1:
        #print(np.shape(X))
        DX = X - np.tile(M, np.shape(X)[1])
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * np.shape(M)[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    elif X.shape()[1] == 1:
        DX = np.tile(X, M.shape()[1])- M
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0],E[0]) 



def run_kalman_filter(stock_name, stock_data):
    # https://www.haikulabs.com/pmdwkf26.htm
    # x[k|k-1]: The filter's best estimate of closing stock price at time k, given all data except y[k]
    # X[k] = [x[k] ; x[k-1] ; x[k-2]]
    # X[k-1] = [x[k-1] ; x[k-2] ; x[k-3]]
    # x[k] = 3*x[k-1] - 3*x[k-2] + x[k-3]
    # X[k|k-1] = A*X[k-1|k-1] + B*u
    A = np.array([[3.0, -3.0, 1.0],
                        [1., 0., 0.],
                        [0., 1., 0.]], dtype=float)
    # measurement update matrix
    # y[k]: Actual closing price at time k from data
    # y[k] = H*X[k]
    H = np.array([[1., 0., 0.]], dtype=float)
    # Covariance of measurement noise (actual stock price)
    R = 100
    # Initial state (first 3 days price)
    X = np.array([all_stocks_data[stock_name]['Close'][0:NUM_STATES]], dtype=float).T
    # Constantly updated covariance of state
    P = np.array([[1000.,    0., 0.],
                [0., 1000., 0.],
                [0., 0., 1000.]], dtype=float)
    # Process noise covariance (how much do you trust the model)
    Q = np.zeros((3,3), dtype=float)
    Q[0,0] = 500.
    Q[1,1] = 500.
    Q[2,2] = 500.
    B = np.zeros((NUM_STATES, 1))
    U = 0

    num_measurements = len(all_stocks_data[stock_name]['Close'][NUM_STATES:])
    prediction_result = np.zeros((num_measurements,))
    correction_result = np.zeros((num_measurements,))
    kalman_gain = np.zeros((num_measurements,))
    residual = np.zeros((num_measurements,))
    measurements = all_stocks_data[stock_name]['Close'][NUM_STATES:]
    for ind, meas in enumerate(np.array(measurements)):
        #predict, gives us X[k|k-1]
        (X, P) = kf_predict(X, P, A, Q, B, U)
        #print(X[0])
        
        prediction_result[ind] = X[0][0]
        #print(kf.K)
        #print(prediction_result[ind])
        
        #correct
        Y = np.array([[meas]])
        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
        # X[k|k]
        correction_result[ind] = X[0][0]

        #debug
        kalman_gain[ind] = np.linalg.norm(K)
        #print(IM[0][0])
        residual[ind] = IM[0][0] - meas
    #print(prediction_result)

    return {
        'predict':prediction_result, 
        'correct': correction_result,
        'kalman_gain':kalman_gain,
        'residual': residual,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot_all_stocks', action="store_true", default=False)
    parser.add_argument('--run_filter', action="store_true", default=False)
    args = parser.parse_args()
    
    all_stocks_data = {}
    for stock_name in STOCKS:
        all_stocks_data[stock_name] = get_stock_data(stock_name)

    if args.run_filter:
        for stock_name in STOCKS:
            kf_results = run_kalman_filter(stock_name, all_stocks_data[stock_name])

            all_stocks_data[stock_name].update(kf_results)


    if args.plot_all_stocks:
        for stock_name in STOCKS:
            plot_closing_price(stock_name, all_stocks_data[stock_name])
        plt.show()
            #print(all_stocks_data[stock_name]['Close'])


