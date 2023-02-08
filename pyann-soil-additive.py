import pandas as pd
import numpy as np
import pyrenn as prn
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class Soil():
    """ Class to store soil with all data """
    soil_type: str
    path: str

    def load_data(self) -> pd.DataFrame:
        df = pd.ExcelFile(self.path).parse('data')
        return self.filter_by_soil_type(df)

    def filter_by_soil_type(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.soil_type == 'bentonite':
            df.drop(df.loc[df['I1']==0].index, inplace=True)
        if self.soil_type == 'kaolin':
            df.drop(df.loc[df['I1']==1].index, inplace=True)
        return df


@dataclass
class Ann():
    """ Class to store ANN data with related methods """
    n_inputs: int
    n_hidden: int
    n_outputs: int
    structure: dict = field(init=False)
    trained_nn: dict = field(init=False)
    normalize: bool = False
    train_size: float = 0.2
    k_max: int = 1000
    E_stop: float = 1e-3

    def __post_init__(self):
        object.__setattr__(self,'structure',self.create_ann())

    def prepare_data(self, data: pd.DataFrame) -> dict[str,np.ndarray]:
        X = data.drop(columns=['I1','O'])
        y = data['O']
        X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X, y, test_size=self.train_size)

        X_train = np.transpose(X_train_.to_numpy())
        Y_train = np.transpose(Y_train_.to_numpy())
        X_test = np.transpose(X_test_.to_numpy())
        Y_test = np.transpose(Y_test_.to_numpy())

        return {'training': {'X': X_train, 'Y': Y_train}, 'testing': {'X': X_test, 'Y': Y_test}}

    def create_ann(self) -> dict:
        return prn.CreateNN([self.n_inputs,self.n_hidden,self.n_outputs], dIn=[0], dIntern=[], dOut=[1])

    def train_ann(self, X: np.ndarray, Y: np.ndarray) -> dict:
        return prn.train_LM(X, Y, self.structure, verbose=True, k_max=self.k_max, E_stop=self.E_stop)

    def predict(self, X_predict: np.ndarray) -> np.ndarray:
        return prn.NNOut(X_predict, self.trained_nn)

def error_estimators(y_train: np.ndarray, y_predict: np.ndarray) -> dict[str, float]:
    # function for various error esimators
    errors = {
        'MSE': metrics.mean_squared_error(y_train,y_predict),
        'MAE': metrics.mean_absolute_error(y_train,y_predict),
        'MAPE': metrics.mean_absolute_percentage_error(y_train,y_predict),
        'MedAE': metrics.median_absolute_error(y_train,y_predict),
        'max': metrics.max_error(y_train,y_predict)
    }

    return errors

def perf_plot(y_train: np.ndarray, y_predict: np.ndarray, save=False) -> None:
    # function for plotting performance of network
    plt.figure(0)
    plt.plot(y_train,y_predict,color='r',marker='o',linestyle='',lw=3,markersize=8,label='Train Data')
    plt.xlabel('Observed $\u03C3_{c}$ ($kN/m^{3}$)')
    plt.ylabel('Predicted $\u03C3_{c}$ ($kN/m^{3}$)')

    if save:
        plt.savefig('performance.png')
    else:
        plt.show()

    return None

def dc_data(soil_type: str) -> list[dict]:
    # function for preparing data for design chart
    type: int = 0
    data: list = []
    chi_cem: list = []
    I1: list = []
    I2: list = []
    I3: list = []
    I2_step: float = 0.05

    if soil_type == 'bentonite':
        type = 0
        chi_cem = [10, 30, 50, 70, 90, 110]
    if soil_type == 'kaolin':
        type = 1
        chi_cem = [17, 26, 34]

    for chi in chi_cem:
        I1 = [type for i in range(21)]
        I2 = [(i*I2_step*1.5)-0.75 for i in range(21)]
        I3 = [chi for i in range(21)]
        data.append({'I1': I1, 'I2': I2, 'I3': I3, 'chi': chi})

    return data

def dc_plot(X: list, Y: list, save=False) -> None:
    # function for preparing plot of design chart
    colors: list = ['r', 'g', 'b', 'y', 'k', 'm']

    plt.figure(1)
    for idx, x in enumerate(X):
        plt.plot(x['I2'], Y[idx], color=colors[idx],marker='o',linestyle='-',lw=3,markersize=8,label=f"$\chi_cem$ = {x['chi']}%")

    plt.xlabel('$I_c^{**}$ $[\%]$')
    plt.ylabel('$\sigma_c$ $[kN/m^3]$')
    plt.legend(loc='upper left')
    plt.ylim((0,450))
    
    if save:
        plt.savefig('design_chart.png')
    else:
        plt.show()

    return None


def main() -> None:
    # load soil data
    soil_type='bentonite'
    soil = Soil(soil_type=soil_type, path='data_erminio.xlsx')
    data = soil.load_data()

    # create neural network
    model = Ann(n_inputs=2, n_hidden=1, n_outputs=1)

    # normalize data
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_feat = scaler.transform(data)
    norm_data = pd.DataFrame(norm_feat, index=data.index, columns=data.columns)

    # prepare data for pyrenn neural network
    data_nn = model.prepare_data(norm_data)

    # train neural network, save trained nn and plot performance
    model.trained_nn = model.train_ann(**data_nn['training'])
    y_predict = model.predict(data_nn['testing']['X'])
    perf_plot(data_nn['testing']['Y'], y_predict, True)
    print(error_estimators(data_nn['testing']['Y'], y_predict))

    # create design charts
    y_dc = []
    X_dc = dc_data(soil_type=soil_type)
    for x in X_dc:
        data_= np.array([x['I1'],x['I2'],x['I3']])
        data__ = np.r_[data_, [np.zeros(len(data_[0]))]]
        
        norm_data = scaler.transform(np.transpose(data__))
        norm_data_tr = np.transpose(norm_data)

        prediction = model.predict(norm_data_tr[1:3])
        norm_data_tr[3] = prediction

        y_dc.append(np.transpose(scaler.inverse_transform(np.transpose(norm_data_tr)))[-1])

    dc_plot(X_dc, y_dc, True)


if __name__ == "__main__":
    main()
