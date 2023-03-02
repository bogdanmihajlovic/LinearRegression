import math
import pandas as pd
import numpy as np
from mymodel import LinearRegressionGradientDescent
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.metrics import mean_squared_error

def show_diagram_cont(data, names):
    for name in names:
        plt.scatter(data[name], data['CO2EMISSIONS'])
        plt.title("title")
        plt.xlabel("x-" + name)
        plt.ylabel("y-CO2EMISSIONS'")
        plt.show()


def show_diagram_cat(data, names):

    for name in names:
        fig, ax = plt.subplots()
        ax.bar(data[name], data['CO2EMISSIONS'])
        plt.xlabel("x-" + name)
        plt.ylabel("y-CO2EMISSIONS'")
        plt.xticks(rotation=90)
        plt.show()


def main():
    # 1. ucitavanje podataka i prikaz prvih pet redova
    data = pd.read_csv('cars.csv')
    pd.set_option('display.width', None)
    print(data.head(), end="\n\n")

    # 2. prikaz informacija
    print(data.info(), end="\n\n")
    print(data.describe(), end="\n\n")
    print(data.describe(include=object), end="\n\n")

    # 3. eliminacija veladignih
    data = data.dropna()

    # 4. korelaciona matrica

    plt.figure()
    sb.heatmap(data.drop(columns=["MODELYEAR", "MODEL", 'TRANSMISSION', 'FUELTYPE', 'VEHICLECLASS', "MAKE"]).corr(), annot=True, fmt=".2f")
    plt.show()

    # 5. grafici
    #show_diagram_cont(data, ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',  'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',  'FUELCONSUMPTION_COMB_MPG' ])
    #show_diagram_cat(data, ['MAKE', 'VEHICLECLASS', 'TRANSMISSION'])

    # 6. odabir atributa
    data_train = data.drop(columns=['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS'])
    labels = data['CO2EMISSIONS']

    # 7. fomiranje skupova za trening i tesitranje
    input_train, input_test, output_train, output_test = train_test_split(data_train, labels, train_size=0.8,
                                                                          random_state=64, shuffle=False)

    # 8. realizacija treniranja i prikaz rezultata
    lr_model = LinearRegression()
    lr_model.fit(input_train, output_train)
    lr_pred = lr_model.predict(input_test)
    lr_ser_pred = pd.Series(data=lr_pred, name="LR Predicted", index=input_test.index)
    lr_score = lr_model.score(input_test, output_test)

    # treniranje mog modela
    lrgd_model = LinearRegressionGradientDescent()
    lrgd_model.fit(input_train, output_train)
    learning_rates = np.array(0.001)
    res_coeff, mse_history = lrgd_model.perform_gradient_descent(learning_rates, 1000000)
    lrgd_pred = lrgd_model.predict(input_test)
    lrgd_ser_pred = pd.Series(data=lrgd_pred, name="LRGD Predicted", index=input_test.index)


    final_res = pd.concat([input_test, output_test, lrgd_ser_pred, lr_ser_pred], axis=1)
    print(final_res.head())

    # Stampanje mse za oba modela
    lrgd_model.set_coefficients(res_coeff)
    print(f'LRGD MSE: {lrgd_model.cost():.2f}')
    c = np.concatenate((np.array([lr_model.intercept_]), lr_model.coef_))
    lrgd_model.set_coefficients(c)
    print(f'LR MSE: {lrgd_model.cost():.2f}')
    # Restauracija koeficijenata
    lrgd_model.set_coefficients(res_coeff)

    lr_coef_ = lr_model.coef_
    lr_int_ = lr_model.intercept_
    lr_model.coef_ = lrgd_model.coeff.flatten()[1:]
    lr_model.intercept_ = lrgd_model.coeff.flatten()[0]
    print(f'LRGD score: {lr_model.score(input_test, output_test):.2f}')

    lr_model.coef_ = lr_coef_
    lr_model.intercept_ = lr_int_
    print(f'LR score: {lr_model.score(input_test, output_test):.2f}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


