#********************* Import packages
import tkinter as tk
from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras

# *******************Class definition and initialization
class App(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=42, width=52)
        self.mylabel  = Label(self, text = "AESO Price Forecast Suit", font=(None, 15), height=1, width=50).grid(row=0, column=1, columnspan=12, sticky=W+E)
        Label(self, text="Hour Ending", font='Helvetica 9 bold').grid(row=2, column=0, sticky=W+E)
        Label(self, text="Demand Forecast",font='Helvetica 9 bold' ).grid(row=2, column=2, sticky=W)
        Label(self, text="Wind Forecast", font='Helvetica 9 bold' ).grid(row=2, column=4, sticky=W)
        Label(self, text="Price Forecast", font='Helvetica 9 bold', foreground="red").grid(row=2, column=5, sticky=W)

#**************Populate Hour ending, demand forecast and wind forcast
        Hourending = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        for i in ( Hourending):
            Label(self, text="HE{}".format(i)).grid(row=2+i, column=0, sticky=W+E)
        self.predict = pd.read_csv("forecast.csv", encoding='latin1')
        for i in range(0, 24, 1):
            Label(self, text=self.predict.iloc[i, 0], pady=5).grid(row=3+i, column=2,  sticky=W+E)
            Label(self, text=self.predict.iloc[i, 1], pady=5).grid(row=3+i, column=4, sticky=W+E)

#*************Definew and Initialize command Variables
        self.checkDemandPrice = tk.IntVar()
        self.checkWindPrice = tk.IntVar()
        self.checkWindDemandPrice = tk.IntVar()
        self.HistoricaldataVar = tk.IntVar()
        self.RegressionmodelsVar = tk.IntVar()
        self.KNNClassification = tk.IntVar()
        self.KmeansClassification = tk.IntVar()

# *************Define buttons, labels, entry and record buttons in for the GUI
        Button(self, text="Plot Training Data", command=self.DatapointPlot).grid(row=3, column=7, sticky='ew')
        self.HistoricaldataVar = StringVar()
        self.HistoricaldataVar.set(None)
        Radiobutton(self, text='Price vs Demand', variable =self.HistoricaldataVar, value= 1, ).grid(row =4, column=7,  sticky='w')
        Radiobutton(self, text='Price vs Wind', variable=self.HistoricaldataVar, value = 2).grid(row =5, column=7,  sticky='w')
        Radiobutton(self, text='Price vs Wind vs Demand', variable=self.HistoricaldataVar, value=3).grid(row =6, column=7, sticky='w')

        Button(self, text="Regression Models", command=self.LinearRegression).grid(row=8, column=7, sticky='ew')

        self.RegressionmodelsVar = StringVar()
        self.RegressionmodelsVar.set(None)
        Radiobutton(self, text='Linear Regression', variable=self.RegressionmodelsVar, value=1, ).grid(row=9, column=7,  sticky='w')
        Radiobutton(self, text='Polynomial Regression', variable=self.RegressionmodelsVar, value=2).grid(row=10, column=7,  sticky='w')
        Radiobutton(self, text='Radial Basis Function (RBF)', variable=self.RegressionmodelsVar ,value=3).grid(row=11, column=7,  sticky='w')
        Radiobutton(self, text='Logistic Regression', variable=self.RegressionmodelsVar, value=4).grid(row=12, column=7, sticky='w')

        Button(self, text="K Nearest Neighbor Model", command=self.KNearestNeigbour).grid(row=14, column=7, sticky='ew')
        Label(self, text="Enter Number of Classification, default =2", font='Helvetica 8 ').grid(row=15, column=7, sticky=W + E)
        self.entry1 = tk.Entry(self, textvariable = self.KNNClassification)
        self.entry1.grid(row=16, column=7, sticky=E)
        Button(self, text="Support Vector Machine (SVM)", command=self.SVM).grid(row=18, column=7, sticky='ew')

        Button(self, text="K-Means Clustering", command=self.KMeans).grid(row=20, column=7, sticky='ew')
        Label(self, text="Enter Number of Clusters", font='Helvetica 8 ').grid(row=21, column=7, sticky=W + E)
        self.entry2 = tk.Entry(self, textvariable = self.KmeansClassification)
        self.entry2.grid(row=22, column=7, sticky=E)

        Button(self, text="Neural Network", command=self.NeuralNetwork).grid(row=24, column=7, sticky='ew')

    def ReadSourceFile(self):  # Read data from the various CSV files which much be saved in the same folder and the headings much match.
        self.df = pd.read_csv("HistoricalPoolPriceReport2019.csv", encoding='latin1')
        self.predict = pd.read_csv("forecast.csv", encoding='latin1')
        self.DemandWindforecast = np.array(self.predict)
        self.df.fillna(0, inplace=True)
        self.df.drop(['Date (HE)'], 1, inplace=True)
        self.X = np.array(self.df.drop(['Price ($)'], 1))
        self.y = np.array(self.df['Price ($)'])
        self.Hourending = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] # Hour ending counts
        self.price23rd =[37, 37, 37, 37, 37, 37, 41, 37, 76, 87, 178, 89, 55, 39, 106, 97, 92, 244, 82, 76, 61, 106, 68, 51]  # Actual prices for the hour of interest
        return self.X, self.y, self.DemandWindforecast, self.Hourending, self.price23rd


    def DatapointPlot(self):
        self.df = pd.read_csv("HistoricalPoolPriceReport2019.csv", encoding='latin1')
        self.df.fillna(0, inplace=True)
        Demand = np.array(self.df['AIL Demand (MW)'])
        Wind = np.array(self.df['Wind'])
        Price = np.array(self.df['Price ($)'])
        print(self.HistoricaldataVar.get())
        if self.HistoricaldataVar.get() == '1':
            figure = plt.Figure(figsize=(9, 5), dpi=100)
            ax = figure.add_subplot(111)
            ax.set_ylabel('Price($)')
            ax.set_xlabel('Demand(MW)')
            ax.scatter(Demand, Price, color='#003F72', label='Demand vs Price' )
            ax.legend(loc=1)
            ax.set_title('Graph of 2019 AESO Hourly Electricity Demands vs Electricity Prices')
            self.canvas = FigureCanvasTkAgg(figure, self)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=10, columnspan=4, sticky='w')
            self.df.fillna(0)
        elif self.HistoricaldataVar.get() == '2':
            figure = plt.Figure(figsize=(9, 5), dpi=100)
            ax = figure.add_subplot(111)
            ax.set_ylabel('Price($)')
            ax.set_xlabel('Wind Generation(MW)')
            ax.scatter(Wind, Price, color='#003F72', label='Wind vs Price')
            ax.set_title('Graph of 2019 AESO Hourly Total Wind Generation vs Electricity Prices')
            ax.legend(loc=1)
            self.canvas = FigureCanvasTkAgg(figure, self)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=10, columnspan=4, sticky='w')
            self.df.fillna(0)
        elif self.HistoricaldataVar.get() == '3':
            figure = plt.Figure(figsize=(9, 5), dpi=100)
            ax = figure.add_subplot(111, projection='3d')
            ax.plot_trisurf(Demand, Wind, Price)
            ax.set_title('Graph of 2019 AESO Hourly Total Wind Generation vs Electricity Prices vs Electricity Prices')
            ax.set_ylabel('Wind Generation(MW)')
            ax.set_xlabel('Demand(MW)')
            ax.set_zlabel('Price ($)')
            ax.legend(loc=1)
            self.canvas = FigureCanvasTkAgg(figure, self)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=10, columnspan=4, sticky='w')
            self.df.fillna(0)
        else:
            messagebox.showinfo("System message", "Please select an option")

    def LinearRegression(self):
        self.ReadSourceFile()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        if self.RegressionmodelsVar.get() == '1':
            regressor = svm.SVR('linear')
        elif self.RegressionmodelsVar.get() == '2':
            regressor = svm.SVR('poly')
        elif self.RegressionmodelsVar.get() == '3':
            regressor = svm.SVR('rbf')
        elif self.RegressionmodelsVar.get() == '4':
            regressor = svm.SVR('sigmoid')
        else:
            "System message", "Please select an option"
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(self.DemandWindforecast)
        figure = plt.Figure(figsize=(9, 5), dpi=100)
        ax = figure.add_subplot(111)
        for i in range(0, 24, 1):
            Label(self, text=str(round(y_pred[i],2)),  pady=5, foreground="red").grid(row=3+i, column=5,  sticky=W+E)
        ax.plot(self.Hourending, self.price23rd, linewidth=2.0,  color='r', label='Actual')
        ax.plot(self.Hourending, y_pred, linewidth=2.0,  color='k', label='Forecast')
        ax.legend(loc=1)
        ax.set_title('AESO Hourly price forecast - Regression Models')
        ax.set_ylabel('Price($)')
        ax.set_xlabel('HE')
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=10, columnspan=5, sticky='w')

    def KNearestNeigbour(self):
        self.ReadSourceFile()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        if self.KNNClassification.get() == 0:
            messagebox.showinfo("System message", "Please enter a classification number")
            #quit()
        neigh = KNeighborsRegressor(n_neighbors=self.KNNClassification.get())
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(self.DemandWindforecast)
        figure = plt.Figure(figsize=(9, 5), dpi=100)
        ax = figure.add_subplot(111)
        for i in range(0, 24, 1):
            Label(self, text=str(round(y_pred[i],2)),  pady=5, foreground="red").grid(row=3+i, column=5,  sticky=W+E)
        ax.plot(self.Hourending, self.price23rd, linewidth=2.0, marker='o', color='r', label='Actual')
        ax.plot(self.Hourending, y_pred, linewidth=2.0, marker='o', color='k', label='Forecast')
        ax.set_title('AESO  Hourly price forecast -K-Nearest Neighbor Model')
        ax.set_ylabel('Price($)')
        ax.set_xlabel('HE')
        ax.legend(loc=1)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=9, columnspan=4, sticky='w')

    def SVM(self):
        self.ReadSourceFile()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        SVmm = svm.SVR()
        SVmm.fit(X_train, y_train)
        y_pred = SVmm.predict(self.DemandWindforecast)
        figure = plt.Figure(figsize=(9, 5), dpi=100)
        ax = figure.add_subplot(111)
        for i in range(0, 24, 1):
            Label(self, text=str(round(y_pred[i],2)),  pady=5, foreground="red").grid(row=3+i, column=5,  sticky=W+E)
        ax.plot(self.Hourending, self.price23rd, linewidth=2.0, marker='o', color='r', label='Actual')
        ax.plot(self.Hourending, y_pred, linewidth=2.0, marker='o', color='k', label='Forecast')
        ax.set_title('AESO  Hourly price forecast - Support Vector Machine (SVM)')
        ax.set_ylabel('Price($)')
        ax.set_xlabel('HE')
        ax.legend(loc=1)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=9, columnspan=4, sticky='w')

    def KMeans(self):
        self.ReadSourceFile()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        if self.KmeansClassification.get() == 0:
            messagebox.showinfo("System message", "Please enter a classification number")
        kmeans = KMeans(n_clusters=self.KmeansClassification.get())
        kmeans.fit(X_train, y_train)
        y_pred = kmeans.predict(self.DemandWindforecast)
        figure = plt.Figure(figsize=(9, 5), dpi=100)
        ax = figure.add_subplot(111)
        for i in range(0, 24, 1):
            Label(self, text=str(round(y_pred[i], 2)), pady=5, foreground="red").grid(row=3 + i, column=5, sticky=W + E)
        ax.plot(self.Hourending, self.price23rd, linewidth=2.0, marker='o', color='r', label='Actual')
        ax.plot(self.Hourending, y_pred, linewidth=2.0, marker='o', color='k', label='Forecast')
        ax.set_title('AESO  Hourly price forecast - KMeans')
        ax.set_ylabel('Price($)')
        ax.set_xlabel('HE')
        ax.legend(loc=1)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, rowspan=24, column=9, columnspan=4, sticky='w')




    def NeuralNetwork(self):
        #self.ReadSourceFile()
        self.df = pd.read_csv("HistoricalPoolPriceReport2019.csv", encoding='latin1')
        self.predict = pd.read_csv("forecast.csv", encoding='latin1')
        self.DemandWindforecast = np.array(self.predict)
        self.df.fillna(0, inplace=True)
        self.df.drop(['Date (HE)'], 1, inplace=True)
        self.XX = np.array(self.df)
        #print(XX)
        #mnist = keras.datasets.XX
        #print(mnist)
        #self.X = np.array(self.df.drop(['Price ($)'], 1))
        #self.y = np.array(self.df['Price ($)'])
        #X_train, X_test, y_train, y_test = train_test_split(self.X,
        #                                                    self.y,
        #                                                    test_size=0.33,
         #                                                   random_state=42)


        #print(X_train, X_test)
        #(x_train, y_train) =
        #messagebox.showinfo("System message", " This module is under development")
        self.nNodesL1 = 1000
        self.nNodesL2 = 1000
        self.nNodesL3 = 1000
        self.numclasses = 10
        self.batch_size = 100

        self.x = tf.placeholder('float', [None, 8760])
        self.y = tf.placeholder('float')
        self. train_neural_network(self.x)

    def NeuralNetworkModel(self, data):

        hiddenlayer1 = {'weights': tf.Variable(tf.random_normal([8760, self.nNodesL1])),
                          'biases': tf.Variable(tf.random_normal([self.nNodesL1]))}

        hiddenlayer2 = {'weights': tf.Variable(tf.random_normal([self.nNodesL1, self.nNodesL2])),
                          'biases': tf.Variable(tf.random_normal([self.nNodesL2]))}

        hiddenlayer3 = {'weights': tf.Variable(tf.random_normal([self.nNodesL2, self.nNodesL3])),
                          'biases': tf.Variable(tf.random_normal([self.nNodesL3]))}

        outputLayer = {'weights': tf.Variable(tf.random_normal([self.nNodesL3, self.numclasses])),
                        'biases': tf.Variable(tf.random_normal([self.numclasses])), }

        level1 = tf.add(tf.matmul(data, hiddenlayer1['weights']), hiddenlayer1['biases'])
        level1 = tf.nn.relu(level1)

        level2 = tf.add(tf.matmul(level1, hiddenlayer2['weights']), hiddenlayer2['biases'])
        level2 = tf.nn.relu(level2)

        level3 = tf.add(tf.matmul(level2, hiddenlayer3['weights']), hiddenlayer3['biases'])
        level3 = tf.nn.relu(level3)

        output = tf.matmul(level3, outputLayer['weights']) + outputLayer['biases']

        return output

    def train_neural_network(self, x):
        PricePrediction = self.NeuralNetworkModel(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=PricePrediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        hm_epochs = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(self.XX.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y =self.XX.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)


        print("Under development")

    #correct = tf.equal(tf.argmax(PricePrediction, 1), tf.argmax(y, 1))

    #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    #print('Accuracy:', accuracy.eval({x: self.XX.test.images, y: self.XX.test.labels}))


#train_neural_network(x)


def main():
    root = tk.Tk()
    App(root).pack(expand=True, fill='both')
    root.mainloop()

if __name__ == "__main__":
    main()




















