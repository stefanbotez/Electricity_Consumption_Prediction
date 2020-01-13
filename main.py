import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Grouper
import numpy as np



TRAIN_BARRIER = 392150 # ultimele 6 luni pt validare
traind = pd.read_csv("train_electricity.csv")
testd = pd.read_csv("test_electricity.csv")
n_train, n_test = len(traind), len(testd)


def new_data(dat):
    newdate = pd.to_datetime(dat.Date * (10 ** 9))
    dat['NewDate'] = newdate
    data = ["year", "month", "week", "day", "hour", "minute", "dayofyear"]

    for d in data:
        dat[d] = getattr(newdate.dt, d)
    print(dat)
    return dat


traind = new_data(traind)
testd = new_data(testd)

traind = traind.sort_values(by=["NewDate"], ascending=True)
testd = testd.sort_values(by=["NewDate"], ascending=True)


def remove_aberrant(dat):
    dat = dat.loc[np.invert(dat['Coal_MW'] > 4500) | np.invert(dat['Coal_MW'] < 300), :]
    dat = dat.loc[np.invert(dat['Gas_MW'] < 30), :]
    dat = dat.loc[np.invert(dat['Nuclear_MW'] < 400), :]
    dat = dat.loc[np.invert(dat['Biomass_MW'] > 80), :]
    dat = dat.loc[np.invert(dat['Production_MW'] < 3000), :]
    dat = dat.loc[np.invert(dat['Consumption_MW'] > 11000) | np.invert(dat['Consumption_MW'] < 2000), :]
    return dat


traind = remove_aberrant(traind)


def plot_cons():
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    axs.plot(traind['NewDate'], traind['Consumption_MW'], color='royalblue')
    axs.set_xlabel('Date')
    axs.set_ylabel('Consumption_MW')
    plt.show()


def plot_prod():
    productions = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW',
                   'Production_MW']
    i = 0
    fig, axs = plt.subplots(len(productions), 1, figsize=(15, 5 * len(productions)))
    for production in productions:
        axs[i].plot(traind['NewDate'], traind[production], color='royalblue')
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel(production)
        i = i + 1

    plt.show()


def plot_prod_cons():
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.plot(traind['Production_MW'], traind['Consumption_MW'], '.', color='royalblue')
    axs.set_xlabel('Production_MW')
    axs.set_ylabel('Consumption_MW')
    axs.set_xlim(3000, 12000)
    axs.set_ylim(3000, 12000)
    plt.show()


def plot_correlations():
    sns.pairplot(traind[['Consumption_MW', 'Coal_MW', 'Gas_MW', 'Hidroelectric_MW',
                           'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Production_MW']])
    plt.show()


def plot_stacked():
    cols = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW']
    leg = []
    last_means = np.zeros(97)
    x = []
    xyrs = []

    for c in cols:
        groups = traind[[c, "NewDate"]].groupby(Grouper(key='NewDate', freq="M"))

        m = []
        for i, (name, group) in enumerate(groups):
            m.append(np.mean(group[[c]].values.flatten()))
            if c == "Coal_MW" and name.month == 1:
                    xyrs.append(name.year)
                    x.append(i)

        plt.bar(range(97), m, 0.5, bottom=last_means)
        last_means += m

        leg.append(c)

    plt.ylabel('Production_MW')
    plt.title('Production')
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(xyrs)
    plt.gcf().set_size_inches(25, 10)
    plt.legend(leg)
    plt.show()


def prod_cons_stats():
    diff = np.mean(np.abs(traind['Consumption_MW'] - traind['Production_MW']))
    mse = np.mean((traind['Consumption_MW'] - traind['Production_MW']) ** 2)
    rmse = np.sqrt(mse)
    print('diff =', diff)
    print('MSE = ', mse)
    print('RMSE = ', rmse)

def plot_prod_cons2():
    f, ax = plt.subplots(1, 1)
    traind.iloc[50000:52000].plot(x='Date', y='Production_MW', figsize=(20, 5), ax=ax, color='r', alpha=0.7)
    traind.iloc[50000:52000].plot(x='Date', y='Consumption_MW', figsize=(20, 5), ax=ax, color='g', alpha=0.7)
    plt.show()

def plot_yearly():
    cons = traind.copy()
    cons['Consumption_MW'] = cons['Consumption_MW'].rolling(window=20000, center=True).mean()
    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    color = iter(plt.cm.Oranges(np.linspace(0, 1, 9)))

    for i in range(2010, 2018):
        data = cons[cons['year'] == i]
        c = next(color)
        ax.plot(data.dayofyear, data.Consumption_MW, c=c)

    ax.legend(('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'))
    plt.show()

# plot_cons()
# plot_prod()
# plot_prod_cons()
# #plot_correlations()
# plot_stacked()
# prod_cons_stats()
#plot_prod_cons2()
plot_yearly()

# train_data = traind.drop(columns=['Consumption_MW', 'Date', 'NewDate'])[:TRAIN_BARRIER]
# train_labels = traind['Consumption_MW'][:TRAIN_BARRIER]
#
# valid_data = traind.drop(columns=['Consumption_MW', 'Date', 'NewDate'])[TRAIN_BARRIER:]
# valid_labels = traind['Consumption_MW'][TRAIN_BARRIER:]
#
# sc = MinMaxScaler(feature_range=(0, 1))
#
# train_data = sc.fit_transform(train_data)
# train_labels = train_labels.values[:, None].astype(np.float)
#
# valid_data = sc.fit_transform(valid_data)
# valid_labels = valid_labels.values[:, None].astype(np.float)
#
# nr_in = train_data.shape[1]
# BACK = 100
# ITERS = 500000
# PRINT = 500
# BATCH_SIZE = 50
#
# class Network(Model):
#     def __init__(self):
#         super().__init__()
#         self.training = tf.placeholder(tf.bool)
#         self.model = tf.keras.Sequential([
#             Dense(512, input_shape=(nr_in * BACK,)),
#             BatchNormalization(),
#             ReLU(),
#             Dense(128),
#             BatchNormalization(),
#             ReLU(),
#             Dense(64),
#             BatchNormalization(),
#             ReLU(),
#             Dense(10),
#             ReLU(),
#             Dense(1)
#         ])
#         # self.model = tf.keras.Sequential([
#         #     Dense(20, input_shape=(nr_in * BACK,)),
#         #     ReLU(),
#         #     Dropout(0.1),
#         #     Dense(10),
#         #     Dropout(0.1),
#         #     ReLU(),
#         #     Dense(1)
#         # ])
#
#     def call(self, x):
#         x = tf.reshape(x, [-1, nr_in * BACK])
#         y = self.model(x, training=self.training)
#         return y
#
#
# sess = tf.InteractiveSession()
# model = Network()
#
# input_ = tf.placeholder(tf.float32, [None, nr_in * BACK])
# label_ = tf.placeholder(tf.float32, [None, 1])
#
# output = model(input_)
# loss = tf.reduce_mean(tf.square(output - label_))
#
# optim = tf.train.AdamOptimizer(learning_rate=0.0005)
# optim_step = optim.minimize(loss)
#
# sess.run(tf.global_variables_initializer())
#
# train_losses = []
#
# minim_valid_loss = 1e9
#
# x = []
# y = []
# for i in range(BACK, valid_data.shape[0]):
#     x.append(valid_data[i - BACK:i])
#     y.append(valid_labels[i])
# x = np.reshape(x, (-1, nr_in * BACK))
#
#
# def validate():
#     los = sess.run(loss, feed_dict={input_: x,
#                                     label_: y,
#                                     model.training: False})
#     return los
#
#
# train_loss = []
# valid_loss = []
# for i in range(1, ITERS + 1):
#     idxs = np.random.randint(len(train_data) - BACK + 1, size=(BATCH_SIZE,))
#
#     data = [train_data[i:i + BACK] for i in idxs]
#     data = np.reshape(data, (BATCH_SIZE, nr_in * BACK))
#
#     label = train_labels[idxs + BACK - 1]
#
#     aux, l = sess.run([optim_step, loss], feed_dict={input_: data,
#                                                     label_: label,
#                                                     model.training: True})
#
#     train_losses.append(l)
#     if i % PRINT == 0:
#         v = validate()
#         t = np.mean(train_losses)
#         print(f"ITER {i:4d}: {t:7.2f} : {v:7.2f}")
#         train_loss.append(t)
#         valid_loss.append(v)
#         train_losses.clear()
#         if v < minim_valid_loss:
#             minim_valid_loss = v
#             model.save_weights("model.h5")
#
# plt.plot(np.sqrt(train_loss))
# plt.plot(np.sqrt(valid_loss))
# plt.show()
#
#
# def test():
#     global output
#     dat = pd.read_csv("test_electricity.csv")
#     print(len(dat))
#     dat = new_data(dat)
#     test_data = dat.drop(columns=["Date", "NewDate"]).values.astype(np.float)
#     print(test_data.shape)
#
#     res = sess.run([output], feed_dict={input_: test_data,
#                                         model.training: False})
#     print(res)
#
#
# test()
