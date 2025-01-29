import numpy as np
from scipy.stats import vonmises, circmean, circstd
from matplotlib import pyplot as plt
import pandas as pd

# create some random data
kappa = 1
degrees_loc = 180
r = vonmises.rvs(kappa , np.deg2rad(degrees_loc), size=10000)

plt.hist(np.rad2deg(r), 50, density=True)
plt.show()
plt.close(0)

# Some descriptive
cmean = circmean(r)
np.rad2deg(cmean)

cstd = circstd(r)
np.rad2deg(cstd)

# Plotting histogram
start = 0
step  = 0.5
num   = 15
x     = np.arange(0,num)*step+start


#x =  np.linspace(0, 10, bin_size, endpoint=True) # np.array([1,2,3,4,5,6])
hist, _ = np.histogram(r, bins = x)
centers = np.ediff1d(x) // 2 + x[:-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.bar(centers, hist, width=step,bottom=0.0, edgecolor='k')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

plt.show()
plt.close()



def vonmises_kde(data, kappa, n_bins=100):
    from scipy.special import i0
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde

from numpy.random import vonmises

kappa_angle = 1.5
# generate complex circular distribution
data_A = np.r_[vonmises(np.deg2rad(90), kappa_angle, 10000), vonmises(np.deg2rad(270), kappa_angle, 10000)] # , vonmises(3, 20, 100)
data_B = np.r_[vonmises(np.deg2rad(0), kappa_angle, 10000), vonmises(np.deg2rad(180), kappa_angle, 10000)] # , vonmises(3, 20, 100)

# plot data histogram
fig, axes = plt.subplots(2, 1)
axes[0].hist(np.rad2deg(data_A), 100, color = "blue" ,alpha = 0.3)
axes[0].hist(np.rad2deg(data_B), 100, color = "orange",alpha = 0.3)
# plot kernel density estimates
xA, kdeA = vonmises_kde(data_A, kappa_angle)
xB, kdeB = vonmises_kde(data_B, kappa_angle)
axes[1].plot(np.rad2deg(xA), kdeA,  color = "blue" ,alpha = 0.5)
axes[1].plot(np.rad2deg(xB), kdeB,  color = "orange",alpha = 0.5)

b = 179
a = 0
orients = (b - a) * np.random.random_sample((1000, 1)) + a
orients

plt.hist(orients)

plt.imshow(orients)


np.random.uniform(size=10)

# generate complex circular distribution
data = np.r_[vonmises(-1, 5, 1000), vonmises(2, 10, 500), vonmises(3, 20, 100)]

# plot data histogram
fig, axes = plt.subplots(2, 1)
axes[0].hist(data, 100)

# plot kernel density estimates
x, kde = vonmises_kde(data, 20)
axes[1].plot(x, kde)


# With this function I can calculate addition or substractions between angles in a circular space
def angle_math(vector_a, vector_b):
    v = vector_a + vector_b
    v = (v + np.pi) % (2*np.pi) - np.pi # make it relative to 0 degs
   # print('in degrees ',  np.array2string(np.rad2deg(v)))
    return v


# Simulating data
ntrials = 200
n_gratings = 5
noise_dev = 15 # deviation of error in degrees
kappa = 10

# Create the stimuli
degrees_loc = 0
thethas = vonmises.rvs(kappa, np.deg2rad(degrees_loc), size=[n_gratings, ntrials])

plt.imshow(thethas, aspect='auto')
plt.suptitle('Trials simulated', fontsize=20)
plt.xlabel('Trials', fontsize=16)
plt.ylabel('Gratings', fontsize=16)
#plt.hist(np.rad2deg(np.matrix.flatten(thethas)), 50, density=True)
#plt.show()

# Create the gaussian noise matrix
noise = np.random.normal(np.zeros([n_gratings, ntrials]), np.deg2rad(noise_dev), [n_gratings, ntrials]) # create matrix of gaussian noise with mean 0 in all and same standard deviation
#g_noise = np.random.normal(0, np.deg2rad(), [n_gratings, ntrials])

mean_tilt = circmean(thethas, axis = 0) # calculate each trial mean orientation

#plt.hist(np.rad2deg(mean_tilt), 50, density=True)
#plt.show()

thethas_noise = thethas + noise
mean_tilt_noise = circmean(thethas_noise, axis = 0) # calculate each trial mean orientation


mean_tilt = (mean_tilt + np.pi) % (2*np.pi) - np.pi # make it relative to 0
rel_tilt = np.deg2rad(np.random.randint(-10,10,ntrials)) # relative decision boundary

# Calculate absolute angle for relative decision boundary combining mean tilts and relative differences
reference = angle_math(mean_tilt, rel_tilt)

# simulate the difference perceived using the simplest model (that is, people has integrated all the gratings information equally)
diff_tilt  = angle_math(mean_tilt_noise, -reference)
#diff_tilt  = angle_math(mean_tilt, -reference) # judgmements relative to reference


decision  = np.array(diff_tilt > 0).astype(int)
correct   = np.array(np.rad2deg(rel_tilt) < 0).astype(int) # here we look at whether the reference is smaller than the actual stimulus

data_matrix = np.vstack((np.rad2deg(rel_tilt) , decision, correct))
data_matrix = np.transpose(data_matrix)

data_matrix_s = np.vstack((thethas_noise, decision, correct))
data_matrix_s = np.transpose(data_matrix_s)

df_s = pd.DataFrame(data_matrix_s,columns = ["G1","G2","G3","G4","G5", "decision", "correct"])
df = pd.DataFrame(data_matrix,columns = ["rel_tilt", "decision", "correct"])
# convert column "a" to int64 dtype and "b" to complex type
df = df.astype({"rel_tilt": float, "decision": int, "correct": int})
df.dtypes

# ag_df = df.groupby(['rel_tilt'])['decision']
# A = ag_df.mean()
# A = A.keys
# A.shape
df.groupby('rel_tilt')['decision'].scatter(legend=True)


plt.scatter(ag_df.groups.keys(), )

A.keys()

ag_df.keys

bp = ag_df.plot(kind='kde')

# plot data
fig, ax = plt.subplots()
# use unstack()
ag_df.mean().plot(ax=ax)

ag_df.mean().plot(kind='kde', ax=ax)


return ag_df
A = ag_df.mean()
A.keys
ag_df["rel_tilt"]

A ag_df.keys
print(ag_df)

df["rel_tilt"]

ag_df["rel_tilt"]




import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

df['rel_tilt'] = preprocessing.scale(df['rel_tilt'])

df_s['G1', 'G2', 'G3', 'G4', 'G5'] = preprocessing.scale(df_s['G1', 'G2', 'G3', 'G4', 'G5'])

formula = 'decision ~ G1 + G2 + G3 + G4 + G5' # fitting model


model = smf.glm(formula = formula, data=df_s, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

predictions = result.predict()

print(predictions[0:200])

formula = 'decision ~ rel_tilt' # fitting model

model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

predictions = result.predict()

print(predictions[0:200])

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(model.exog, model.endog)
plt.plot(ag_df["rel_tilt"], yfit);



model.exog
model.exog.shape
model.endog.shape
import seaborn as sns; sns.set(style="ticks", color_codes=True)
   tips = sns.load_dataset("tips")
  g = sns.FacetGrid(tips, col="time", row="smoker")












X = np.transpose(x)
X = sm.add_constant(X)  # add intercept

# Fit
mu = (y + 0.5) / 2  # initialize mu
eta = np.log(mu / (1 - mu))  # initialize eta with the Bernoulli link

for i in range(10):
    w = mu * (1 - mu);  # variance function
    z = eta + (y - mu) / (mu * (1 - mu))  # working response
    mod = sm.WLS(z, X, weights=w).fit()  # weigthed regression
    eta = mod.predict()  # linear predictor
    mu = 1 / (1 + np.exp(-eta))  # fitted value
    print(mod.params)  # print iteration log

​

# Output
print(mod.summary())

​

# Write data as dictionary
mydata = {}
mydata['x'] = x
mydata['y'] = y

​

# fit using glm package

import statsmodels.formula.api as sm

from sklearn import preprocessing


mylogit = sm.glm(formula='y ~ x', data=mydata, family=sm.families.Binomial())
res = mylogit.fit()
print(res.summary())

['rel_tilt'])['decision']

import statsmodels.formula.api as smf















dtype = [('ref','float32'), ('decision','int32'), ('correct','int32')]
values = np.zeros(20, dtype=dtype)
index = ['Row'+str(i) for i in range(1, len(values)+1)]

df = pandas.DataFrame(values, index=index)

print(decision)
print(correct)


decision - correct

np.array(diff_tilt1 > 0).astype(int)
np.array(diff_tilt > 0).astype(int)

boundaries(b) = 0
c = b.astype(int)



import pandas as pd

pd.DataFrame(boundaries)

np.rad2deg(mean_tilt)
8


a = np.random.randint(0, 5, size=(5, 4))
>>> a
array([[4, 2, 1, 1],
       [3, 0, 1, 2],
       [2, 0, 1, 1],
       [4, 0, 2, 3],
       [0, 0, 0, 2]])
>>> b = a < 3
>>> b
array([[False,  True,  True,  True],
       [False,  True,  True,  True],
       [ True,  True,  True,  True],
       [False,  True,  True, False],
       [ True,  True,  True,  True]], dtype=bool)
>>>
>>> c = b.astype(int)
>>> c
array([[0, 1, 1, 1],
       [0, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 1, 1, 0],
       [1, 1, 1, 1]])


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

grouped = df.groupby('Team')

f.groupby(['id','mth'])['cost'].sum()

grouped = df.groupby(['Team','Rank'])


df1 = grouped['Points'].agg([np.sum, np.mean, np.std]).reset_index()

df['Points'].agg([np.sum, np.mean, np.std])
B['sum']




import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
df = sb.load_dataset('tips')
g = sb.FacetGrid(df, col = "sex", hue = "smoker")
g.map(plt.scatter, "total_bill", "tip")
plt.show(



)

def triangle(length, amplitude):
    section = length // 4
    for direction in (1, -1):
        for i in range(section):
            yield i * (amplitude / section) * direction
        for i in range(section):
            yield (amplitude - (i * (amplitude / section))) * direction




list(triangle(100, 0.5))


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
degs = np.deg2rad(np.linspace(1, 180, 180))
triangle = signal.sawtooth(4*degs, 0.5)
plt.plot(np.rad2deg(degs ), triangle)

deg_x = np.arange(0, 181, step=45)
plt.xticks(deg_x)
[plt.axvline(_x, linewidth=1, color='black') for _x in deg_x]
plt.axhline(y=0.0, color='r')


plt.axvline(x=deg_x.all,  color='black')


plt.axhline(y=0.5, xmin=0.0, xmax=1.0, color='r')