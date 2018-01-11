from netCDF4 import Dataset as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_file='data/atmPrf_C001.2010.001.00.00.G20_2013.3520.nc'
test_ncfile=dt(test_file,'r')

test_lat = np.array(test_ncfile.variables['Lat'][:]) 
test_lon = np.array(test_ncfile.variables['Lon'][:])
test_pres = np.array(test_ncfile.variables['Pres'][:]) 
test_temp = np.array(test_ncfile.variables['Temp'][:])

test_s=(len(test_lat),4)
test_data=np.zeros(test_s)

for i in range(len(test_lat)):
    test_data[i][0]=test_pres[i]
    test_data[i][1]=test_lat[i]
    test_data[i][2]=test_lon[i]
    test_data[i][3]=test_temp[i]


train_file='ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2010010100-2010063018.nc'
train_ncfile=dt(train_file,'r')

train_time = np.array(train_ncfile.variables['time'][:])  #len 724
train_plev = np.array(train_ncfile.variables['plev'][:])  #len 87
train_lat  = np.array(train_ncfile.variables['lat'][:])   #len 32
train_lon  = np.array(train_ncfile.variables['lon'][:])   #len 64
train_temp = np.array(train_ncfile.variables['ta'])


a = test_file.split('/')[1]
a = a.split('.')
month = int(a[0][10])
day = int(a[2])
hour = int(a[3])

if hour>=0 and hour<=3:
    h=0
elif hour>=4 and hour<=9:
    h=1
elif hour>=10 and hour<=15:
    h=2
elif hour>=16 and hour<=21:
    h=3
else:
    h=4  
    
t = 4 * (30 * (month-1) + day - 1) + h
if t>723:
    t=723
    

train_s=(178176,4)
train_data=np.zeros(train_s)
p=0
for j in range(len(train_plev)):
    for k in range(len(train_lat)):
        for l in range(len(train_lon)):
            train_data[p][0]=train_plev[j]/100
            train_data[p][1]=train_lat[k]
            train_data[p][2]=train_lon[l]
            train_data[p][3]=train_temp[t][j][k][l]-273.15
            p+=1

            
x_train = train_data[:,0:3]
y_train = train_data[:,3]
x_test = test_data[:,0:3]
y_test = test_data[:,3]  


#linear regresion
'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)'''


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test)     

plt.scatter(test_pres, y_test, color = 'red')
plt.plot(test_pres, y_pred, color = 'blue')
plt.title('pressure vs tempreture (Test set)')
plt.xlabel('pressure')
plt.ylabel('tempreture')
plt.show()

a=[]
for i in range(2871):
    a.append(abs(y_test[i]-y_pred[i])/y_test[i])
print(sum(a)/2871)