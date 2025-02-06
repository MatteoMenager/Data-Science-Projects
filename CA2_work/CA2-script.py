# core data science libraries
import numpy as np
import pandas as pd

# statistical modules
import scipy.stats as stats 
import statsmodels.api as sm 
import pingouin as pg

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets

# set the default style for plots
plt.style.use("seaborn-darkgrid")

# matplotlib magic to show plots inline
%matplotlib inline

df = pd.read_csv('data/hour.csv', index_col='instant', parse_dates=['dteday'])
df.head(2)

df.rename(columns={'weathersit':'weather', 'mnth':'month', 'hr':'hour', 'yr':'year', 'hum':'humidity', 'cnt':'count', 'temp':'temperature', 'atemp':'feel_temp', 'dteday':'date'}, inplace=True)
df.head(2)

df['date'] = df.date + pd.to_timedelta(df.hour, unit='h')
df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
df.drop('date', axis=1, inplace=True)
df.head()


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train set shape: {train_set.shape}")
print(f"Test set shape: {test_set.shape}")

df.tail()

# check that casual + registered = count for all rows
(df['casual'] + df['registered'] == df['count']).value_counts()

df.info()

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False, sharex=False)
sns.boxplot(x='year', y='count', data=train_set, ax=axes[0])
sns.boxplot(x='year', y='casual', data=train_set, ax=axes[1])
sns.boxplot(x='year', y='registered', data=train_set, ax=axes[2])

df_EDA = train_set.copy()
df_EDA.season = pd.Categorical(df_EDA.season.map({2:'spring', 3:'summer', 4:'fall', 1:'winter'}), categories=['spring', 'summer', 'fall', 'winter'], ordered=True)
df_EDA.year = pd.Categorical(df_EDA.year.map({0:2011, 1:2012}), categories=[2011,2012], ordered=True)
df_EDA.hour = pd.Categorical(df_EDA.hour, categories=[i for i in range(24)], ordered=True)
df_EDA.month = pd.Categorical(df_EDA.month.map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}), categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], ordered=True)
df_EDA.weekday = pd.Categorical(df_EDA.weekday.map({0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'}), categories=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], ordered=True)
df_EDA.weather = pd.Categorical(df_EDA.weather.map({1:'Clear', 2:'OK', 3:'Bad', 4:'Very bad'}), categories=['Clear', 'OK', 'Bad', 'Very bad'], ordered=True)
df_EDA.workingday = pd.Categorical(df_EDA.workingday.map({0:'No', 1:'Yes'}), categories=['No', 'Yes'])
df_EDA.holiday = pd.Categorical(df_EDA.holiday.map({0:'No', 1:'Yes'}), categories=['No', 'Yes'])
df_EDA.temperature = df_EDA.temperature * 41
df_EDA.feel_temp = df_EDA.feel_temp * 50
df_EDA.humidity = df_EDA.humidity * 100
df_EDA.windspeed = df_EDA.windspeed * 67
df_EDA.head()

df_EDA.info()

casual_ratio = df_EDA['casual'].sum() / df_EDA['count'].sum()
casual_ratio

df_EDA.describe()

df_EDA.plot(kind='box', figsize=(20,6), subplots=True, layout=(2,4), sharex=False, sharey=False, vert=False)

numericals = ['temperature', 'feel_temp', 'humidity', 'windspeed']
categoricals = ['season', 'year', 'month', 'hour', 'weekday', 'weather', 'workingday', 'holiday']
targets = ['casual', 'registered', 'count']

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
for column, ax in zip(numericals + targets, axes.flatten()):
    sns.histplot(df_EDA[column], kde=True, ax=ax, bins=20)
    # plot median and mean as vertical lines
    ax.axvline(df_EDA[column].median(), color='r', linestyle='--', label='median')
    ax.axvline(df_EDA[column].mean(), color='g', linestyle='-', label='mean')
    ax.set_title(column)
    ax.legend()

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
for column, ax in zip(numericals + targets, axes.flatten()):
    pg.qqplot(df_EDA[column], dist='norm', ax=ax)
    ax.set_title(column)

df_EDA.corr().iloc[:,:-3]

# PCA on numericals
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df_EDA[numericals])
print(pca.explained_variance_ratio_)
pca3comp = PCA(n_components=3)
PCs = pca3comp.fit_transform(df_EDA[numericals])
PCs = pd.DataFrame(PCs, columns=['PC1', 'PC2', 'PC3'])
# corr between targets and PCs
pd.concat([PCs, df_EDA[targets].copy().reset_index().drop('date', axis=1)], axis=1).corr().iloc[3:,:-3]

# corr between PCs and numericals
pd.concat([PCs, df_EDA[numericals].copy().reset_index().drop('date', axis=1)], axis=1).corr().iloc[:3,3:]

numericals.remove('feel_temp')

fig, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 4))
for column, ax in zip(categoricals[-3:], axes):
    sns.countplot(x=column, data=df_EDA, ax=ax)
    ax.set_title(column)

categoricals.remove('holiday')

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='workingday', y='count', data=df_EDA, ax=axes[0])
sns.boxplot(x='workingday', y='count', data=df_EDA, ax=axes[1])

df_EDA.weather.value_counts()

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='weather', y='count', data=df_EDA, ax=axes[0])
sns.boxplot(x='weather', y='count', data=df_EDA, ax=axes[1])

# if df_EDA where wheather is very bad put bad instead
df_EDA.loc[df_EDA.weather == 'Very bad', 'weather'] = 'Bad'
# remove category very bad from categorical variable weather def
df_EDA.weather = pd.Categorical(df_EDA.weather, categories=['Clear', 'OK', 'Bad'], ordered=True)
df_EDA.weather.value_counts()

fig, axes = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(20, 16))
axes = axes.flatten()
for i, column in enumerate(categoricals):
    sns.pointplot(x=column, y='casual', data=df_EDA, ax=axes[i])
    sns.pointplot(x=column, y='registered', data=df_EDA, ax=axes[i+8])

sns.countplot(x='season', hue='weather', data=df_EDA)

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='season', y='temperature', data=df_EDA, ax=axes[0])
sns.boxplot(x='season', y='temperature', data=df_EDA, ax=axes[1])

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='season', y='humidity', data=df_EDA, ax=axes[0])
sns.boxplot(x='season', y='humidity', data=df_EDA, ax=axes[1])

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='weather', y='windspeed', data=df_EDA, ax=axes[0])
sns.boxplot(x='weather', y='windspeed', data=df_EDA, ax=axes[1])

df_REG = df_EDA.copy().drop(['holiday', 'feel_temp'], axis=1)
# use month number instead of month name (for cyclic variables)
df_REG.month = train_set.month
# same for weekday
df_REG.weekday = train_set.weekday
# add dummies for season, year, workingday, weather
df_REG = pd.get_dummies(df_REG, columns=['season', 'year', 'workingday', 'weather'], drop_first=True)
df_REG.head()

df_REG['hour_sin'] = np.sin(2*np.pi*df_REG.hour.astype(int)/24)
df_REG['hour_cos'] = np.cos(2*np.pi*df_REG.hour.astype(int)/24)
df_REG['month_sin'] = np.sin(2*np.pi*df_REG.month.astype(int)/12)
df_REG['month_cos'] = np.cos(2*np.pi*df_REG.month.astype(int)/12)
df_REG['weekday_sin'] = np.sin(2*np.pi*df_REG.weekday.astype(int)/7)
df_REG['weekday_cos'] = np.cos(2*np.pi*df_REG.weekday.astype(int)/7)
df_REG.head()

fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 6))
sns.pointplot(x='hour', y='registered', data=df_REG, ax = axes[0])
sns.boxplot(x='hour', y='registered', data=df_REG, ax = axes[1])

df_REG['is_peak_time'] = df_REG.hour.isin([8,17,18]).astype(int)
df_REG['is_larger_peak_time'] = df_REG.hour.isin([7,8,9,16,17,18,19]).astype(int)

#combine is_peak_time with workingday in new feature
df_REG['is_peak_work_commute'] = df_REG.is_peak_time*df_REG.workingday_Yes
df_REG['is_work_commute'] = df_REG.is_larger_peak_time*df_REG.workingday_Yes
df_REG.drop(['hour','month','weekday'], axis=1, inplace=True)
df_REG.head()

df_candidates = df_REG[["casual", "registered", "count"]].copy()
df_candidates["temperature"] = df_REG["temperature"]
df_candidates["temp_squared"] = df_REG["temperature"]**2
df_candidates["temp_cubed"] = df_REG["temperature"]**3
df_candidates["temp_p4"] = df_REG["temperature"]**4
df_candidates["temp_p5"] = df_REG["temperature"]**5
df_candidates["temp_log"] = np.log(df_REG["temperature"])
df_candidates["temp_exp"] = np.exp(df_REG["temperature"])
df_candidates["temp_sin"] = np.sin(df_REG["temperature"])
df_candidates["temp_cos"] = np.cos(df_REG["temperature"])
df_candidates["temp_tan"] = np.tan(df_REG["temperature"])

# same for humidity
df_candidates["humidity"] = df_REG["humidity"]
df_candidates["hum_squared"] = df_REG["humidity"]**2
df_candidates["hum_cubed"] = df_REG["humidity"]**3
df_candidates["hum_p4"] = df_REG["humidity"]**4
df_candidates["hum_p5"] = df_REG["humidity"]**5
df_candidates["hum_log"] = np.log(df_REG["humidity"])
df_candidates["hum_exp"] = np.exp(df_REG["humidity"])
df_candidates["hum_sin"] = np.sin(df_REG["humidity"])
df_candidates["hum_cos"] = np.cos(df_REG["humidity"])
df_candidates["hum_tan"] = np.tan(df_REG["humidity"])

# same for windspeed
df_candidates["windspeed"] = df_REG["windspeed"]
df_candidates["wind_squared"] = df_REG["windspeed"]**2
df_candidates["wind_cubed"] = df_REG["windspeed"]**3
df_candidates["wind_p4"] = df_REG["windspeed"]**4
df_candidates["wind_p5"] = df_REG["windspeed"]**5
df_candidates["wind_log"] = np.log(df_REG["windspeed"])
df_candidates["wind_exp"] = np.exp(df_REG["windspeed"])
df_candidates["wind_sin"] = np.sin(df_REG["windspeed"])
df_candidates["wind_cos"] = np.cos(df_REG["windspeed"])
df_candidates["wind_tan"] = np.tan(df_REG["windspeed"])

df_candidates.head()

candidates_corr = df_candidates.corr()

# for temperature, humidity and windspeed 
    # for casual, registered, count
        # get best 3 correlated features to target among derived functions including original feature (absolute value) with the coefficient 
derived_features_names = {
    "temperature":["temperature", "temp_squared", "temp_cubed", "temp_p4", "temp_p5", "temp_log", "temp_exp", "temp_sin", "temp_cos", "temp_tan"],
    "humidity":["humidity", "hum_squared", "hum_cubed", "hum_p4", "hum_p5", "hum_log", "hum_exp", "hum_sin", "hum_cos", "hum_tan"],
    "windspeed":["windspeed", "wind_squared", "wind_cubed", "wind_p4", "wind_p5", "wind_log", "wind_exp", "wind_sin", "wind_cos", "wind_tan"]
}
for feature in ["temperature", "humidity", "windspeed"] :
    print(f"\n---- FEATURE : {feature} ----")
    for target in ["casual", "registered", "count"] :
        print(f"\nTARGET : {target} ")
        corr = candidates_corr[target].copy()
        corr = corr[derived_features_names[feature]]
        corr = corr.sort_values(ascending=False, key=lambda col: np.abs(col))
        print(corr.head(5))

df_REG.drop('windspeed', axis=1, inplace=True)

corr= df_REG.corr()

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, square=True, vmin=-1, vmax=1, cmap=cmap)

fig, axes = plt.subplots(4, 5, sharex=False, sharey=False, figsize=(20, 16))
axes=axes.flatten()
for index, column in enumerate(df_REG.drop(['casual', 'registered', 'count'], axis=1).columns):
    sns.kdeplot(df_REG[column], ax=axes[index])

df_REG.info()

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def forward_selection_regression(features, target, cv=10):
    model = LinearRegression(fit_intercept=False)
    
    remaining_features = list(features.columns)
    selected_features = []
    model_error = []
    
    # compute the error for the smallest model with only a constant fitted (no need for cross validation for that)
    X = pd.DataFrame({'constant':np.ones(len(features))})
    X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_val)
    model_error.append(mean_squared_error(y_val, Y_pred, squared=False))
    
    model = LinearRegression(fit_intercept=True)
    last_min_error_feature = None
    last_min_error = None
    while last_min_error == None and len(remaining_features) > 0:
        #print(f"\n---{len(selected_features)} features selected for now---")
        for feature in remaining_features:
            X = features[selected_features + [feature]]
            cv_score = - cross_val_score(model, X, target, cv=cv, scoring='neg_root_mean_squared_error').mean()
            #print(f"Trying {feature} - CV score: {cv_score}")
            if last_min_error is None or cv_score < last_min_error:
                last_min_error = cv_score
                last_min_error_feature = feature
        if last_min_error < model_error[-1]:
            #print(f"Adding {last_min_error_feature} with error {last_min_error}")
            selected_features.append(last_min_error_feature)
            remaining_features.remove(last_min_error_feature)
            model_error.append(last_min_error)
            last_min_error = None
            last_min_error_feature = None
            
    selected_features = ["constant"] + selected_features
    return selected_features, model_error
    

def show_forward_selection_results(selected_features, model_error):
    # create a dataframe with selected features, their error and percentage decrease from previous one
    df = pd.DataFrame({'feature':selected_features, 'error':model_error})
    # add column with percentage decrease from previous error, put 0 for first feature
    df['pct_error_decrease'] = df.error.pct_change().fillna(0)
    return df


# descale temperature and humidity 
test_set.temperature = test_set.temperature * 41
test_set.humidity = test_set.humidity * 100

# seasons
test_set['season_winter'] = (test_set.season == 1).astype(int)
test_set['season_summer'] = (test_set.season == 3).astype(int)
test_set['season_fall'] = (test_set.season == 4).astype(int)

# year_2012
test_set['year_2012'] = test_set.year

# workingday_Yes
test_set['workingday_Yes'] = test_set.workingday

# weather
test_set['weather_OK'] = (test_set.weather == 2).astype(int)
test_set['weather_Bad'] = test_set.weather.isin([3, 4]).astype(int)

# hour_sin, hour_cos
test_set['hour_sin'] = np.sin(2*np.pi*test_set.hour/24)
test_set['hour_cos'] = np.cos(2*np.pi*test_set.hour/24)

# month_sin, month_cos
test_set['month_sin'] = np.sin(2*np.pi*test_set.month/12)
test_set['month_cos'] = np.cos(2*np.pi*test_set.month/12)

# weekday_sin, weekday_cos
test_set['weekday_sin'] = np.sin(2*np.pi*test_set.weekday/7)
test_set['weekday_cos'] = np.cos(2*np.pi*test_set.weekday/7)

# peak time
test_set['is_peak_time'] = test_set.hour.isin([8,17,18]).astype(int)
test_set['is_larger_peak_time'] = test_set.hour.isin([7,8,9,16,17,18,19]).astype(int)

# work commute
test_set['is_work_commute'] = test_set.is_larger_peak_time * test_set.workingday
test_set['is_peak_work_commute'] = test_set.is_peak_time * test_set.workingday

# keep same columns as in df_REG
test_set = test_set[['temperature', 'humidity', 'casual', 'registered', 'count',
       'season_summer', 'season_fall', 'season_winter', 'year_2012',
       'workingday_Yes', 'weather_OK', 'weather_Bad', 'hour_sin', 'hour_cos',
       'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'is_peak_time',
       'is_larger_peak_time', 'is_peak_work_commute', 'is_work_commute']]

test_set.info()

X = df_REG.drop(['casual', 'registered', 'count'], axis=1)
selected_features, model_error = forward_selection_regression(X, df_REG['casual'], cv=10)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_error)

show_forward_selection_results(selected_features, model_error)

final_casual_lin_reg_model = LinearRegression(fit_intercept=True)
casual_kept_features = ['temperature', 'hour_cos', 'workingday_Yes', 'hour_sin', 'year_2012', 'humidity']
final_casual_lin_reg_model.fit(df_REG[casual_kept_features], df_REG['casual'])

casual_test_pred = final_casual_lin_reg_model.predict(test_set[casual_kept_features])
casual_RMSE = mean_squared_error(test_set['casual'], casual_test_pred, squared=False)
casual_RMSE_normalized = casual_RMSE / test_set['casual'].mean()
print(f"RMSE: {casual_RMSE}, normalized RMSE: {casual_RMSE_normalized}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
casual_residuals = test_set['casual'] - casual_test_pred
casual_standardized_residuals = casual_residuals / casual_residuals.std()
sns.histplot(casual_standardized_residuals, kde=True, ax=axes[0])
axes[0].set_xlabel('Casual standardized residuals')
pg.qqplot(casual_standardized_residuals, dist='norm', ax=axes[1])

selected_features, model_error = forward_selection_regression(X, df_REG['registered'], cv=10)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_error)

show_forward_selection_results(selected_features, model_error)

final_registered_lin_reg_model = LinearRegression(fit_intercept=True)
registered_kept_features = ['is_work_commute', 'temperature', 'is_peak_work_commute', 'year_2012', 'hour_sin', 'hour_cos', 'weather_Bad', 'season_winter']
final_registered_lin_reg_model.fit(df_REG[registered_kept_features], df_REG['registered'])

registered_test_pred = final_registered_lin_reg_model.predict(test_set[registered_kept_features])
registered_RMSE = mean_squared_error(test_set['registered'], registered_test_pred, squared=False)
registered_RMSE_normalized = registered_RMSE / test_set['registered'].mean()
print(f"RMSE: {registered_RMSE}, normalized RMSE: {registered_RMSE_normalized}")

fig, axes = plt.subplots(1, 2, figsize=(14,4))
registered_residuals = test_set['registered'] - registered_test_pred
registered_standardized_residuals = registered_residuals / registered_residuals.std()
sns.histplot(registered_standardized_residuals, kde=True, ax=axes[0])
axes[0].set_xlabel('Registered standardized residuals')
pg.qqplot(registered_standardized_residuals, dist='norm', ax=axes[1])

selected_features, model_error = forward_selection_regression(X, df_REG['count'], cv=10)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_error)

show_forward_selection_results(selected_features, model_error)

final_count_lin_reg_model = LinearRegression(fit_intercept=True)
count_kept_features = ['is_work_commute', 'temperature', 'hour_sin', 'hour_cos', 'year_2012', 'is_peak_work_commute', 'workingday_Yes', 'weather_Bad', 'season_winter']
final_count_lin_reg_model.fit(df_REG[count_kept_features], df_REG['count'])

count_test_pred = final_count_lin_reg_model.predict(test_set[count_kept_features])
count_RMSE = mean_squared_error(test_set['count'], count_test_pred, squared=False)
count_RMSE_normalized = count_RMSE / test_set['count'].mean()
print(f"RMSE: {count_RMSE}, normalized RMSE: {count_RMSE_normalized}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
count_residuals = test_set['count'] - count_test_pred
count_standardized_residuals = count_residuals / count_residuals.std()
sns.histplot(count_standardized_residuals, kde=True, ax=axes[0])
axes[0].set_xlabel('Count standardized residuals')
pg.qqplot(count_standardized_residuals, dist='norm', ax=axes[1])

from sklearn import linear_model
regularized_model = linear_model.Lasso()
regularized_model.fit(df_REG.drop(['casual', 'registered', 'count'], axis=1), df_REG['count'])
regularization_prediction = regularized_model.predict(test_set.drop(['casual', 'registered', 'count'], axis=1))
# normalized rmse 
mean_squared_error(test_set['count'], regularization_prediction, squared=False) / test_set['count'].mean()

regularized_model.coef_

#show residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
regularization_residuals = test_set['count'] - regularization_prediction
regularization_standardized_residuals = regularization_residuals / regularization_residuals.std()
sns.histplot(regularization_standardized_residuals, kde=True, ax=axes[0])
axes[0].set_xlabel('Regularization standardized residuals')
pg.qqplot(regularization_standardized_residuals, dist='norm', ax=axes[1])

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca = PCA()
pca_scaler = StandardScaler()
pca.fit(pca_scaler.fit_transform(df_REG.drop(['casual', 'registered', 'count'], axis=1)))
pca.explained_variance_ratio_

plt.figure(figsize=(8, 4))
variance_curve_points = np.cumsum(pca.explained_variance_ratio_)
plt.plot([i+1 for i in range(len(variance_curve_points))],variance_curve_points)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.axvline(x=12, color='r', linestyle='--', label='12 components')
plt.axhline(y=variance_curve_points[11], color='g', linestyle='--', label=f'{variance_curve_points[11]:.2f}% explained variance')
plt.legend()
plt.show()

pca = PCA(n_components=12)
components = pca.fit_transform(pca_scaler.fit_transform(df_REG.drop(['casual', 'registered', 'count'], axis=1)))

PCA_model = LinearRegression(fit_intercept=True)
PCA_model.fit(components, df_REG['count'])

df_REG.drop(['casual', 'registered', 'count'], axis=1).columns

pca_prediction = PCA_model.predict(pca.transform(pca_scaler.transform(test_set.drop(['casual', 'registered', 'count'], axis=1))))
mean_squared_error(test_set['count'], pca_prediction, squared=False) / test_set['count'].mean()

# show residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
pca_residuals = test_set['count'] - pca_prediction
pca_standardized_residuals = pca_residuals / pca_residuals.std()
sns.histplot(pca_standardized_residuals, kde=True, ax=axes[0])
axes[0].set_xlabel('PCA standardized residuals')
pg.qqplot(pca_standardized_residuals, dist='norm', ax=axes[1])

sns.pointplot(x='hour', y='casual', data=train_set)

hour_shifted_abs = np.abs(train_set.hour.apply(lambda x: x+24 if x<4 else x) - 15)
hour_sa_x_casual = pd.concat([train_set.casual, hour_shifted_abs], axis=1)
sns.pointplot(x='hour', y='casual', data=hour_sa_x_casual)

hour_sa_x_casual.corr()

clf_train = df_REG.copy().drop(['casual', 'registered'], axis=1)
clf_test = test_set.copy().drop(['casual', 'registered'], axis=1)

quantiles = clf_train.quantile([0.25, 0.5, 0.75])['count']
quantile25limit = quantiles[0.25]
quantile50limit = quantiles[0.5]
quantile75limit = quantiles[0.75]
quantiles

# create target and assign quantile for train set
clf_train['countQ'] = np.nan
clf_train.loc[clf_train['count'] <= quantile25limit, 'countQ'] = 'Q1'
clf_train.loc[(clf_train['count'] <= quantile50limit) & (clf_train['count'] > quantile25limit), 'countQ'] = 'Q2'
clf_train.loc[(clf_train['count'] <= quantile75limit) & (clf_train['count'] > quantile50limit), 'countQ'] = 'Q3'
clf_train.loc[clf_train['count'] > quantile75limit, 'countQ'] = 'Q4'
clf_train.countQ = pd.Categorical(clf_train.countQ, categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)

# same for test set
clf_test['countQ'] = np.nan
clf_test.loc[clf_test['count'] <= quantile25limit, 'countQ'] = 'Q1'
clf_test.loc[(clf_test['count'] <= quantile50limit) & (clf_test['count'] > quantile25limit), 'countQ'] = 'Q2'
clf_test.loc[(clf_test['count'] <= quantile75limit) & (clf_test['count'] > quantile50limit), 'countQ'] = 'Q3'
clf_test.loc[clf_test['count'] > quantile75limit, 'countQ'] = 'Q4'
clf_test.countQ = pd.Categorical(clf_test.countQ, categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)

# drop count
clf_train.drop('count', axis=1, inplace=True)
clf_test.drop('count', axis=1, inplace=True)

clf_train.head()

clf_train.countQ.value_counts()

clf_scaler = StandardScaler() 
clf_scaler.fit(clf_train.drop('countQ', axis=1))
clf_X_train = pd.DataFrame(clf_scaler.transform(clf_train.drop('countQ', axis=1)), columns=clf_train.drop('countQ', axis=1).columns)
clf_y_train = clf_train['countQ']
clf_X_test = pd.DataFrame(clf_scaler.transform(clf_test.drop('countQ', axis=1)), columns=clf_test.drop('countQ', axis=1).columns)
clf_y_test = clf_test['countQ']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def forward_selection_logistic_regression(features, target, cv=10):
    model = LogisticRegression(fit_intercept=False)
    
    remaining_features = list(features.columns)
    selected_features = []
    model_score = []
    
    # compute the score for the smallest model with only a constant fitted (no need for cross validation for that)
    X = pd.DataFrame({'constant':np.ones(len(features))})
    X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_val)
    model_score.append(f1_score(y_val, Y_pred, average='micro'))
    
    model = LogisticRegression(fit_intercept=True)
    last_max_score_feature = None
    last_max_score = None
    while last_max_score == None and len(remaining_features) > 0:
        #print(f"\n---{len(selected_features)} features selected for now---")
        for feature in remaining_features:
            X = features[selected_features + [feature]]
            cv_score = cross_val_score(model, X, target, cv=cv, scoring='f1_micro').mean()
            #print(f"Trying {feature} - CV score: {cv_score}")
            if last_max_score is None or cv_score > last_max_score:
                last_max_score = cv_score
                last_max_score_feature = feature
        if last_max_score > model_score[-1]:
            #print(f"Adding {last_max_score_feature} with score {last_max_score}")
            selected_features.append(last_max_score_feature)
            remaining_features.remove(last_max_score_feature)
            model_score.append(last_max_score)
            last_max_score = None
            last_max_score_feature = None
            
    selected_features = ["constant"] + selected_features
    return selected_features, model_score

def show_clf_forward_selection_results(selected_features, model_score):
    # create a dataframe with selected features, their score and percentage increase from previous one
    df = pd.DataFrame({'feature':selected_features, 'score':model_score})
    # add column with percentage increase from previous score, put 0 for first feature
    df['pct_score_increase'] = df.score.pct_change().fillna(0)
    return df

selected_features, model_score = forward_selection_logistic_regression(clf_X_train, clf_y_train, cv=10)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_clf_forward_selection_results(selected_features, model_score)

final_log_reg_model = LogisticRegression(fit_intercept=True)
log_reg_kept_features = ['hour_sin', 'hour_cos', 'is_work_commute', 'temperature', 'workingday_Yes', 'year_2012', 'weather_Bad', 'season_winter']
final_log_reg_model.fit(clf_X_train[log_reg_kept_features], clf_y_train)

log_reg_pred = final_log_reg_model.predict(clf_X_test[log_reg_kept_features])
log_reg_F1 = f1_score(clf_y_test, log_reg_pred, average='micro')
print(f"F1: {log_reg_F1}")

log_reg_cf = pd.crosstab(clf_y_test, log_reg_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(log_reg_cf, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

from sklearn.neighbors import KNeighborsClassifier

def forward_selection_KNN(features, target, cv=10, k=int(len(clf_X_train)**0.5)):  
    model = KNeighborsClassifier(n_neighbors=k)
    remaining_features = list(features.columns)
    selected_features = []
    model_score = []
    last_max_score_feature = None
    last_max_score = None
    while last_max_score == None and len(remaining_features) > 0:
        #print(f"\n---{len(selected_features)} features selected for now---")
        for feature in remaining_features:
            X = features[selected_features + [feature]]
            cv_score = cross_val_score(model, X, target, cv=cv, scoring='f1_micro').mean()
            #print(f"Trying {feature} - CV score: {cv_score}")
            if last_max_score is None or cv_score > last_max_score:
                last_max_score = cv_score
                last_max_score_feature = feature
        if len(model_score) == 0 or last_max_score > model_score[-1]:
            #print(f"Adding {last_max_score_feature} with score {last_max_score}")
            selected_features.append(last_max_score_feature)
            remaining_features.remove(last_max_score_feature)
            model_score.append(last_max_score)
            last_max_score = None
            last_max_score_feature = None
            
    return selected_features, model_score

selected_features, model_score = forward_selection_KNN(clf_X_train, clf_y_train, cv=10)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_clf_forward_selection_results(selected_features, model_score)

knn_kept_features = ['hour_cos', 'workingday_Yes', 'season_winter', 'hour_sin', 'year_2012', 'temperature']

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, int(len(clf_X_train)**0.5)), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=5, verbose=1, scoring='f1_micro')
clf.fit(clf_X_train[knn_kept_features], clf_y_train)

# best pca component number
clf.best_params_

final_knn_model = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn_model.fit(clf_X_train[knn_kept_features], clf_y_train)

knn_pred = final_knn_model.predict(clf_X_test[knn_kept_features])
knn_F1 = f1_score(clf_y_test, knn_pred, average='micro')
print(f"F1: {knn_F1}")

knn_cf = pd.crosstab(clf_y_test, knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cf, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

import gower
dist_matrix = gower.gower_matrix(clf_X_train[knn_kept_features])
# train another knn model with gower distance and grid search for best k
pipe = Pipeline([('knn', KNeighborsClassifier(metric='precomputed'))])
search_space = [{'knn__n_neighbors': range(3, 30)}]
clf = GridSearchCV(pipe, search_space, cv=5, verbose=1, scoring='f1_micro')
clf.fit(dist_matrix, clf_y_train)
clf.best_params_

# compute final model using best k
final_gower_knn_model = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric='precomputed')
final_gower_knn_model.fit(dist_matrix, clf_y_train)
# Precomputed matrix of distances between test instances and training instances
dist_matrix_test = gower.gower_matrix(clf_X_test[knn_kept_features], clf_X_train[knn_kept_features])
# evaluate on test set
gower_knn_pred = final_gower_knn_model.predict(dist_matrix_test)
gower_knn_F1 = f1_score(clf_y_test, gower_knn_pred, average='micro')
print(f"F1: {gower_knn_F1}")


gower_knn_cf = pd.crosstab(clf_y_test, gower_knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(gower_knn_cf, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

def get_quartile(value):
    if value <= quantile25limit:
        return 'Q1'
    elif value <= quantile50limit:
        return 'Q2'
    elif value <= quantile75limit:
        return 'Q3'
    else:
        return 'Q4'

count_test_pred_Q = pd.Series(count_test_pred).apply(get_quartile)
count_test_pred_Q

linear_regression_F1 = f1_score(clf_y_test, count_test_pred_Q, average='micro')
print(f"F1: {linear_regression_F1}")

linear_reg_cf = pd.crosstab(clf_y_test, list(count_test_pred_Q), rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(linear_reg_cf, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

# compute classification accuracy for our 3 models 
from sklearn.metrics import accuracy_score
knn_accuracy = accuracy_score(clf_y_test, knn_pred)
log_reg_accuracy = accuracy_score(clf_y_test, log_reg_pred)
linear_reg_accuracy = accuracy_score(clf_y_test, count_test_pred_Q)
print(f"KNN accuracy: {knn_accuracy:.3f}, logistic regression accuracy: {log_reg_accuracy:.3f}, linear regression accuracy: {linear_reg_accuracy:.3f}")

