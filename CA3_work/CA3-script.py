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

# matplotlib magic to show plots inline
%matplotlib inline

#plots style
sns.set_style('darkgrid')
plt.style.use("seaborn-darkgrid")

df = pd.read_csv('data/Drug_Consumption.csv').drop(columns=['ID'])
df.head()

df.columns

df.rename(columns={'Nscore':'Neuroticism', 'Escore':'Extraversion', 'Oscore':'Openness', 'AScore':'Agreeableness', 'Cscore':'Conscientiousness', 'SS':'SensationSeeking', 'Caff':'Caffeine', 'Choc':'Chocolate','Semer':'Semeron'}, inplace=True)
df.columns

df.info()

targets = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Chocolate', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semeron', 'VSA']
models_targets = ['Alcohol', 'Caffeine', 'Coke', 'Heroin']
numericals = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsive', 'SensationSeeking']
categoricals = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
features = numericals + categoricals

targets_levels = ["Never Used", "Used over a Decade Ago", "Used in Last Decade", "Used in Last Year", "Used in Last Month", "Used in Last Week", "Used in Last Day"]

df[targets] = df[targets].replace(['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'], [0,1,2,3,4,5,6]).astype(int)
df[['Country', 'Gender', 'Ethnicity']] = df[['Country', 'Gender', 'Ethnicity']].astype('category')
df['Age'] = pd.Categorical(df['Age'], categories=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], ordered=True)
df['Education'] = pd.Categorical(df['Education'], categories=['Left school before 16 years', 'Left school at 16 years', 'Left school at 17 years', 'Left school at 18 years', 'Some college or university, no certificate or degree', 'Professional certificate/ diploma', 'University degree', 'Masters degree', 'Doctorate degree'], ordered=True)
df.info()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)
print(f"Train set shape: {train_set.shape}, test set shape: {test_set.shape}")

train_set.tail()

train_set.isna().sum()

train_set.Semeron.map(lambda x: targets_levels[x]).value_counts()

train_set.describe()

train_set[numericals].plot(kind='box', subplots=True, layout=(2,4), figsize=(20,10), sharex=False, sharey=False)

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
for column, ax in zip(numericals, axes.flatten()):
    sns.histplot(train_set[column], kde=True, ax=ax, bins=20)
    # plot median and mean as vertical lines
    ax.axvline(train_set[column].median(), color='r', linestyle='--', label='median')
    ax.axvline(train_set[column].mean(), color='g', linestyle='-', label='mean')
    ax.legend()
fig.delaxes(axes[1][3])

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
for column, ax in zip(numericals, axes.flatten()):
    pg.qqplot(train_set[column], dist='norm', ax=ax)
    ax.set_title(column)
fig.delaxes(axes[1][3])

fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 8))
for column, ax in zip(categoricals, axes.flatten()):
    sns.countplot(x=column, data=train_set, ax=ax)
    ax.tick_params(axis='x', labelrotation=45)
fig.delaxes(axes[1][2])

fig, axes = plt.subplots(5, 4, sharex=False, sharey=False, figsize=(20, 20))
for column, ax in zip(targets, axes.flatten()):
    sns.countplot(x=column, data=train_set, ax=ax)
fig.delaxes(axes[4][3])

fig, axes = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(16, 3))
for column, ax in zip(models_targets, axes):
    sns.countplot(x=column, data=train_set, ax=ax)
    ax.tick_params(axis='x', labelrotation=45)

corrs = train_set[numericals+models_targets].corr()
corrs

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corrs, square=True, vmin=-1, vmax=1, cmap=cmap)

fig, axes = plt.subplots(7, 4, sharex=False, sharey=False, figsize=(20, 30))
for i, feature in enumerate(numericals):
    for j, target in enumerate(models_targets):
        sns.pointplot(x=target, y=feature, data=train_set, ax=axes[i][j])

fig, axes = plt.subplots(5, 4, sharex=False, sharey=False, figsize=(20, 20))
for i, feature in enumerate(categoricals):
    for j, target in enumerate(models_targets):
        sns.pointplot(x=feature, y=target, data=train_set, ax=axes[i][j])
        axes[i][j].tick_params(axis='x', labelrotation=30)

def show_chi2_contributions(feature, target, data=train_set):
    observed = pd.crosstab(data[feature], data[target], margins=True)
    expected = observed.copy()
    for i in expected.index[:-1]:
        for j in expected.columns[:-1]:
            expected.loc[i,j] = expected.loc['All',j] * expected.loc[i,'All'] / expected.loc['All','All']  
    chi2_contribution = ((observed-expected)**2 / expected)
    # add totals
    chi2_contribution.loc['All'] = chi2_contribution.sum()
    chi2_contribution.loc[:,'All'] = chi2_contribution.sum(axis=1)
    # print chi2 p value
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f"Chi2 p-value: {p}")
    return chi2_contribution

show_chi2_contributions('Ethnicity', 'Alcohol')

# encode category Gender as 0 and 1 in new feature called isMale
train_set["isMale"] = (train_set["Gender"] == 'M').astype(int)
# same on test set
test_set["isMale"] = (test_set["Gender"] == 'M').astype(int)

# for Age combine 55-64 and 65+ into 55+ category 
train_set["Age"] = train_set["Age"].replace(['55-64', '65+'], '55+')
# same on test set
test_set["Age"] = test_set["Age"].replace(['55-64', '65+'], '55+')

# encode category Age as 0, 1, 2, 3, 4 in new feature called AgeEncoded
train_set["AgeEncoded"] = train_set["Age"].cat.codes
# same on test set
test_set["AgeEncoded"] = test_set["Age"].cat.codes

# add young and middle age categories
train_set["isYoung"] = (train_set["Age"].isin(['18-24', '25-34'])).astype(int)
train_set["isMiddleAge"] = (train_set["Age"].isin(['35-44', '45-54'])).astype(int)
# same on test set
test_set["isYoung"] = (test_set["Age"].isin(['18-24', '25-34'])).astype(int)
test_set["isMiddleAge"] = (test_set["Age"].isin(['35-44', '45-54'])).astype(int)

# encode category Education as 0, 1, 2, 3, 4, 5, 6, 7, 8 in new feature called EducationEncoded
train_set["EducationEncoded"] = train_set["Education"].cat.codes
# same on test set
test_set["EducationEncoded"] = test_set["Education"].cat.codes

# add wentToCollege category
train_set["wentToCollege"] = (train_set["Education"].isin(['Some college or university, no certificate or degree', 'Professional certificate/ diploma', 'University degree', 'Masters degree', 'Doctorate degree'])).astype(int)
# same on test set
test_set["wentToCollege"] = (test_set["Education"].isin(['Some college or university, no certificate or degree', 'Professional certificate/ diploma', 'University degree', 'Masters degree', 'Doctorate degree'])).astype(int)

# is USA and is UK categories
train_set["isUSA"] = (train_set["Country"] == 'USA').astype(int)
train_set["isUK"] = (train_set["Country"] == 'UK').astype(int)
# same on test set
test_set["isUSA"] = (test_set["Country"] == 'USA').astype(int)
test_set["isUK"] = (test_set["Country"] == 'UK').astype(int)

# Ethnicity dummies (white, black, asian)
train_set["isWhite"] = (train_set.Ethnicity == 'White').astype(int)
train_set["isAsian"] = (train_set.Ethnicity == 'Asian').astype(int)
train_set["isBlack"] = (train_set.Ethnicity == 'Black').astype(int)
# same on test set
test_set["isWhite"] = (test_set.Ethnicity == 'White').astype(int)
test_set["isAsian"] = (test_set.Ethnicity == 'Asian').astype(int)
test_set["isBlack"] = (test_set.Ethnicity == 'Black').astype(int)

categoricals_encoded = ['isMale', 'AgeEncoded', 'isYoung', 'isMiddleAge', 'EducationEncoded', 'wentToCollege', 'isUSA', 'isUK', 'isWhite', 'isAsian', 'isBlack']
clf_features = numericals + categoricals_encoded

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit scaler on train set and transform both train and test sets
train_set[clf_features] = scaler.fit_transform(train_set[clf_features])
test_set[clf_features] = scaler.transform(test_set[clf_features])

train_set[clf_features].info()

from sklearn.model_selection import cross_val_score

def forward_selection(features, target, model, cv=10): 
    remaining_features = list(features.columns)
    selected_features = []
    model_score = []
    last_max_score_feature = None
    last_max_score = None
    while last_max_score == None and len(remaining_features) > 0:
        #print(f"\n---{len(selected_features)} features selected for now---")
        for feature in remaining_features:
            X = features[selected_features + [feature]]
            cv_score = cross_val_score(model, X, target, cv=cv, scoring='f1_macro').mean()
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

def show_forward_selection_results(selected_features, model_score):
    # create a dataframe with selected features, their score and percentage increase from previous one
    df = pd.DataFrame({'feature':selected_features, 'score':model_score})
    # add column with percentage increase from previous score, put 0 for first feature
    df['pct_score_increase'] = df.score.pct_change().fillna(0)
    return df

# create decision tree model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
selected_features, model_score = forward_selection(train_set[clf_features], train_set['Alcohol'], dt)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

decision_tree_kept_features = selected_features[:3]

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([('classifier', DecisionTreeClassifier(random_state=42))])
search_space = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'], 'classifier__max_depth': [None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}]
clf = GridSearchCV(pipe, search_space, cv=5, verbose=1, scoring='f1_macro')
clf.fit(train_set[decision_tree_kept_features], train_set['Alcohol'])
clf.best_params_

# fit model on all train data with best params
final_dt = DecisionTreeClassifier(criterion=clf.best_params_['classifier__criterion'], max_depth=clf.best_params_['classifier__max_depth'], random_state=42)
final_dt.fit(train_set[decision_tree_kept_features], train_set['Alcohol'])

# predict on test set
dt_pred = final_dt.predict(test_set[decision_tree_kept_features])

from sklearn.metrics import f1_score, classification_report
print("F1 score:", f1_score(test_set['Alcohol'], dt_pred, average='macro'))
# print classification report
print(classification_report(test_set['Alcohol'], dt_pred))

dt_cm = pd.crosstab(test_set['Alcohol'], dt_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

# for target alcohol, derive new target grouped_alcohol with 3 levels (not_user : 0, occasional : 1, recent : 2) (0:[0,1], 1:[2,3,4], 2:[5,6])
train_set['groupedAlcohol'] = train_set['Alcohol'].map(lambda x: 0 if x in [0,1] else 1 if x in [2,3,4] else 2)
test_set['groupedAlcohol'] = test_set['Alcohol'].map(lambda x: 0 if x in [0,1] else 1 if x in [2,3,4] else 2)

from imblearn.over_sampling import SMOTE

print(train_set.groupedAlcohol.value_counts())

# use SMOTE to oversample groupedAlcohol target
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(train_set[clf_features], train_set['groupedAlcohol'])
y_train.value_counts()

dt = DecisionTreeClassifier(random_state=42)
selected_features, model_score = forward_selection(X_train, y_train, dt)
#selected_features, model_score = forward_selection(train_set[clf_features], train_set['groupedAlcohol'], dt) # leads to 0.42 at most
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

grouped_alcohol_kept_features = selected_features[:6]

pipe = Pipeline([('classifier', DecisionTreeClassifier(random_state=42))])
search_space = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'], 'classifier__max_depth': list(range(2, 15))}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[grouped_alcohol_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_dt = DecisionTreeClassifier(criterion=clf.best_params_['classifier__criterion'], max_depth=clf.best_params_['classifier__max_depth'], random_state=42)
final_dt.fit(X_train[grouped_alcohol_kept_features], y_train)

# predict on test set
dt_pred = final_dt.predict(test_set[grouped_alcohol_kept_features])

print("F1 score:", f1_score(test_set['groupedAlcohol'], dt_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedAlcohol'], dt_pred))

dt_cm = pd.crosstab(test_set['groupedAlcohol'], dt_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
selected_features, model_score = forward_selection(X_train, y_train, knn)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

knn_kept_features = selected_features[:8]

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[knn_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_knn = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn.fit(X_train[knn_kept_features], y_train)

# predict on test set
knn_pred = final_knn.predict(test_set[knn_kept_features])
print("F1 score:", f1_score(test_set['groupedAlcohol'], knn_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedAlcohol'], knn_pred))

knn_cm = pd.crosstab(test_set['groupedAlcohol'], knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

# predict on train set
knn_pred = final_knn.predict(X_train[knn_kept_features])
print("F1 score:", f1_score(y_train, knn_pred, average='macro'))
# print classification report
print(classification_report(y_train, knn_pred))

knn_cm = pd.crosstab(y_train, knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_knn, X_train[knn_kept_features], y_train, cv=10, scoring='f1_macro').mean()

train_set['groupedCaffeine'] = train_set['Caffeine'].map(lambda x: 0 if x in [0,1] else 1 if x in [2,3,4] else 2)
test_set['groupedCaffeine'] = test_set['Caffeine'].map(lambda x: 0 if x in [0,1] else 1 if x in [2,3,4] else 2)
train_set['groupedCoke'] = train_set['Coke'].map(lambda x: 0 if x in [0] else 1 if x in [1,2,3] else 2)
test_set['groupedCoke'] = test_set['Coke'].map(lambda x: 0 if x in [0] else 1 if x in [1,2,3] else 2)
train_set['groupedHeroin'] = train_set['Heroin'].map(lambda x: 0 if x in [0] else 1 if x in [1,2,3] else 2)
test_set['groupedHeroin'] = test_set['Heroin'].map(lambda x: 0 if x in [0] else 1 if x in [1,2,3] else 2)

X_train, y_train = sm.fit_resample(train_set[clf_features], train_set['groupedCaffeine'])

dt = DecisionTreeClassifier(random_state=42)
selected_features, model_score = forward_selection(X_train, y_train, dt)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

dt_kept_features = selected_features[:5]

pipe = Pipeline([('classifier', DecisionTreeClassifier(random_state=42))])
search_space = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'], 'classifier__max_depth': list(range(2, 15))}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[dt_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_dt = DecisionTreeClassifier(criterion=clf.best_params_['classifier__criterion'], max_depth=clf.best_params_['classifier__max_depth'], random_state=42)
final_dt.fit(X_train[dt_kept_features], y_train)

# predict on test set
dt_pred = final_dt.predict(test_set[dt_kept_features])
print("F1 score:", f1_score(test_set['groupedCaffeine'], dt_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedCaffeine'], dt_pred))

dt_cm = pd.crosstab(test_set['groupedCaffeine'], dt_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_dt, X_train[dt_kept_features], y_train, cv=10, scoring='f1_macro').mean()

knn = KNeighborsClassifier()
selected_features, model_score = forward_selection(X_train, y_train, knn)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

knn_kept_features = selected_features[:10]

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[knn_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_knn = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn.fit(X_train[knn_kept_features], y_train)

# predict on test set
knn_pred = final_knn.predict(test_set[knn_kept_features])
print("F1 score:", f1_score(test_set['groupedCaffeine'], knn_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedCaffeine'], knn_pred))

knn_cm = pd.crosstab(test_set['groupedCaffeine'], knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_knn, X_train[knn_kept_features], y_train, cv=10, scoring='f1_macro').mean()

X_train, y_train = sm.fit_resample(train_set[clf_features], train_set['groupedCoke'])

dt = DecisionTreeClassifier(random_state=42)
selected_features, model_score = forward_selection(X_train, y_train, dt)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

dt_kept_features = selected_features[:8]

pipe = Pipeline([('classifier', DecisionTreeClassifier(random_state=42))])
search_space = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'], 'classifier__max_depth': list(range(2, 15))}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[dt_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_dt = DecisionTreeClassifier(criterion=clf.best_params_['classifier__criterion'], max_depth=clf.best_params_['classifier__max_depth'], random_state=42)
final_dt.fit(X_train[dt_kept_features], y_train)

# predict on test set
dt_pred = final_dt.predict(test_set[dt_kept_features])
print("F1 score:", f1_score(test_set['groupedCoke'], dt_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedCoke'], dt_pred))

dt_cm = pd.crosstab(test_set['groupedCoke'], dt_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_dt, X_train[dt_kept_features], y_train, cv=10, scoring='f1_macro').mean()

knn = KNeighborsClassifier()
selected_features, model_score = forward_selection(X_train, y_train, knn)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

knn_kept_features = selected_features

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[knn_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_knn = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn.fit(X_train[knn_kept_features], y_train)

# predict on test set
knn_pred = final_knn.predict(test_set[knn_kept_features])
print("F1 score:", f1_score(test_set['groupedCoke'], knn_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedCoke'], knn_pred))

knn_cm = pd.crosstab(test_set['groupedCoke'], knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_knn, X_train[knn_kept_features], y_train, cv=10, scoring='f1_macro').mean()

X_train, y_train = sm.fit_resample(train_set[clf_features], train_set['groupedHeroin'])

dt = DecisionTreeClassifier(random_state=42)
selected_features, model_score = forward_selection(X_train, y_train, dt)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

dt_kept_features = selected_features

pipe = Pipeline([('classifier', DecisionTreeClassifier(random_state=42))])
search_space = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'], 'classifier__max_depth': list(range(2, 15))}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[dt_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_dt = DecisionTreeClassifier(criterion=clf.best_params_['classifier__criterion'], max_depth=clf.best_params_['classifier__max_depth'], random_state=42)
final_dt.fit(X_train[dt_kept_features], y_train)

# predict on test set
dt_pred = final_dt.predict(test_set[dt_kept_features])
print("F1 score:", f1_score(test_set['groupedHeroin'], dt_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedHeroin'], dt_pred))

dt_cm = pd.crosstab(test_set['groupedHeroin'], dt_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_dt, X_train[dt_kept_features], y_train, cv=10, scoring='f1_macro').mean()

knn = KNeighborsClassifier()
selected_features, model_score = forward_selection(X_train, y_train, knn)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

knn_kept_features = selected_features[:9]

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[knn_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_knn = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn.fit(X_train[knn_kept_features], y_train)

# predict on test set
knn_pred = final_knn.predict(test_set[knn_kept_features])
print("F1 score:", f1_score(test_set['groupedHeroin'], knn_pred, average='macro'))
# print classification report
print(classification_report(test_set['groupedHeroin'], knn_pred))

knn_cm = pd.crosstab(test_set['groupedHeroin'], knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_knn, X_train[knn_kept_features], y_train, cv=10, scoring='f1_macro').mean()

train_set['binaryHeroin'] = train_set['Heroin'].map(lambda x: 0 if x in [0, 1] else 1)
test_set['binaryHeroin'] = test_set['Heroin'].map(lambda x: 0 if x in [0, 1] else 1)

X_train, y_train = sm.fit_resample(train_set[clf_features], train_set['binaryHeroin'])

knn = KNeighborsClassifier()
selected_features, model_score = forward_selection(X_train, y_train, knn)
fig = plt.figure(figsize=(20, 6))
plt.plot(selected_features, model_score)

show_forward_selection_results(selected_features, model_score)

knn_kept_features = selected_features

pipe = Pipeline([('knn', KNeighborsClassifier())])
search_space = [{'knn__n_neighbors': range(3, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev']}]
clf = GridSearchCV(pipe, search_space, cv=10, verbose=1, scoring='f1_macro')
clf.fit(X_train[knn_kept_features], y_train)
print(clf.best_params_)
print(clf.best_score_)

# fit model on all train data with best params
final_knn = KNeighborsClassifier(n_neighbors=clf.best_params_['knn__n_neighbors'], metric=clf.best_params_['knn__metric'])
final_knn.fit(X_train[knn_kept_features], y_train)

# predict on test set
knn_pred = final_knn.predict(test_set[knn_kept_features])
print("F1 score:", f1_score(test_set['binaryHeroin'], knn_pred, average='macro'))
# print classification report
print(classification_report(test_set['binaryHeroin'], knn_pred))

knn_cm = pd.crosstab(test_set['binaryHeroin'], knn_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6,5))
g = sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
g.xaxis.set_label_position('top')

cross_val_score(final_knn, X_train[knn_kept_features], y_train, cv=10, scoring='f1_macro').mean()

import gower
dist_matrix = gower.gower_matrix(train_set[clf_features])

# hierarchical clustering using sklearn and gower distance
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='complete')
agg.fit(dist_matrix)
train_set['cluster'] = agg.labels_

# show cluster sizes
train_set.cluster.value_counts()

fig, axes = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(20, 4))

clustering_cm = pd.crosstab(train_set['groupedAlcohol'], train_set['cluster'], rownames=['Alcohol grouped label'], colnames=['Cluster'])
sns.heatmap(clustering_cm, annot=True, cmap="Blues", fmt='g', ax=axes[0])

clustering_cm = pd.crosstab(train_set['groupedCaffeine'], train_set['cluster'], rownames=['Caffeine grouped label'], colnames=['Cluster'])
sns.heatmap(clustering_cm, annot=True, cmap="Blues", fmt='g', ax=axes[1])

clustering_cm = pd.crosstab(train_set['groupedCoke'], train_set['cluster'], rownames=['Cocaine grouped label'], colnames=['Cluster'])
sns.heatmap(clustering_cm, annot=True, cmap="Blues", fmt='g', ax=axes[2])

clustering_cm = pd.crosstab(train_set['groupedHeroin'], train_set['cluster'], rownames=['Heroin grouped label'], colnames=['Cluster'])
sns.heatmap(clustering_cm, annot=True, cmap="Blues", fmt='g', ax=axes[3])

show_chi2_contributions('cluster', 'groupedAlcohol')

show_chi2_contributions('cluster', 'groupedCaffeine')

show_chi2_contributions('cluster', 'groupedCoke')

show_chi2_contributions('cluster', 'groupedHeroin')

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
# point plot for each numerical feature against cluster
axes = axes.flatten()
for i, feature in enumerate(numericals):
    sns.pointplot(x='cluster', y=feature, data=train_set, ax=axes[i])

fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 8))
# count plot for each categorical feature against cluster
axes = axes.flatten()
for i, feature in enumerate(categoricals):
    sns.countplot(x='cluster', hue=feature, data=train_set, ax=axes[i])

sns.scatterplot(x='AgeEncoded', y='EducationEncoded', hue='cluster', data=train_set)

sns.scatterplot(x='Impulsive', y='SensationSeeking', hue='cluster', data=train_set)

# silouhette plot and score
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

silhouette_avg = silhouette_score(dist_matrix, train_set['cluster'])
print("For n_clusters =", 3, "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(dist_matrix, train_set['cluster'], metric='precomputed')

# Note : this code is produced by github copilot
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(train_set) + (3 + 1) * 10])
y_lower = 10
for i in range(3):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[train_set['cluster'] == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    #color = cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
ax1.set_title("Silhouette plot for the various clusters")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, init="k-means++")
kmeans.fit(dist_matrix)
train_set['cluster_2'] = kmeans.labels_

train_set.cluster_2.value_counts()

clustering_cm = pd.crosstab(train_set['cluster'], train_set['cluster_2'], rownames=['Hierarchichal clusters'], colnames=['KMeans clusters'])
plt.figure(figsize=(6,5))
sns.heatmap(clustering_cm, annot=True, cmap="Blues", fmt='g')


silhouette_avg = silhouette_score(dist_matrix, train_set['cluster_2'])
print("For n_clusters =", 3, "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(dist_matrix, train_set['cluster_2'], metric='precomputed')

# Note : this code is produced by github copilot
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(train_set) + (3 + 1) * 10])
y_lower = 10
for i in range(3):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[train_set['cluster_2'] == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    #color = cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
ax1.set_title("Silhouette plot for the various clusters")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

show_chi2_contributions('cluster_2', 'groupedAlcohol')

show_chi2_contributions('cluster_2', 'groupedCaffeine')

show_chi2_contributions('cluster_2', 'groupedCoke')

show_chi2_contributions('cluster_2', 'groupedHeroin')

fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(20, 8))
# point plot for each numerical feature against cluster
axes = axes.flatten()
for i, feature in enumerate(numericals):
    sns.pointplot(x='cluster_2', y=feature, data=train_set, ax=axes[i])

fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 8))
# count plot for each categorical feature against cluster
axes = axes.flatten()
for i, feature in enumerate(categoricals):
    sns.countplot(x='cluster_2', hue=feature, data=train_set, ax=axes[i])

