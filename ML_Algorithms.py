import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, confusion_matrix

#         import CSV files
Xy_train_DF1 = pd.read_csv("Xy_train.csv")                       #X,Y train
# Xy_train_DF2=Xy_train_DF1.drop('gender',1)
Xy_train_DF=Xy_train_DF1.drop('id',1)
# ca - Handling non accurate data  
indexNames_ca = Xy_train_DF.loc[Xy_train_DF['ca'] == 4].index
Xy_train_DF.drop(indexNames_ca , inplace=True)
# thal - Handling non accurate data by set to most common value
Xy_train_DF.loc[Xy_train_DF['thal'] == 0,'thal']= 2
# age - cancel discretization and drop non accurate data
indexNames_age = Xy_train_DF.loc[Xy_train_DF['age'] >120].index
Xy_train_DF.drop(indexNames_age , inplace=True)
#trestbps - make categorical
Xy_train_DF.loc[Xy_train_DF['trestbps'] >140,'trestbps']= 2     #trestbps > 140 set to 2
Xy_train_DF.loc[Xy_train_DF['trestbps'] >100, 'trestbps']= 1    #100 < trestbps <= 140 set to 1
Xy_train_DF.loc[Xy_train_DF['trestbps'] >2, 'trestbps']= 0      #trestbps <= 100 set to 0
#split train xy into x, y
X_train_DF=Xy_train_DF.drop('y',1)                              #X train
Y_train_DF=Xy_train_DF.loc[ : , ['y'] ]                         #Y train

scaler = StandardScaler()
size=0.3                                                        #validation %
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     function
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def print_accuricy_tree(data_type,depth1,x1,y1,model1,size1):
    print("model: decision tree: " , data_type)
    print("depth is: ",depth1,"    validation % is: ",size1)
    print(data_type," accuracy: ", accuracy_score(y_true=y1, y_pred=model1.predict(x1)))
    print("confusion matrix ","\n",confusion_matrix(y_true=y1, y_pred=model1.predict(x1)),"\n");
def print_accuricy_ANN(data_type,x1,y1,model1,size1):
    print("model: neural network:" , data_type)
    print("validation % is: ",size1)
    print(data_type," accuracy: ", accuracy_score(y_true=y1, y_pred=model1.predict(x1)))
    print("confusion matrix ","\n",confusion_matrix(y_true=y1, y_pred=model1.predict(x1)),"\n");
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     models
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# -------full decision tree model-------
# -------initialized data-------    
xtrain_TREE, xvalid_TREE, ytrain_TREE, yvalid_TREE = train_test_split(X_train_DF, Y_train_DF, test_size=size)   #split validation and train
# -------entropy-------
model_tree = DecisionTreeClassifier(max_depth=None, criterion='entropy', random_state=42)
# train and print basic model
model_tree.fit(xtrain_TREE,ytrain_TREE)
print(model_tree)
# -------print tree-------
plt.figure(figsize=(16,9))
plot_tree(model_tree, filled=True, class_names=True)
plt.show()
# -------print accuracy train, validation-------
print_accuricy_tree("train",'None',xtrain_TREE,ytrain_TREE,model_tree,size)
print_accuricy_tree("validation",'None',xvalid_TREE,yvalid_TREE,model_tree,size) 
# # -------Grid search-------
hyperparameters = {'max_depth': np.arange(1, 20),'criterion': ['entropy', 'gini'] ,'max_features': ['auto', 'sqrt', 'log2', None]}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),param_grid=hyperparameters,cv=10)
grid_search.fit(xtrain_TREE,ytrain_TREE)
best_model_tree = grid_search.best_estimator_
print('best_model',best_model_tree)
print( 'best_params_',grid_search.best_params_)
plt.figure(figsize=(16,9))
plot_tree(best_model_tree, filled=True, class_names=True)
plt.show()
best_model_tree.fit(xtrain_TREE,ytrain_TREE)
print_accuricy_tree("train",grid_search.best_params_['max_depth'],xtrain_TREE,ytrain_TREE,best_model_tree,size)
print_accuricy_tree("validation",grid_search.best_params_['max_depth'],xvalid_TREE,yvalid_TREE,best_model_tree,size) 
# -------feature importance-------
importances = best_model_tree.feature_importances_
print("feature importance table: ",importances)
print("feature importance: ", dict(zip(xtrain_TREE.columns, importances)))
print("features name: ", xtrain_TREE.columns)


# -------artificial neural network model-------    
# -------initialized data-------
X_train_DF_ANN = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
Y_train_DF_ANN = Y_train_DF
#split validation and train and normelized data
Xtrain_ANN, Xvalid_ANN, Ytrain_ANN, Yvalid_ANN = train_test_split(scaler.fit_transform(X_train_DF_ANN), Y_train_DF_ANN, test_size=size)
model_ANN = MLPClassifier(random_state=42,hidden_layer_sizes=(5),max_iter=500,activation='relu',verbose=True) 
#train and show basic model
model_ANN.fit(Xtrain_ANN, Ytrain_ANN)
print_accuricy_ANN("train",Xtrain_ANN,Ytrain_ANN,model_ANN,size)
print_accuricy_ANN("validation",Xvalid_ANN,Yvalid_ANN,model_ANN,size) 
# -------Grid search-------
hyperparameters_ANN = {'hidden_layer_sizes': [(4),(5),(6),(7),(3,2),(4,2),(6,3),(7,3),(7,4)],'activation':['tanh', 'relu'],'solver': ['sgd', 'adam'],'learning_rate_init' : [0.001,0.0001], 'alpha': [0.0001,0.00001]} 
Random_search_ANN = RandomizedSearchCV(estimator=MLPClassifier(random_state=42, max_iter=500,verbose=True),param_distributions=hyperparameters_ANN,cv=10)
Random_search_ANN.fit(Xtrain_ANN,Ytrain_ANN)
best_model_ANN = Random_search_ANN.best_estimator_
#train and show improve model
best_model_ANN.fit(Xtrain_ANN,Ytrain_ANN)
print("best model: ",best_model_ANN)
print("best parameters: ",Random_search_ANN.best_params_)
print("--normal model--")
print_accuricy_ANN("train",Xtrain_ANN,Ytrain_ANN,model_ANN,size)
print_accuricy_ANN("validation",Xvalid_ANN,Yvalid_ANN,model_ANN,size)
print("--best model--")
print_accuricy_ANN("train",Xtrain_ANN,Ytrain_ANN,best_model_ANN,size)
print_accuricy_ANN("validation",Xvalid_ANN,Yvalid_ANN,best_model_ANN,size)

# -------K-Means model-------
# -------initialized data-------
X_train_DF_KMeans = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
#normalized data  
xtrain_KMeans = scaler.fit_transform(X_train_DF_KMeans)
ytrain_KMeans = Y_train_DF
# -------create basic KMeans model-------
model_kmeans = KMeans(n_clusters = 2, random_state=42)
model_kmeans.fit(xtrain_KMeans)
y_kmeans = model_kmeans.predict(xtrain_KMeans)
print("100% data kmeans accuracy: ", accuracy_score(y_true=ytrain_KMeans, y_pred=y_kmeans))
print("confusion matrix ","\n",confusion_matrix(y_true=ytrain_KMeans, y_pred=y_kmeans),"\n");
 
# -------Create PCA - Reduce dimension-------
pca = PCA(n_components=2)
pca.fit(xtrain_KMeans)
print(pca.explained_variance_ratio_)            #what influence more, PC1 or PC2 (variance)
print(pca.explained_variance_ratio_.sum())      #sum of euclidean distance, as small as possible
kmeans_pca = pca.transform(xtrain_KMeans)
kmeans_pca = pd.DataFrame(kmeans_pca, columns=['PC1', 'PC2'])
kmeans_pca['y'] = ytrain_KMeans
new_model_kmeans = KMeans(n_clusters = 2, random_state=42)
new_model_kmeans.fit(xtrain_KMeans)
print(kmeans_pca.head(20))
sns.scatterplot(x='PC1', y='PC2', hue='y', data=kmeans_pca)
plt.show()

# -------number of clusters plots-------
dbi_list = []
sil_list = []
for n_clusters in range(2, 10,1):
    model_kmeans = KMeans(n_clusters = n_clusters, random_state=42, max_iter=1000, n_init=30)
    model_kmeans.fit(xtrain_KMeans)
    assignment = model_kmeans.predict(xtrain_KMeans)
    iner = model_kmeans.inertia_
    sil = silhouette_score(xtrain_KMeans, assignment)
    dbi = davies_bouldin_score(xtrain_KMeans, assignment)
    dbi_list.append(dbi)
    sil_list.append(sil)
   
plt.plot(range(2, 10, 1), sil_list, marker='o')             #high is better
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()
   
plt.plot(range(2, 10, 1), dbi_list, marker='o')             #low is better
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show() 
# -------create an updated KMeans model-------
model_kmeans = KMeans(n_clusters = 2, random_state=42, max_iter=1000, n_init=30)
model_kmeans.fit(xtrain_KMeans)
y_kmeans = model_kmeans.predict(xtrain_KMeans)
print("kmeans clusring accuracy: ", accuracy_score(y_true=ytrain_KMeans, y_pred=y_kmeans)) 
 
sns.scatterplot(x='PC1', y='PC2', hue='y', data=kmeans_pca)
plt.scatter(pca.transform(model_kmeans.cluster_centers_)[:, 0], pca.transform(model_kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.show()

# -------different clustering algorithm-------
Spectral_model = SpectralClustering(n_clusters=8, random_state=42, n_init=10, n_neighbors=10, degree=3)
y_kmeans = Spectral_model.fit_predict(xtrain_KMeans)
print("spectral clustring accuracy: ", accuracy_score(y_true=ytrain_KMeans, y_pred=y_kmeans))

# -------KMeans model 70% data equals to other algorithms-------
X_train = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
Y_train = Y_train_DF
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(scaler.fit_transform(X_train), Y_train, test_size=size)
new_model_kmeans = KMeans(n_clusters = 2, random_state=42, max_iter=500, n_init=10)
new_model_kmeans.fit(Xtrain)
print("70% (train) data kmeans accuracy: ", accuracy_score(y_true=Ytrain, y_pred=new_model_kmeans.predict(Xtrain)))
print("confusion matrix ","\n",confusion_matrix(y_true=Ytrain, y_pred=new_model_kmeans.predict(Xtrain)),"\n");
print("30% (valid) data kmeans accuracy: ", accuracy_score(y_true=Yvalid, y_pred=new_model_kmeans.predict(Xvalid)))
print("confusion matrix ","\n",confusion_matrix(y_true=Yvalid, y_pred=new_model_kmeans.predict(Xvalid)),"\n");
 
# graph PCA orgenized
pca = PCA(n_components=2)
X_train_DF_KMeans = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
xtrain_KMeans = scaler.fit_transform(X_train_DF_KMeans)
ytrain_KMeans = Y_train_DF
pca.fit(xtrain_KMeans)
  
kmeans_pca = pca.transform(xtrain_KMeans)
kmeans_pca = pd.DataFrame(kmeans_pca, columns=['PC1', 'PC2'])
kmeans_pca['clustering'] = model_kmeans.predict(xtrain_KMeans)
  
sns.scatterplot(x='PC1', y='PC2', hue='clustering', data=kmeans_pca)
plt.show()
  
sns.scatterplot(x='PC1', y='PC2', hue='clustering', data=kmeans_pca)
plt.scatter(pca.transform(model_kmeans.cluster_centers_)[:, 0], pca.transform(model_kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='green')
plt.show()

#  -------final conclusion-------
prepare data
X_test_DF = pd.read_csv("X_test.csv")
X_test_DF=X_test_DF.drop('id',1)
indexNames_ca = X_test_DF.loc[X_test_DF['ca'] == 4, 'ca'] = 0
X_test_DF.loc[X_test_DF['thal'] == 0,'thal']= 2
indexNames_age = X_test_DF.loc[X_test_DF['age'] >120, 'age']=120
X_test_DF.loc[X_test_DF['trestbps'] >140,'trestbps']= 2
X_test_DF.loc[X_test_DF['trestbps'] >100, 'trestbps']= 1
X_test_DF.loc[X_test_DF['trestbps'] >2, 'trestbps']= 0
   
Y_train = Y_train_DF
X_train_DF_ANN = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
X_test_DF_ANN = pd.get_dummies(X_test_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
X_train = scaler.fit_transform(X_train_DF_ANN)
X_test = scaler.fit_transform(X_test_DF_ANN)
# build best model
best_model = MLPClassifier(hidden_layer_sizes=(7,4),max_iter=3000,activation='relu', solver= 'sgd', learning_rate_init= 0.001, alpha = 0.00001,verbose=True) 
print(best_model)
best_model.fit(X_train, Y_train)
   
Y_test = best_model.predict(X_test)
df_prediction = pd.DataFrame(Y_test)
df_prediction.columns = ['y']
df_prediction.to_csv("Y_test_prediction.csv", index=False)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     important tools
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#    tune max_depth
res = pd.DataFrame()
for max_depth in range(1,20):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(xtrain_TREE, ytrain_TREE)
    res = res.append({'max_depth': max_depth,
                      'train_acc':accuracy_score(ytrain_TREE, model.predict(xtrain_TREE)),
                      'valid_acc':accuracy_score(yvalid_TREE, model.predict(xvalid_TREE))}, ignore_index=True)
plt.figure(figsize=(9, 4))
plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
plt.plot(res['max_depth'], res['valid_acc'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.xlabel("max depth")
plt.ylabel("accuracy")
plt.show()
print(res.sort_values('valid_acc', ascending=False))

#    K-fold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
df_dummies = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)
xdf=df_dummies
ydf=Y_train_DF
res = pd.DataFrame()
k = 0
for train_index, val_index in kfold.split(xdf):
    for max_depth in range(1,20):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        model.fit(xdf.iloc[train_index], ydf.iloc[train_index])
        acc = accuracy_score(ydf.iloc[val_index,:], model.predict(xdf.iloc[val_index,:]))
        res = res.append({'max_depth': max_depth,'k':k,'acc': acc}, ignore_index=True)
print(res[['max_depth', 'acc']].groupby(['max_depth']).mean().reset_index().sort_values('acc', ascending=False).head(5))
print(res[['max_depth', 'acc']].groupby(['max_depth']).std().reset_index().sort_values('acc', ascending=False).head(10))


# neural network hidden layer graph
train_accs = []
test_accs = []
for hidden_layer_sizes in range(1, 100):
        print("size of layers: ",hidden_layer_sizes)
        model = MLPClassifier(random_state=1,hidden_layer_sizes=(hidden_layer_sizes),max_iter=100,activation='relu',verbose=False,learning_rate_init=0.001)
        model.fit(Xtrain_ANN, Ytrain_ANN)
        train_acc = model.score(Xtrain_ANN, Ytrain_ANN)
        train_accs.append(train_acc)
        test_acc = model.score(Xvalid_ANN, Yvalid_ANN)
        test_accs.append(test_acc)
 
plt.figure(figsize=(7, 4))
plt.plot(range(1, 100), train_accs, label='Train')
plt.plot(range(1, 100), test_accs, label='Test')
plt.legend()
plt.show()

# build a few graphs for K-Means
x= pd.DataFrame(X_train_DF)
x=x.drop('cp',1)
x=x.drop('restecg',1)
x=x.drop('slope',1)
x=x.drop('ca',1)
x=x.drop('thal',1)
x=x.drop('age',1)
x=x.drop('gender',1)
 
y= pd.DataFrame(X_train_DF)
y=y.drop('trestbps',1)
y=y.drop('chol',1)
y=y.drop('fbs',1)
y=y.drop('thalach',1)
y=y.drop('exang',1)
y=y.drop('oldpeak',1)
 
x['y']=Y_train_DF
sns.pairplot(x, hue='y')
plt.show()
 
y['y']=Y_train_DF
sns.pairplot(y, hue='y')
plt.show()

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     notes
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#         scatter chart
# col1=Xy_train_DF['trestbps']
# col2=Xy_train_DF['chol']
# labelx='label x'
# labely='label y'
# plt.scatter(col1,col2)
# plt.xlabel(labelx)
# plt.ylabel(labely)
 
#         bar chart
# high1=1
# high2=2
# high3=3
# col1="A"
# col2="B"
# col3="C"
# labelx='label x'
# labely='label y'
# plt.bar(x = [col1,col2,col3], height=[high1, high2, high3])
# plt.xlabel(labelx)
# plt.ylabel(labely)
 
#        box-plot chart
# sns.boxplot(x = 'y', y='age', data=Xy_train_DF)
  
#        correlation matrix
# dataFrame = Xy_train_DF.loc[:,['trestbps','chol','thalach','oldpeak']]
# sns.heatmap(dataFrame.corr(), annot=True, cmap='coolwarm')
# plt.show()

#         useful notes
# Xy_train_DF.loc[Xy_train_DF['age'] > 80 , 'age']= 80    #fixing age problem
# trainDF.info()                                          #print info
# trainDF.describe()                                      #print describe
# plt.show()                                              #print plot
# print (Xy_train_DF.head())                              #print first 5 rows
# print (Xy_train_DF.tail())                              #print last 5 rows
# Xy_train_DF = Xy_train_DF.loc[:,['slope','cp','thalach']]
# print(Xtrain_DF.shape[0])
