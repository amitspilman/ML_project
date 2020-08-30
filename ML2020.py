import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from numpy.distutils.fcompiler import none


from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisers import EqualWidthDiscretiser

        #first table
trainDF = pd.read_csv("Xy_train.csv")
    
id = trainDF['id']
age =  trainDF['age']                #####
gender = trainDF['gender']    
cp = trainDF['cp']  
trestbps  = trainDF['trestbps']       #####
chol  = trainDF['chol']               #####
fbs = trainDF['fbs']
restecg  = trainDF['restecg']
thalach  = trainDF['thalach']         #####
exang  = trainDF['exang']
oldpeak  = trainDF['oldpeak']         #####
slope  = trainDF['slope']
ca  = trainDF['ca']
thal  = trainDF['thal']

#         second table
testDF = pd.read_csv("X_test.csv")

id2 = testDF['id']
age2 = testDF['age']
gender2 = testDF['gender']
cp2 = testDF['cp']
trestbps2  = testDF['trestbps']
chol2  = testDF['chol']
fbs2 = testDF['fbs']
restecg2  = testDF['restecg']
thalach2  = testDF['thalach']
exang2  = testDF['exang']
oldpeak2  = testDF['oldpeak']
slope2  = testDF['slope']
ca2  = testDF['ca']
thal2  = testDF['thal']

#         age

age1=0
age2=0
age3=0
age4=0
age5=0
age6=0
age7=0

for i in age :
    if i <= 30 :
        age1+=1
    elif i <= 40:
        age2+=1
    elif i <= 50 :
        age3+=1
    elif i <= 60 :
        age4+=1
    elif i <= 70 :
        age5+=1
    elif i <= 80 :
        age6+=1
    else:
        age7+=1

#             cp

typical_angina=0
atypical_angina=0
non_anginal_pain=0
asymptomatic=0

for i in cp :
    if i == 0 :
        typical_angina = typical_angina + 1 
    if i == 1 :
        atypical_angina = atypical_angina + 1
    if i == 2 :
        non_anginal_pain = non_anginal_pain +1
    if i == 3 :
        asymptomatic = asymptomatic + 1

#             gender
count_male = 0
count_female = 0
 
for i in gender:
    if gender[i] == 0 :
        count_female += 1
    else:
        count_male += 1


#             trestbps
low_blood_pressure = 0
proper_blood_pressure = 0
high_blood_pressure = 0
  
for i in trestbps:
    if trestbps[i] > 140 :
        high_blood_pressure += 1
    elif trestbps[i] > 100:
        proper_blood_pressure += 1
    else :
        low_blood_pressure +=1
        
############################################################################
            #--new code--#

testDF.info()




############################################################################

#         # trestbps graph            
# plt.bar(x = ["low blood pressure","proper","high blood pressure"], height=[low_blood_pressure, proper_blood_pressure, high_blood_pressure])
# plt.xlabel('trestbps')
# plt.ylabel('amount')
# plt.show()



############################################################################

# plt.scatter(x=gender,y=ca,color=['r','b'])


# mosaic(trainDF, ['fbs', 'thal'])
# plt.show()
# mosaic(trainDF, ['fbs', 'ca'])
# plt.show()
# mosaic(trainDF, ['restecg', 'slope'])
# plt.show()
# mosaic(trainDF, ['cp', 'restecg'])
# plt.show()

############################################################################

# plt.scatter(trestbps, chol)
# plt.scatter(trestbps, thalach)
# plt.scatter(trestbps, oldpeak)
# plt.scatter(chol, trestbps)
# plt.scatter(chol, thalach)
# plt.scatter(chol, oldpeak)
# plt.scatter(thalach, trestbps)
# plt.scatter(thalach, chol)
# plt.scatter(thalach, oldpeak)
# plt.scatter(oldpeak, trestbps)
# plt.scatter(oldpeak, chol)
# plt.scatter(oldpeak, thalach)
# 
# plt.show()

############################################################################

#         # correlation matrix
# sns.heatmap(trainDF.drop('y', 1).corr(), annot=True, cmap='coolwarm')
# plt.show()

############################################################################

#        # Continuous variable graph
# sns.distplot(trainDF['chol'], color="skyblue" )
# plt.show()
# sns.distplot(trainDF['trestbps'], color="skyblue" )
# plt.show()
# sns.distplot(trainDF['thalach'], color="skyblue" )
# plt.show()
# sns.distplot(trainDF['oldpeak'], color="skyblue")
# plt.show()


#        # Continuous variable graph 2
# sns.kdeplot(trainDF['chol'], shade=True, bw=.5, color="olive")
# plt.show()
# sns.kdeplot(trainDF['trestbps'], shade=True, bw=.5, color="olive")
# plt.show()
# sns.kdeplot(trainDF['thalach'], shade=True, bw=.5, color="olive")
# plt.show()
# sns.kdeplot(trainDF['oldpeak'], shade=True, bw=.25, color="olive")
# plt.show()

############################################################################

        # oldpeak graph
# trainDF['oldpeak'].hist( rwidth=0.9, color='#607c8e')
# plt.xlabel('oldpeak')
# plt.ylabel('amount')
# plt.grid(axis='x', alpha=0.75)
# plt.show() 
  
############################################################################

#        #cp graph
# 
# plt.bar(x = ["typical angina","atypical angina", "non anginal pain", "asymptomatic"], height=[typical_angina, atypical_angina, non_anginal_pain, asymptomatic])
# plt.show()

############################################################################

#         # gender graph
# plt.bar(x = ["female","male"], height=[count_female,count_male])
# plt.show()

############################################################################

        # age graph            
# plt.bar(x = ["0-30","30-40", "40-50", "50-60", "60-70", "70-80", "80+"], height=[age1, age2, age3, age4,age5, age6,age7])
# plt.xlabel('age')
# plt.ylabel('amount')
# plt.show()

############################################################################

#        # age-trestbps graph
# plt.scatter(age,trestbps)
# plt.show()

############################################################################

#        # age-chol graph
# plt.scatter(age,chol)
# plt.show()

############################################################################

#         # update age 
# trainDF.loc[trainDF['age'] > 80, 'age'] = 80
# age = trainDF['age']
# print(age)

############################################################################

# #         # update age 
# trainDF.loc[trainDF['age'] > 80, 'age'] = 80
# age = trainDF['age']
# print(age)
# 
# sns.boxplot(x = trainDF['y'], y=trainDF['age'], data=trainDF)
# plt.show()

############################################################################

#        # minimize table to only Continuous variable
# tempDF = pd.read_csv("Xy_train.csv")
#  
# tempDF = tempDF.drop('id', 1)
# tempDF = tempDF.drop('age', 1)
# tempDF = tempDF.drop('gender', 1)
# tempDF = tempDF.drop('cp', 1)
# tempDF = tempDF.drop('fbs', 1)
# tempDF = tempDF.drop('restecg', 1)
# tempDF = tempDF.drop('exang', 1)
# tempDF = tempDF.drop('slope', 1)
# tempDF = tempDF.drop('ca', 1)
# tempDF = tempDF.drop('thal', 1)

############################################################################

# usefull notes
# 
# trainDF.info()

