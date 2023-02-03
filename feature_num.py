import binary_optimization as opt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from genetic_selection import GeneticSelectionCV
import pickle
import re
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR
import same_gene
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_selection import RFE
from numpy import mean
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True)

def ReadData(filePath):
    data = pd.read_csv(filePath,header=0)
    # data = data.loc[data.loc[:,'gene_name'].isin(gene),:] #selected same gene with gse15222
    #data = data.dropna(axis=0,how='any')
    data.index = data['gene_name'].values
    data = data.drop(labels='gene_name',axis=1)
    geneName = data.index.values
    labels = data.columns.values #0 没病，1 有病
    geneData = data.T
    geneData = preprocessing.scale(geneData,axis=1) #axis=1 363 row mean is 0;
    geneData = pd.DataFrame(geneData,columns=geneName)
    # print(geneData)
    label_list = []
    for i in labels:
        if 'CON' in i:
            label_list.append(0)
        else:
            label_list.append(1)
    return geneName,geneData,label_list

def Read15222(filePath):

    data = pd.read_csv(filePath,index_col=0,header=0)
    data = data.astype('float')
    data = data.T
    geneName = data.columns[1:]
    label = data.iloc[:,0]
    geneData = data.iloc[:,1:]
    geneData = preprocessing.scale(geneData,axis=0) #axis=1 363 row mean is 0;
    geneData = pd.DataFrame(geneData,columns=geneName)
    return geneName,geneData,label


if __name__ == "__main__":
    dataPath = 'voom_normalization_log2.csv'
    gse15222 = 'gse15222.csv'
    gse1297 = 'GSE1297.csv'
    gse199225 = 'GSE199225.csv'
    gse203206 = 'gse203206.csv'
    # geneName, geneData, label = ReadData(dataPath)
    # geneName,geneData,label = ReadData(gse15222)
    geneName, geneData, label = ReadData(gse1297)
    # geneName, geneData, label = ReadData(gse199225)
    # geneName, geneData, label = ReadData(gse203206)
    geneDataTrain,geneDataTest,labelTrain,labelTest = train_test_split(
        geneData,label,shuffle=True,train_size=0.8)

    geneNameFilter = geneName
    geneDataFilterTrain = geneDataTrain
    geneDataFilterTest = geneDataTest
    GA_model = RandomForestClassifier()
    op = ['gwo','cxuniform','cxonepoint','cxtwopoint','cxpartialymatched','cxuniformpartialymatched','cxordered','cxBlend','cxSimulatedBinary',
          'cxSimulatedBinaryBounded']
    # num = [5,15,25,35,45,55,65,75,85,95]
    num = [1,2,3,4,5,6,7,8,9]
    print(geneData.shape[1])

    # writer = pd.ExcelWriter(excelName)
    for i in num:
        print(i)
        excelName = 'com_num/GSE199225/' + str(i) + '/model validation num.xlsx'
        for k in op:
            print(k)
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1']
            GAGWO = GeneticSelectionCV(estimator=GA_model,n_jobs=-1,cv=5, scoring=None, fit_params=None,max_features=i,
                     n_population=200, crossover_proba=0.3, mutation_proba=0.2, n_generations=3, tournament_size=5, n_gen_no_change=None,cro=k
                     )
            GAGWO = GAGWO.fit(geneData, label)
            print('GAFS has been done')
        # 存
            name = 'com_num/GSE199225/' + str(i) + '/'+k +'.pickle'
            with open(name,'wb') as fw:
                pickle.dump(GAGWO,fw)

            cv_num = 5
            clf = RandomForestClassifier()
            GWOGeneData = GAGWO.transform(geneData)
            geneNameSelected = geneNameFilter[GAGWO.get_support()]
            print(len(geneNameSelected))
            scores = cross_validate(clf, GWOGeneData, label, scoring=scoring, cv=cv_num, return_train_score=False)
            fitness_value = []
            print(len(geneNameSelected))
            for j in range(cv_num):
                fitness_value.append(1 - scores['test_accuracy'] + len(geneNameSelected) / len(geneName))
            print('fitness value:', mean(fitness_value))
            sorted(scores.keys())
            print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
            # scores['fitness'] = fitness_value
            if k == 'gwo':
                with pd.ExcelWriter(excelName,engine="openpyxl") as writer:
                    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name=k)
            else:
                with pd.ExcelWriter(excelName, mode='a',engine="openpyxl") as writer:
                    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name=k)



    writer.close()





