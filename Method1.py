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
    # same_gene = same_gene.same_gene()
    # geneName, geneData, label = ReadData(dataPath)
    geneName,geneData,label = ReadData(gse15222)
    # geneName, geneData, label = ReadData(gse1297)
    # geneName, geneData, label = ReadData(gse199225)
    # geneName, geneData, label = ReadData(gse203206)
    geneDataTrain,geneDataTest,labelTrain,labelTest = train_test_split(
        geneData,label,shuffle=True,train_size=0.8)
    # print('geneName len is',len(geneName))
    # print('geneData is \n',geneData)
    # print('label',label)
    flag10 = round(geneData.shape[1] * 0.1)
    flag20= round(geneData.shape[1] * 0.2)
    flag30 = round(geneData.shape[1] * 0.3)
    flag40 = round(geneData.shape[1] * 0.4)
    flag50 = round(geneData.shape[1] * 0.5)



    clf = RandomForestClassifier()
    print('未进行特征选择的准确读：-----------------------------')
    scores = cross_val_score(clf, geneData, label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, geneData, label, scoring=scoring, cv=5, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）

    geneNameFilter = geneName
    geneDataFilterTrain = geneDataTrain
    geneDataFilterTest = geneDataTest


    #
    # GAFS
    GA_model = RandomForestClassifier()
    op = ['cxuniform','cxonepoint','cxtwopoint','cxpartialymatched','cxuniformpartialymatched','cxordered']
    for i in op:
        GAGWO = GeneticSelectionCV(estimator=GA_model,n_jobs=-1,cv=5, scoring=None, fit_params=None,max_features=5,
                 n_population=200, crossover_proba=0.3, mutation_proba=0.2, n_generations=20, tournament_size=5, n_gen_no_change=None,cro=i
                 )
        GAGWO = GAGWO.fit(geneData, label)
        print('GAFS has been done')
    # 存
    #     name = 'Genetic Feature Selection Base Random Forest ' + i +'.pickle'
    #     with open(name,'wb') as fw:
    #         pickle.dump(GAGWO,fw)
    # # #读
    with open('Genetic Feature Selection Base Random Forest.pickle','rb') as fr:
        GAGWO = pickle.load(fr)

    geneDataSelectedTrain = GAGWO.transform(geneDataFilterTrain)
    geneDataSelectedTest = GAGWO.transform(geneDataFilterTest)
    geneNameSelected = geneNameFilter[GAGWO.get_support()]

    # save to excel
    excelName = 'Genetic Feature Selection.xlsx'
    writer = pd.ExcelWriter(excelName)
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer,sheet_name='Selected Gene Name')
    # pd.DataFrame(geneDataTrain).to_excel(excel_writer=writer, sheet_name='Training Data')
    # pd.DataFrame(geneDataTest).to_excel(excel_writer=writer, sheet_name='Testing Data')
    # pd.DataFrame(labelTrain).to_excel(excel_writer=writer, sheet_name='Training Label')
    # pd.DataFrame(labelTest).to_excel(excel_writer=writer, sheet_name='Test Label')

    writer.save()
    writer.close()

    #model validation
    excelName = 'model validation.xlsx'
    writer = pd.ExcelWriter(excelName)
    cv_num = 5
    # clf = SVC(kernel='rbf')
    clf = RandomForestClassifier()
    print('未进行特征选择的准确读：-----------------------------')
    scores = cross_val_score(clf, geneData, label, cv=cv_num)
    fitness_value = 2 - scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, geneData, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer,sheet_name='None feature selection')

    print('使用GWO算法的准确度：---------------------------------')
    GWOGeneData = GAGWO.transform(geneData)
    scores = cross_val_score(clf, GWOGeneData, label, cv=cv_num)
    geneNameSelected = geneNameFilter[GAGWO.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] +  len(geneNameSelected) / len(geneName))
    print('fitness value:',mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, GWOGeneData, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='GWO feature selection')

    print('使用方差选择法的准确度-----------------')
    Varian_filter = VarianceThreshold(threshold=1)
    Varian_filter.fit(geneDataTrain)
    Varian_filter_data = Varian_filter.transform(geneData)
    scores = cross_val_score(clf, Varian_filter_data, label, cv=cv_num)
    geneNameSelected = geneNameFilter[Varian_filter.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] +  len(geneNameSelected) / len(geneName))
    print('fitness value:',mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, Varian_filter_data, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='variance feature selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='variance selected gene')


    print('使用k-best的准确度-----------------')
    kbest_Model = SelectPercentile(score_func=f_classif,percentile=30)
    kbest_Model.fit(geneDataTrain,labelTrain)
    kbest_data = kbest_Model.transform(geneData)
    scores = cross_val_score(clf, kbest_data, label, cv=cv_num)
    geneNameSelected = geneNameFilter[kbest_Model.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] +  len(geneNameSelected) / len(geneName))
    print('fitness value:',mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, kbest_data, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='kbest feature selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='kbest selected gene')

    # L1-based feature selection
    print('使用L1-based feature selection的准确度-----------------')
    lsvc = LinearSVC(C=0.3, penalty="l2").fit(geneDataTrain, labelTrain)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(geneData)
    scores = cross_val_score(clf, X_new, label, cv=cv_num)
    geneNameSelected = geneNameFilter[model.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] + len(geneNameSelected) / len(geneName))
    print('fitness value:', mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, X_new, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='L1-based feature selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='L1-based selected gene')

    print('使用tree-model feature selection的准确度-----------------')
    # Tree-based feature selection
    tree_model = ExtraTreesClassifier(n_estimators=200)
    tree_model = tree_model.fit(geneDataTrain, labelTrain)
    tree_model = SelectFromModel(tree_model, prefit=True)
    tree_data = tree_model.transform(geneData)
    scores = cross_val_score(clf, tree_data, label, cv=cv_num)
    geneNameSelected = geneNameFilter[tree_model.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] + len(geneNameSelected) / len(geneName))
    print('fitness value:', mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, tree_data, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='tree model feature selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='tree selcted gene')

    #logistic
    print('使用logistic-model feature selection的准确度-----------------')
    log_model = SelectFromModel(estimator=LogisticRegression()).fit(geneDataTrain, labelTrain)
    log_data = log_model.transform(geneData)
    scores = cross_val_score(clf, log_data, label, cv=cv_num)
    geneNameSelected = geneNameFilter[log_model.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] + len(geneNameSelected) / len(geneName))
    print('fitness value:', mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, log_data, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='logistic selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='logistic selected gene')


    print('使用rfe准确度-----------------------')
    est = SVC(kernel='linear')
    rfe_model = RFE(est)
    rfe_model = rfe_model.fit(geneDataTrain,labelTrain)
    rfe_data = rfe_model.transform(geneData)
    scores = cross_val_score(clf, rfe_data, label, cv=cv_num)
    geneNameSelected = geneNameFilter[rfe_model.get_support()]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] +  len(geneNameSelected) / len(geneName))
    print('fitness value:',mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, rfe_data, label, scoring=scoring, cv=cv_num, return_train_score=True)
    sorted(scores.keys())
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
    scores['fitness'] = fitness_value
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='rfe feature selection')
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='rfe selected gene')

    writer.save()
    writer.close()

