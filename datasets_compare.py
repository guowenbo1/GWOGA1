import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import cross_validate
from numpy import mean
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile,f_classif
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import numpy as np
import pyswarms as ps
import SA


def f_per_particle(X,m, alpha,y):

    total_features = X.shape[0]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X.loc[:, m == 1]
    # Perform classification and store performance in P
    classifier = RandomForestClassifier()
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


def f(x,data,label):
    alpha = 0.88
    n_particles = x.shape[0]
    j = [f_per_particle(data,x[i], alpha,label) for i in range(n_particles)]
    return np.array(j)

if __name__ == "__main__":
    path = ['dataset/gas.csv','dataset/Gastrointestinal.csv','dataset/rejafada.csv','dataset/setapProcessT3.csv','dataset/setapProcessT2.csv','dataset/setapProcessT1.csv'
            ,'dataset/breast-cancer.csv','dataset/heart-disease.csv','dataset/nba.csv','dataset/codon.csv',
            'dataset/abalone.csv','dataset/NewsPopularity.csv','dataset/Surgical.csv']
    name = ['gas','gastrointestinal','rejafada','setaT3','setaT2','setaT1','breast-cancer','heart-disease','nba','codon','abalone','news','surgical']
    # path = ['dataset/Surgical.csv']
    # name = ['surgical']
    for i, j in zip(path,name):
        result_csv = 'dataset/result/' + j + '.xlsx'
        result_model = 'dataset/model/' + j + '.pickle'
        data = pd.read_csv(i)
        label = data.iloc[:,-1]
        data = data.iloc[:,:-1]
        data_train,data_test,label_train,label_test = train_test_split(
            data,label,shuffle=True,train_size=0.8)

        clf = RandomForestClassifier()
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1']
        #
        #gagwo train
        GA_model = RandomForestClassifier()
        GAGWO = GeneticSelectionCV(estimator=GA_model, n_jobs=-1, cv=5, scoring=None, fit_params=None, max_features=int(data.shape[1]/10) + 1,
                                   n_population=200, crossover_proba=0.3, mutation_proba=0.2, n_generations=20,
                                   tournament_size=5, n_gen_no_change=None, cro='gwo'
                                   )
        GAGWO = GAGWO.fit(data_train, label_train)
        # # 存
        with open(result_model,'wb') as fw:
            pickle.dump(GAGWO,fw)
        # # #读
        # with open(result_model,'rb') as fr:
        #     GAGWO = pickle.load(fr)
        #
        cv_num = 5
        # #GWO
        print('使用gwo的准确度-----------------')
        GWOGeneData = GAGWO.transform(data)
        print(GWOGeneData.shape[1])
        scores = cross_validate(clf, GWOGeneData, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + GWOGeneData.shape[1] / data.shape[1])
        print('fitness value:', fitness_value)
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='gwo')

        print('使用方差选择法的准确度-----------------')
        Varian_filter = VarianceThreshold(threshold=1)
        Varian_filter.fit(data_train)
        Varian_filter_data = Varian_filter.transform(data)
        scores = cross_validate(clf, Varian_filter_data, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + Varian_filter_data.shape[1] / data.shape[1])
        print('fitness value:', mean(fitness_value))
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, mode='a', engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='Variance Threshold')

        print('使用k-best的准确度-----------------')
        kbest_Model = SelectPercentile(score_func=f_classif,percentile=20)
        kbest_Model.fit(data_train,label_train)
        kbest_data = kbest_Model.transform(data)
        scores = cross_validate(clf, kbest_data, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + kbest_data.shape[1] / data.shape[1])
        print('fitness value:', mean(fitness_value))
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, mode='a', engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='Select K Best')

        print('使用rfe的准确度-----------------')
        est = SVC(kernel='linear')
        rfe_model = RFE(est)
        rfe_model = rfe_model.fit(data_train,label_train)
        rfe_data = rfe_model.transform(data)
        scores = cross_validate(clf, rfe_data, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + rfe_data.shape[1] / data.shape[1])
        print('fitness value:', mean(fitness_value))
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, mode='a', engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='RFE')

        print('使用bpso的准确度-----------------')
        # Initialize swarm, arbitrary
        options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}
        dimensions = data.shape[1]  # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
        cost, pos = optimizer.optimize(f, iters=20,data = data_train,label = label_train)
        bpso_data = data.loc[:, pos == 1]  # subset
        scores = cross_validate(clf, bpso_data, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + bpso_data.shape[1] / data.shape[1])
        print('fitness value:', mean(fitness_value))
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, mode='a', engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='BPSO')

        print('使用SA的准确度-----------------')

        feature_size = data.shape[1]
        sel_feature = int(feature_size/20)
        # print(sel_feature)
        if sel_feature == 0:
            sel_feature = 1
        estimator = RandomForestClassifier()
        sa = SA.SimulatedAnnealing(initT=100,
                                minT=20,
                                alpha=0.95,
                                iteration=50,
                                features=feature_size,
                                init_features_sel=sel_feature,
                                estimator=estimator)

        X_selected_features,_,selected_gene = sa.fit(data_train, data_test, label_train, label_test)
        sa_data = pd.RangeIndex(stop=len(data.columns)).isin(X_selected_features)
        sa_data = data.iloc[:,sa_data]
        scores = cross_validate(clf, sa_data, label, scoring=scoring, cv=cv_num, return_train_score=False)
        fitness_value = []
        for j in range(cv_num):
            fitness_value.append(1 - scores['test_accuracy'] + sa_data.shape[1] / data.shape[1])
        print('fitness value:', mean(fitness_value))
        scores['fitness'] = fitness_value[0]
        sorted(scores.keys())
        print('测试结果：', scores)
        with pd.ExcelWriter(result_csv, mode='a', engine="openpyxl") as writer:
            pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='SA')

