import pandas as pd
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_validate
from numpy import mean

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




if __name__ == "__main__":

    dataPath = 'voom_normalization_log2.csv'
    gse15222 = 'gse15222.csv'
    gse1297 = 'GSE1297.csv'
    gse199225 = 'GSE199225.csv'
    gse203206 = 'gse203206.csv'
    # geneName,geneData,label = ReadData(gse15222)
    # geneName, geneData, label = ReadData(dataPath)
    # geneName, geneData, label = ReadData(gse1297)
    geneName, geneData, label = ReadData(gse203206)
    # geneDataTrain = geneData
    # labelTrain = label
    geneDataTrain, geneDataTest, labelTrain, labelTest = train_test_split(geneData, label, shuffle=True, train_size=0.8)
    geneDataTrain = np.array(geneDataTrain)
    geneData = np.array(geneData)
    X = geneDataTrain
    y = labelTrain

    from sklearn import linear_model

    # Create an instance of the classifier
    # classifier = linear_model.LogisticRegression(solver= 'liblinear')
    classifier = RandomForestClassifier()


    # Define objective function
    def f_per_particle(m, alpha):
        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = X.shape[0]
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:, m == 1]
        # Perform classification and store performance in P
        classifier.fit(X_subset, y)
        P = (classifier.predict(X_subset) == y).mean()
        # Compute for the objective function
        j = (alpha * (1.0 - P)
             + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        return j


    def f(x, alpha=0.88):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)


    # Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

    # Call instance of PSO
    dimensions = X.shape[1]  # dimensions should be the number of features

    optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=20)

    # Create two instances of LogisticRegression
    classfier = RandomForestClassifier()

    # Get the selected features from the final positions
    X_selected_features = geneData[:, pos == 1]  # subset
    print(len(X_selected_features))
    # Perform classification and store performance in P
    classifier.fit(X_selected_features, label)

    # Compute performance
    clf = RandomForestClassifier()
    # subset_performance = classifier.score(x_test_features,labelTest)
    scores = cross_val_score(clf, X_selected_features, label, cv=3)
    geneNameSelected = geneName[pos==1]
    fitness_value = []
    for i in range(len(scores)):
        fitness_value.append(1 - scores[i] + len(geneNameSelected) / len(geneName))
    print('fitness value:', mean(fitness_value))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    scores = cross_validate(clf, X_selected_features, label, scoring=scoring, cv=3)
    sorted(scores.keys())
    scores['fitness'] = fitness_value
    print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）


    excelName = 'BPSO.xlsx'
    writer = pd.ExcelWriter(excelName)
    pd.DataFrame(geneNameSelected).to_excel(excel_writer=writer, sheet_name='selected_geneName')
    pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='BPSO socres')
    writer.save()
    writer.close()





