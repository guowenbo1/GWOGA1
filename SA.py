import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate
from numpy import mean

class SimulatedAnnealing(object):
    """Feature selection with simulated annealing algorithm.
    parameters
    ----------
    initT: int or float, default: 100
        The maximum temperature
    minT: int or float, default: 1
        The minimum temperature
    alpha：float， default:0.98
        Decay coefficient of temperature
    iteration: int, default:50
        Balance times at present temperature
    features: int
        The number of attributes in the original data
    init_features_sel: int
        The index of selected fatures
    estimator: object
        A supervised learning estimator with a `fit` method.
    Attributes
    ----------
    temp_history: array
        record the temperatures
    best_cost_history: array
        record the MSEs
    best_solution_history: array
        record the solutions
    """

    def __init__(self, features, init_features_sel, estimator, initT=100, minT=1, alpha=0.98, iteration=50):

        self.initT = initT
        self.minT = minT
        self.alpha = alpha
        self.iteration = iteration
        self.feature_size = features
        self.init_feature_sel = init_features_sel
        self.estimator = estimator

    def get_initial_solution(self):
        sol = np.arange(self.feature_size - 1)
        # print('sol',sol)
        # print('feature_sel',self.init_feature_sel)
        np.random.shuffle(sol)
        return sol[:self.init_feature_sel]

    def get_cost(self, solution, x_train, x_test, y_train, y_test):
        """ compute the evaluated results of current solution
        :param solution: array of shape (selected, )
        :param x_train: array of shape (n_samples, n_features)
        :param x_test: array of shape (n_samples, n_features)
        :param y_train: array of shape (n_samples, )
        :param y_test: array of shape (n_samples, n_features)
        :return: mse
        """
        # print('solution is',solution)
        limited_train_data = self.get_data_subset(x_train, solution)
        limited_test_data = self.get_data_subset(x_test, solution)
        estimator = self.estimator.fit(limited_train_data, y_train)
        y_test_pred = estimator.predict(limited_test_data)
        return round(mean_squared_error(y_test, y_test_pred), 4)

    @staticmethod
    def get_data_subset(x_data, soln):
        return x_data.iloc[:, soln]

    def get_neighbor(self, current_solution, temperature):
        """
        :param current_solution: array of shape (selected, )
        :param temperature: int or float.
        :return: selected ：the index of selected features, array of shape (selected, ).
        """
        all_features = range(self.feature_size - 1)
        selected = current_solution
        not_selected = np.setdiff1d(all_features, selected)

        # swap one selected feature with one non-selected feature
        num_swaps = int(
            min(np.ceil(np.abs(np.random.normal(0, 0.1 * len(selected) * temperature))), np.ceil(0.1 * len(selected))))
        feature_out = np.random.randint(0, len(selected), num_swaps)  # 产生num_swaps个样本索引（从range(len(selected))中）
        selected = np.delete(selected, feature_out)
        feature_in = np.random.randint(0, len(not_selected), num_swaps)  # 产生num_swaps个样本索引（从range(len(not_selected))中）
        selected = np.append(selected, not_selected[feature_in])
        return selected

    @staticmethod
    def get_probability(temperature, delta_cost):
        return np.exp(delta_cost / temperature)

    def fit(self, x_train, x_test, y_train, y_test):
        """
        :param x_train: array of shape (n_samples, n_features)
        :param x_test: array of shape (n_samples, n_features)
        :param y_train: array of shape (n_samples, )
        :param y_test: array of shape (n_samples, )
        :return:
        best_solution: the index of final selected attributes, array of shape (selected, )
        best_cost : minimum mse
        """
        temperature = self.initT  # 当前温度
        solution = self.get_initial_solution()
        # print('initsoultoin',solution)
        cost = self.get_cost(solution, x_train, x_test, y_train, y_test)

        temp_history = [temperature]
        best_cost_history = []
        best_solution_history = []

        best_cost = cost
        best_solution = solution

        while temperature > self.minT:
            for k in range(self.iteration):
                next_solution = self.get_neighbor(solution, temperature)
                next_cost = self.get_cost(next_solution, x_train, x_test, y_train, y_test)

                probability = 0
                if next_cost > cost:  # 计算向差方向移动的概率 (即移动后的解比当前解要差)
                    probability = self.get_probability(temperature, cost - next_cost)
                if next_cost < cost or np.random.random() < probability:  # 朝着最优解移动或以一定概率向差方向移动
                    cost = next_cost
                    solution = next_solution
                if next_cost < best_cost:  # 最优值和最优解
                    best_cost = cost
                    best_solution = solution

            print("当前温度：", round(temperature, 2))
            print("当前温度下最好的得分：", best_cost)
            print("当前温度下波长数量：", len(solution))

            temperature *= self.alpha
            temp_history.append(temperature)
            best_cost_history.append(best_cost)
            best_solution_history.append(best_solution)

        self.temp_history_ = temp_history
        self.best_cost_history_ = best_cost_history
        self.best_solution_history = best_solution_history
        return best_solution, best_cost,solution

def ReadData_sel(filePath,select_gene):
    data = pd.read_csv(filePath,header=0)
    selected_gene = data.index.isin(select_gene)
    data.index = data['gene_name'].values
    data = data.drop(labels='gene_name',axis=1)
    data = data.iloc[selected_gene,:]
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

def ReadData(filePath):
    data = pd.read_csv(filePath,header=0)
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

# dataPath = 'voom_normalization_log2.csv'
# gse15222 = 'gse15222.csv'
# gse1297 = 'GSE1297.csv'
# gse199225 = 'GSE199225.csv'
# gse203206 = 'gse203206.csv'
# # geneName, geneData, label = ReadData(dataPath)
# # geneName,geneData,label = ReadData(gse15222)
# # geneName, geneData, label = ReadData(gse1297)
# # geneName, geneData, label = ReadData(gse199225)
# geneName, geneData, label = ReadData(gse203206)
#
# geneDataTrain,geneDataTest,labelTrain,labelTest = train_test_split(geneData,label,shuffle=True,train_size=0.8)
# geneDataTrain = np.array(geneDataTrain)
# geneDataTest = np.array(geneDataTest)
# labelTrain = np.array(labelTrain)
# labelTest = np.array(labelTest)
#
# # 3.使用模拟退火算法优化特征数量
# feature_size = geneDataTrain.shape[1]
# sel_feature = 3000
# # estimator = MLPRegressor(hidden_layer_sizes=43)
# estimator = RandomForestClassifier()
# sa = SimulatedAnnealing(initT=100,
#                         minT=10,
#                         alpha=0.95,
#                         iteration=100,
#                         features=feature_size,
#                         init_features_sel=sel_feature,
#                         estimator=estimator)
# X_selected_features,_,selected_gene = sa.fit(geneDataTrain, geneDataTest, labelTrain, labelTest)
# print(len(X_selected_features))
# print(len(selected_gene))
# # geneName,geneData,label = ReadData_sel(gse15222,X_selected_features)
# geneName,geneData,label = ReadData_sel(gse203206,X_selected_features)
# # Compute performance
# clf = RandomForestClassifier()
# # subset_performance = classifier.score(x_test_features,labelTest)
# scores = cross_val_score(clf, geneData, label, cv=3)
# fitness_value = []
# for i in range(len(scores)):
#     fitness_value.append(1 - scores[i] + len(X_selected_features) / len(geneName))
# print('fitness value:', mean(fitness_value))
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
# scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
# scores = cross_validate(clf, geneData, label, scoring=scoring, cv=3, return_train_score=True)
# sorted(scores.keys())
# scores['fitness'] = fitness_value
# print('测试结果：', scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
#
# excelName = 'simulated_annealing.xlsx'
# writer = pd.ExcelWriter(excelName)
# pd.DataFrame(X_selected_features).to_excel(excel_writer=writer, sheet_name='selected_geneName')
# pd.DataFrame(scores).to_excel(excel_writer=writer, sheet_name='BPSO socres')
# writer.save()
# writer.close()