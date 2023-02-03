# sklearn-genetic - Genetic feature selection module for scikit-learn
# Copyright (C) 2016-2021  Manuel Calzolari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Genetic algorithm for feature selection"""

import multiprocessing
import numbers
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
try:
    from sklearn.feature_selection import SelectorMixin  # scikit-learn>=0.23.0
except ImportError:
    try:
        from sklearn.feature_selection._base import SelectorMixin  # scikit-learn==0.22.*
    except ImportError:
        from sklearn.feature_selection.base import SelectorMixin  # scikit-learn<0.22.0
from sklearn.utils._joblib import cpu_count
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random


creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)


def _eaFunction(population, toolbox, cxpb, mutpb, ngen,flag, ngen_no_change=None, stats=None,
                halloffame=None, verbose=0):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # print('invalid_ind[0] is',invalid_ind[0]) #ind[0]是第0个个体

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)


    # print('fitnesses is',fitnesses)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        # print('ind.fitnvess.values is',ind.fitness.values)

    if halloffame is None:
        raise ValueError("The 'halloffame' parameter should not be None.")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    wait = 0
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print('offspring', offspring[0])
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb,flag)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fitnesses = toolbox.map(toolbox.evaluate, offspring)

        sum = 0
        max_fit = 0.0
        for i in fitnesses:
            sum += i[0]
            if i[0] > max_fit:
                max_fit = i[0]
        print('mean fitvale', sum / len(fitnesses))
        print('best fitvalue', max_fit)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best back to population
        offspring.extend(halloffame.items)

        # Get the previous best individual before updating the hall of fame
        prev_best = halloffame[0]

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # If the new best individual is the same as the previous best individual,
        # increment a counter, otherwise reset the counter
        if halloffame[0] == prev_best:
            wait += 1
        else:
            wait = 0

        # If the counter reached the termination criteria, stop the optimization
        if ngen_no_change is not None and wait >= ngen_no_change:
            break

    return population, logbook


def _createIndividual(icls, n, max_features):
    n_features = np.random.randint(1, max_features + 1)
    genome = ([1] * n_features) + ([0] * (n - n_features))
    np.random.shuffle(genome)
    return icls(genome)


def _evalFunction(individual, estimator, X, y, cv, scorer, fit_params, max_features,
                  caching, scores_cache={}):
    individual_sum = np.sum(individual, axis=0)
    if individual_sum == 0 or individual_sum > max_features:
        return -10000, individual_sum
    individual_tuple = tuple(individual)
    if caching and individual_tuple in scores_cache:
        return scores_cache[individual_tuple], individual_sum
    X_selected = X[:, np.array(individual, dtype=np.bool)]
    scores = cross_val_score(estimator=estimator, X=X_selected, y=y, scoring=scorer, cv=cv,
                             fit_params=fit_params)
    scores_mean = np.mean(scores)
    if caching:
        scores_cache[individual_tuple] = scores_mean

    return scores_mean, individual_sum



class GeneticSelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    max_features : int or None, optional
        The maximum number of features selected.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    n_population : int, default=300
        Number of population for the genetic algorithm.

    crossover_proba : float, default=0.5
        Probability of crossover for the genetic algorithm.

    mutation_proba : float, default=0.2
        Probability of mutation for the genetic algorithm.

    n_generations : int, default=40
        Number of generations for the genetic algorithm.

    crossover_independent_proba : float, default=0.1
        Independent probability for each attribute to be exchanged, for the genetic algorithm.

    mutation_independent_proba : float, default=0.05
        Independent probability for each attribute to be mutated, for the genetic algorithm.

    tournament_size : int, default=3
        Tournament size for the genetic algorithm.

    n_gen_no_change : int, default None
        If set to a number, it will terminate optimization when best individual is not
        changing in all of the previous ``n_gen_no_change`` number of generations.

    caching : boolean, default=False
        If True, scores of the genetic algorithm are cached.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    generation_scores_ : array of shape [n_generations]
        The maximum cross-validation score for each generation.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    An example showing genetic feature selection.

    >>> import numpy as np
    >>> from sklearn import datasets, linear_model
    >>> from genetic_selection import GeneticSelectionCV
    >>> iris = datasets.load_iris()
    >>> E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    >>> X = np.hstack((iris.data, E))
    >>> y = iris.target
    >>> estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    >>> selector = GeneticSelectionCV(estimator, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True  True  True  True False False False False False False False False
           False False False False False False False False False False False False], dtype=bool)
    """
    def __init__(self, estimator, cv=None, scoring=None, fit_params=None, max_features=None,
                 verbose=0, n_jobs=1, n_population=300, crossover_proba=0.5, mutation_proba=0.2,
                 n_generations=40, crossover_independent_proba=0.1,
                 mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=None,
                 caching=False,cro = None):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
        self.max_features = max_features
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.n_gen_no_change = n_gen_no_change
        self.caching = caching
        self.scores_cache = {}
        self.cro = cro

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the GeneticSelectionCV model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between 1 and {} features."
                                " Got {!r} instead."
                                .format(n_features, self.max_features))
            elif self.max_features < 1 or self.max_features > n_features:
                raise ValueError("'max_features' should be between 1 and {} features."
                                 " Got {} instead."
                                 .format(n_features, self.max_features))
            max_features = self.max_features
        else:
            max_features = n_features

        if not isinstance(self.n_gen_no_change, (numbers.Integral, np.integer, type(None))):
            raise ValueError("'n_gen_no_change' should either be None or an integer."
                             " {} was passed."
                             .format(self.n_gen_no_change))

        estimator = clone(self.estimator)

        # Genetic Algorithm
        toolbox = base.Toolbox()

        toolbox.register("individual", _createIndividual, creator.Individual, n=n_features,
                         max_features=max_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #print('y is',y)
        #print('len y is',len(y))
        toolbox.register("evaluate", _evalFunction, estimator=estimator, X=X, y=y, cv=cv,
                         scorer=scorer, fit_params=self.fit_params, max_features=max_features,
                         caching=self.caching, scores_cache=self.scores_cache)
        #fault
        if self.cro =='cxuniform':
            toolbox.register("mate", tools.cxUniform, indpb=self.crossover_independent_proba)
        elif self.cro == 'cxonepoint':
            toolbox.register("mate", tools.cxOnePoint)
        elif self.cro == 'cxpartialymatched':
            toolbox.register("mate", tools.cxPartialyMatched)
        elif self.cro == 'cxuniformpartialymatched':
            toolbox.register("mate", tools.cxUniformPartialyMatched,indpb=self.crossover_independent_proba)
        elif self.cro == 'cxordered':
            toolbox.register("mate", tools.cxOrdered)
        elif self.cro == 'cxtwopoint':
            toolbox.register("mate", tools.cxTwoPoint)
        elif self.cro == 'gwo':
            toolbox.register('mate', GWOCros, toolbox=toolbox)
        elif self.cro == 'cxBlend':
            toolbox.register("mate", tools.cxBlend)
        elif self.cro == 'cxSimulatedBinary':
            toolbox.register("mate", tools.cxSimulatedBinary)
        elif self.cro == 'cxSimulatedBinaryBounded':
            toolbox.register("mate", tools.cxSimulatedBinaryBounded)
        elif self.cro == 'cxMessyOnePoint':
            toolbox.register("mate", tools.cxMessyOnePoint)
        elif self.cro == 'cxESBlend':
            toolbox.register("mate", tools.cxESBlend)
        elif self.cro == 'cxESTwoPoint':
            toolbox.register("mate", tools.cxESTwoPoint)

        #蜻蜓算法-交叉操作
        # toolbox.register("mate", drogonFlyCros,toolbox=toolbox)
        # 狼群算法-交叉操作
        # t

        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_independent_proba)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs == 0:
            raise ValueError("n_jobs == 0 has no meaning.")
        elif self.n_jobs > 1:
            pool = multiprocessing.Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)
        elif self.n_jobs < 0:
            pool = multiprocessing.Pool(processes=max(cpu_count() + 1 + self.n_jobs, 1))
            toolbox.register("map", pool.map)

        pop = toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if self.verbose > 0:
            print("Selecting features with genetic algorithm.")

        _, log = _eaFunction(pop, toolbox, cxpb=self.crossover_proba, mutpb=self.mutation_proba,
                             ngen=self.n_generations, ngen_no_change=self.n_gen_no_change,
                             stats=stats, halloffame=hof, verbose=self.verbose,flag=self.cro)
        if self.n_jobs != 1:
            pool.close()
            pool.join()

        # Set final attributes
        support_ = np.array(hof, dtype=np.bool)[0]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, support_], y)

        self.generation_scores_ = np.array([score for score, _ in log.select("max")])
        self.n_features_ = support_.sum()
        self.support_ = support_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))


def drogonFlyCros(ind1, ind2,toolbox):
    # offspring = [toolbox.clone(ind) for ind in population]
    enemyDifMax = 0
    foodDifMax = 0
    temp2 = ind2
    temp1 = ind1
    fitValue = toolbox.map(toolbox.evaluate, [temp1])
    print('原始数据fitvalue is',fitValue)
    for i in range(0,len(ind1),5):
        #寻找敌人
        if ind1[i]==0:
            temp1[i]==1
            fitValueChanged = toolbox.map(toolbox.evaluate, [temp1])
            print('第',i,'次改变 fitvalue',fitValueChanged)
            temp1 = ind1
            enemyDif=(abs(fitValueChanged[0][0]-fitValue[0][0]))
            if enemyDif>enemyDifMax:
                 enemyDifMax = enemyDif
                 enemy=i #enemy 是第i个位置
         #寻找食物
        else:
            temp1[i] == 0
            fitValueChanged = toolbox.map(toolbox.evaluate, [temp1])
            foodDif = (abs(fitValueChanged[0][0] - fitValue[0][0]))
            temp2 = ind2
            if foodDif>foodDifMax:
                foodDifMax = foodDif
                food = i #food 位置

    #交叉点
    point = min(food,enemy)
    ind1[point:] = ind2[point:]
    ind2[:point] = ind1[:point]

    return ind1,ind2

def GWOCros(population, toolbox,beta_pb = 0.05):
    # invalid_ind = [ind for ind in population if not ind.fitness.valid] #invalid_ind[0]是第0个个体
    # for i in range(len(population)):
    #     print('population',len(population[i]))
    #     print(i)
    fitvalue = []
    fitnesses = toolbox.map(toolbox.evaluate, population)


    for i in fitnesses:
        fitvalue.append(i[0])


    # print('fitvalue is',fitvalue)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    # print(fitvalue)
    # print('len fitvalue',len(fitvalue))
    # print('population',population[0])
    offspring = [toolbox.clone(ind) for ind in population]
    # print('offspring',offspring)
    beta = []
    alpha,beta1,beta2 = FindList3MaxNum(fitvalue)

    # print(alpha,beta1,beta2)
    alpha_position = fitvalue.index(alpha)
    # print(alpha_position)
    alpha = population[alpha_position]
    del fitvalue[alpha_position]
    # del delt[alpha_position]

    beta1_position = fitvalue.index(beta1)
    beta1 = population[beta1_position]
    # beta.append(population[beta1_position])
    del fitvalue[beta1_position]
    # del delt[beta1_position]
    beta2_position = fitvalue.index(beta2)
    beta2 = population[beta2_position]
    # beta.append(population[beta2_position])
    del fitvalue[beta2_position]
    # del delt[beta2_position]
    # print('alpha position',alpha_position)
    # print('beta1 position',beta1_position)
    a = 2
    x1 = []
    x2 = []
    x3 = []
    A1, A2, A3 = a * (2 * random.random()), a * (2 * random.random()),a * (2 * random.random())
    C1, C2, C3 = 2 * random.random(), 2 * random.random(),2 * random.random()
    e = 2.71

    # # bstep_j = []
    # for j in alpha:
    #     if (1 / (1 + e ** -10 * (A1 * C1 - 0.5))) >= random.uniform(0,2):
    #         bstep = 1
    #     else:
    #         bstep = 0
    #     # bstep_j.append(bstep)
    #     if (j + bstep) >=1:
    #         x1_value = 1
    #     else:
    #         x1_value = 0
    #     x1.append(x1_value)
    # # bstep_k = []
    # for k in beta[0]:
    #     if (1 / (1 + e ** -10 * (A2 * C2 - 0.5))) >= random.uniform(0,2):
    #         bstep = 1
    #     else:
    #         bstep = 0
    #     # bstep_k.append(bstep)
    #     if (k + bstep) >= 1:
    #         x2_value = 1
    #     else:
    #         x2_value = 0
    #     x2.append(x2_value)

    # print('bstep_j',bstep_j)
    # print('bstep_k',bstep_k)
    # print('x1',x1)
    # print('x2',x2)
    #更新当前位置
    # for i in range(len(offspring)):
    #     for j in range(i):
    #         flag = random.random()
    #         # print('delt[i][j]',delt[i][j])
    #         if flag > 0.6:
    #             offspring[i][j] = x1[j]
    #             # print('flag > 0.6', delt[i][j])
    #         elif flag > 0.3:
    #             offspring[i][j] = x2[j]
    # print('gwo: offspring',len(offspring))
    # print('gwo alpha',len(alpha))
    for i in range(len(offspring)): #each individual
        for j in range(len(alpha)): #each position
            # print('i',i)
            # print('j',j)
            bstep1 = bstep2 = bstep3 = 0
            x1 = x2 = x3 = 0
            d1 = abs(C1 * alpha[j] - offspring[i][j])
            d2 = abs(C2 * beta1[j] - offspring[i][j])
            d3 = abs(C3 * beta2[j] - offspring[i][j])
            cstep1 = 1 / (1 + e ** -10 * (A1 * d1 - 0.5))
            cstep2 = 1 / (1 + e ** -10 * (A2 * d2 - 0.5))
            cstep3 = 1 / (1 + e ** -10 * (A3 * d3 - 0.5))
            if cstep1 >= random.uniform(0, 2):
                bstep1 = 1
            if cstep2 >= random.uniform(0, 2):
                bstep2 = 1
            if cstep3 >= random.uniform(0, 2):
                bstep3 = 1
            if offspring[i][j] + bstep1 >= 1:
                x1 = 1
            if offspring[i][j] + bstep2 >= 1:
                x2 = 1
            if offspring[i][j] + bstep3 >= 1:
                x3 = 1
            flag = random.random()
            if flag > 0.6:
                offspring[i][j] = x1
            elif flag > 0.3:
                offspring[i][j] = x2
            else:
                offspring[i][j] = x3
    # print(len(population))
    return offspring


def FindList3MaxNum(ls):  # 快速获取list中最大的三个元素
    max1, max2, max3 = None, None, None
    for num in ls:
        if max1 is None or max1 < num:
            max1, num = num, max1
        if num is None:
            continue
        if max2 is None or num > max2:
            max2, num = num, max2
        if num is None:
            continue
        if max3 is None or num > max3:
            max3 = num
    return max1, max2, max3