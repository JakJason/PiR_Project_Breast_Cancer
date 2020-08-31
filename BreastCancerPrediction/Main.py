import pandas as pd
import sklearn
import sklearn.neighbors
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
import sklearn.ensemble


# Analiza przypadkow zawalow serca.
# Weryfikacja jakie czynniki zwiekszaja/zmniejszaja prawdopodobienstwo zgonu.
# Dane pobrane z serwisu kaggle.com

# 3. PRZYGOTOWANIE DANYCH


def data_preparation():
    # przygotowanie dwoch zestawow danych
    # zestaw 1 ze wszystkimi parametrami
    # zestaw 2 ze zredukowanymi parametrami
    data = pd.read_csv("DataSet/Breast_cancer_data.csv")
    print(data.info())
    reduced_data = data.copy(deep=True)
    reduced_data = reduced_data.drop(labels=['mean_perimeter', 'mean_area'], axis='columns')
    print(reduced_data.info())
    return data, reduced_data


# 4. ANALIZA DANYCH


def data_split(dataset, size):
    state = 123
    target = dataset['diagnosis']
    feature = dataset.drop(labels='diagnosis',axis='columns')

    # Podzial na zbior uczacy i testowy
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(feature, target,
                                                                                test_size=size,
                                                                                random_state=123)
    return X_train, X_test, y_train, y_test


def data_normalize(train, test, feature):

    # normalizacja wektorow
    # L1
    X_norm_l1_train = sklearn.preprocessing.normalize(train, norm="l1")
    X_norm_l1_test = sklearn.preprocessing.normalize(test, norm="l1")
    X_norm_l1 = sklearn.preprocessing.normalize(feature, norm="l1")

    # L2
    X_norm_l2_train = sklearn.preprocessing.normalize(train, norm="l2")
    X_norm_l2_test = sklearn.preprocessing.normalize(test, norm="l2")
    X_norm_l2 = sklearn.preprocessing.normalize(feature, norm="l2")

    return X_norm_l1_train, X_norm_l1_test, X_norm_l1, X_norm_l2_train, X_norm_l2_test, X_norm_l2


# 5. MODELOWANIE DANYCH


def data_model(x_train, y_train, x_test, y_test, n):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    y_pred = knn.fit(x_train, y_train)


def fit_classifier(alg, X_ucz, X_test, y_ucz, y_test):
    alg.fit(X_ucz, y_ucz)
    y_pred_ucz = alg.predict(X_ucz)
    y_pred_test = alg.predict(X_test)
    return {
        "ACC_ucz": sklearn.metrics.accuracy_score(y_pred_ucz, y_ucz),
        "ACC_test": sklearn.metrics.accuracy_score(y_pred_test, y_test),
        "P_ucz":   sklearn.metrics.precision_score(y_pred_ucz, y_ucz),
        "P_test":   sklearn.metrics.precision_score(y_pred_test, y_test),
        "R_ucz":   sklearn.metrics.recall_score(y_pred_ucz, y_ucz),
        "R_test":   sklearn.metrics.recall_score(y_pred_test, y_test),
        "F1_ucz":  sklearn.metrics.f1_score(y_pred_ucz, y_ucz),
        "F1_test":  sklearn.metrics.f1_score(y_pred_test, y_test)
    }


def model_optymalization(x, y):
    search_grid = [
        {
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'n_neighbors': [5, 10, 15, 20]
        }
    ]

    scorer = {'acc': 'accuracy', 'f1': 'f1', 'prec': 'precision', 'rec': 'recall'}

    search_func = GridSearchCV(estimator=sklearn.neighbors.KNeighborsClassifier(),
                                param_grid=search_grid,
                                scoring=scorer,
                                n_jobs=-1, iid=False, refit='acc', cv=5)
    search_func.fit(x, y)
    print(search_func.best_estimator_)
    print(search_func.best_params_)
    print(search_func.best_score_)
    results = pd.DataFrame(search_func.cv_results_)

    return results




if __name__ == "__main__":
    data, r_data = data_preparation()
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    # count = data.diagnosis.value_counts()
    # count.plot(kind='bar')
    # plt.title("Rozklad nowotworow zlosliwych i lagodnych")
    # plt.xlabel("Diagnoza")
    # plt.ylabel("Ilosc przypadkow");
    # data['Diagnoza'] = data['diagnosis'].map({0: 'lagodny', 1: 'zlosliwy'})  # converting the data into categorical
    # g = sns.pairplot(data.drop('diagnosis', axis=1), hue="Diagnoza", palette='prism')
    # plt.show()

    X_train, X_test, y_train, y_test = data_split(data, 0.2)
    X_r_train, X_r_test, y_r_train, y_r_test = data_split(r_data, 0.2)

    X_norm_l1_train, X_norm_l1_test, X_norm_l1, X_norm_l2_train, X_norm_l2_test, X_norm_l2 \
        = data_normalize(X_train,
                         X_test,
                         data)

    X_r_norm_l1_train, X_r_norm_l1_test, X_r_norm_l1, X_r_norm_l2_train, X_r_norm_l2_test, X_r_norm_l2 \
        = data_normalize(X_r_train,
                         X_r_test,
                         r_data)

    results = pd.DataFrame({'knn_05_full': fit_classifier(knn, X_train, X_test, y_train, y_test)}).T
    results = results.append(pd.DataFrame({'knn_05_red': fit_classifier(knn, X_r_train, X_r_test, y_r_train, y_r_test)}).T)
    results = results.append(pd.DataFrame({'knn_05_full_norm_l1': fit_classifier(knn, X_norm_l1_train, X_norm_l1_test, y_train, y_test)}).T)
    results = results.append(pd.DataFrame({'knn_05_red_norm_l1': fit_classifier(knn, X_r_norm_l1_train, X_r_norm_l1_test, y_r_train, y_r_test)}).T)
    results = results.append(pd.DataFrame({'knn_05_full_norm_l2': fit_classifier(knn, X_norm_l2_train, X_norm_l2_test, y_train, y_test)}).T)
    results = results.append(pd.DataFrame({'knn_05_red_norm_l2': fit_classifier(knn, X_r_norm_l2_train, X_r_norm_l2_test, y_r_train, y_r_test)}).T)

    # results = model_optymalization(X_train, y_train)
    #
    # KNeighborsClassifier(n_neighbors=10, weights='distance')
    # {'n_neighbors': 10, 'p': 2, 'weights': 'distance'}
    # 0.9010989010989011

    # results2 = model_optymalization(X_r_train, y_r_train)
    # KNeighborsClassifier(n_neighbors=15)
    # {'n_neighbors': 15, 'p': 2, 'weights': 'uniform'}
    # 0.9032967032967033

    # results3 = model_optymalization(X_norm_l1_train, y_train)
    # KNeighborsClassifier(n_neighbors=10, p=1)
    # {'n_neighbors': 10, 'p': 1, 'weights': 'uniform'}
    # 0.8703296703296705

    # results4 = model_optymalization(X_norm_l2_train, y_train)
    # KNeighborsClassifier(n_neighbors=15)
    # {'n_neighbors': 15, 'p': 2, 'weights': 'uniform'}
    # 0.8791208791208792

    # results5 = model_optymalization(X_r_norm_l1_train, y_r_train)
    # KNeighborsClassifier(n_neighbors=20, p=1, weights='distance')
    # {'n_neighbors': 20, 'p': 1, 'weights': 'distance'}
    # 0.6637362637362638

    # results6 = model_optymalization(X_r_norm_l2_train, y_r_train)
    # KNeighborsClassifier(n_neighbors=20, weights='distance')
    # {'n_neighbors': 20, 'p': 2, 'weights': 'distance'}
    # "removed in 0.24.", FutureWarning
    # 0.6571428571428573

    results = results.append(pd.DataFrame({'knn_10_2_dist':
        fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance'),
                       X_train,
                       X_test,
                       y_train,
                       y_test)}).T)

    results = results.append(pd.DataFrame({'knn_15_red':
         fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=15),
                        X_r_train,
                        X_r_test,
                        y_r_train,
                        y_r_test)}).T)

    results = results.append(pd.DataFrame({'knn_10_1_norm_l1':
        fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, p=1),
                       X_norm_l1_train,
                       X_norm_l1_test,
                       y_train,
                       y_test)}).T)

    results = results.append(pd.DataFrame({'knn_15_norm_l2':
        fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=15),
                       X_norm_l2_train,
                       X_norm_l2_test,
                       y_train,
                       y_test)}).T)

    results = results.append(pd.DataFrame({'knn_20_1_dist_red_norm_l1':
         fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=20, p=1, weights='distance'),
                        X_r_norm_l1_train,
                        X_r_norm_l1_test,
                        y_r_train,
                        y_r_test)}).T)

    results = results.append(pd.DataFrame({'knn_20_dist_red_norm_l2':
          fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=20, weights='distance'),
                         X_r_norm_l2_train,
                         X_r_norm_l2_test,
                         y_r_train,
                         y_r_test)}).T)

    results.to_csv("results.csv")

    forest = sklearn.ensemble.RandomForestClassifier(random_state=4021)
    forest.fit(X_train, y_train)
    pd.Series(forest.feature_importances_,
              index=X_train.columns[0:5]).sort_values(ascending=False).to_csv("relevance.csv")
    forest.fit(X_r_train, y_r_train)
    pd.Series(forest.feature_importances_,
              index=X_r_train.columns[0:3]).sort_values(ascending=False).to_csv("relevance_red.csv")