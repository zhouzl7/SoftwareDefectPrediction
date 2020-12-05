import scipy.io as scio
import sklearn.svm as sk_svm
import sklearn.neural_network as sk_nn
import sklearn.linear_model as sk_linear
from sklearn.ensemble import RandomForestClassifier
import sklearn.naive_bayes as sk_bayes
import sklearn.decomposition as sk_decomposition
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from GBcls import GBClassification
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import  AdaBoostClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score


from imblearn.over_sampling import SMOTE

import numpy as np
import csv

# name_arr = ['ant1','ivy2','jedit4','lucene2','synapse1','velocity1','xalan2']
name_arr = ['cm1','kc3','mc2','mw1','pc1','pc3','pc4','pc5']

arr_10 = [[],[],[],[],[]]
arr_20 = [[],[],[],[],[]]
arr_30 = [[],[],[],[],[]]
min_max_scaler = preprocessing.MinMaxScaler()
save_arr = []
data_set_arr = []
for name in name_arr:
    train_name = f'{name}train'
    test_name = f'{name}test'

    trian_path = f'NASA/NASATrain/{train_name}.mat'
    test_path = f'NASA/NASATest/{test_name}.mat'
    data = scio.loadmat(trian_path)
    train = []
    for key in data[f'{train_name}']:
        for k in key:
            train.append(k)

    data = scio.loadmat(test_path)
    test = []
    for key in data[f'{test_name}']:
        for k in key:
            test.append(k)


    for i in range(0, 3):
        save_arr.append([])
        data_set_arr.append([])
        X_train = train[i][:, :-1]
        y_train = train[i][:, -1]
        

        if y_train.mean()>1:
            for idx,(key) in enumerate(y_train):
                if key==2:
                    y_train[idx] = -1.0

        X_test = test[i][:, :-1]
        y_test = test[i][:, -1]
        if y_test.mean()>1:
            for idx,(key) in enumerate(y_test):
                if key==2:
                    y_test[idx] = -1.0

        

        smo_1 = SMOTE(random_state=0,k_neighbors=3)
        smo_2 = SMOTE(random_state=0,k_neighbors=1)

        try:
            X_train, y_train =  smo_1.fit_sample(X_train, y_train)
        except ValueError:
            X_train, y_train =  smo_2.fit_sample(X_train, y_train)

        data_set_arr[len(data_set_arr)-1].append(X_train.shape[0])
        data_set_arr[len(data_set_arr)-1].append(sum([int(xi>0) for xi in y_train]))
        data_set_arr[len(data_set_arr)-1].append(sum([int(xi<0) for xi in y_train]))
        data_set_arr[len(data_set_arr)-1].append(X_test.shape[0])
        data_set_arr[len(data_set_arr)-1].append(sum([int(xi>0) for xi in y_test]))
        data_set_arr[len(data_set_arr)-1].append(sum([int(xi<0) for xi in y_test]))

        # pca = sk_decomposition.PCA(n_components=15, whiten=False, svd_solver='auto')

        # model = sk_svm.SVC(C=1.0, kernel='rbf', gamma='auto')

        # model = sk_nn.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
        #                     learning_rate_init=0.001, max_iter=500)

        model = sk_linear.LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20,
                                             fit_intercept=True)

        # model = sk_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)  # 多项式分布的朴素贝叶斯
        # model = sk_bayes.BernoulliNB()  # 伯努利分布的朴素贝叶斯
        # model = sk_bayes.GaussianNB()  # 高斯分布的朴素贝叶斯

        # model = sk_svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        # model = XGBClassifier(n_estimators=500,max_depth=5, min_samples_split=2,subsample=1.0,learning_rate=0.1) # XGBC
        # model = AdaBoostClassifier(n_estimators=50)
        # model = RandomForestClassifier(n_estimators=100)
        # {'learning_rate': 0.7, 'max_depth': 5, 'n_estimators': 3000, 'subsample': 0.4} 0.95
        # model = GradientBoostingClassifier(loss='deviance',n_estimators=3000, max_depth=3, min_samples_split=2,subsample=0.5,learning_rate=0.1)
        # model = GradientBoostingClassifier(n_estimators=500,max_depth=5, min_samples_split=2,subsample=1.0,learning_rate=0.1)


        # model = GBClassification(M=3000, base_learner=DecisionTreeRegressor(max_depth=3, random_state=1),
        #                          learning_rate=0.1,subsample=0.5,
        #                          method="classification", loss="modified_huber")
        # pca.fit(X_train)
        # X_train = pca.transform(X_train)  # reduced_X为降维后的数据
        # X_train = min_max_scaler.fit_transform(X_train)
        # transformer = Normalizer().fit(X_train)
        # X_train = transformer.transform(X_train)
        model.fit(X_train, y_train)

        # tree_param_grid = {'n_estimators': list((3000,4000,5000)), 'max_depth': list((3,4,5)),'subsample': list((0.4, 0.5,0.6)),'learning_rate':list((0.1,0.3,0.5,0.7,0.9))}

        # grid = GridSearchCV(GradientBoostingClassifier(), param_grid=tree_param_grid,verbose=2,n_jobs=5)
        # grid.fit(X_train, y_train)
        # print(grid.grid_scores_)
        # print(grid.best_params_, grid.best_score_)

        # pca.fit(X_test)
        # X_test = pca.transform(X_test)  # reduced_X为降维后的数据
        # transformer = Normalizer().fit(X_test)
        # X_test = transformer.transform(X_test)


        y = model.predict(X_test)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for idx in range(0,len(y_test)):
            if y_test[idx]==1 and y[idx] == 1:
                tp+=1
            elif y_test[idx] == 1 and y[idx] == -1:
                fn += 1
            elif y_test[idx] == -1 and y[idx] == 1:
                fp += 1
            elif y_test[idx] == -1 and y[idx] == -1:
                tn += 1

        print(tp,fp,fn,tn)

        p = tp/(fp+tp)
        pf = fp/(fp+tn)
        pd = tp/(tp+fn)
        f = 2*pd*p/(pd+p)
        auc = roc_auc_score(y_test,y)

        print(f'{name}-{i}:'
              f'pd:',pd,
              f'pf:',pf,
              f'auc:',auc,
              f'p:',p,
              f'F:',f)
        save_arr[len(save_arr)-1].append(pd)
        save_arr[len(save_arr) - 1].append(pf)
        save_arr[len(save_arr) - 1].append(auc)
        save_arr[len(save_arr) - 1].append(p)
        save_arr[len(save_arr) - 1].append(f)

        if i==0:
            arr_10[0].append(pd)
            arr_10[1].append(pf)
            arr_10[2].append(auc)
            arr_10[3].append(p)
            arr_10[4].append(f)
        if i==1:
            arr_20[0].append(pd)
            arr_20[1].append(pf)
            arr_20[2].append(auc)
            arr_20[3].append(p)
            arr_20[4].append(f)
        if i==2:
            arr_30[0].append(pd)
            arr_30[1].append(pf)
            arr_30[2].append(auc)
            arr_30[3].append(p)
            arr_30[4].append(f)

print('10%')
print('pd:',np.mean(arr_10[0]))
print('pf:',np.mean(arr_10[1]))
print('auc:',np.mean(arr_10[2]))
print('p:',np.mean(arr_10[3]))
print('F:',np.mean(arr_10[4]))

print('20%')
print('pd:',np.mean(arr_20[0]))
print('pf:',np.mean(arr_20[1]))
print('auc:',np.mean(arr_20[2]))
print('p:',np.mean(arr_20[3]))
print('F:',np.mean(arr_20[4]))

print('30%')
print('pd:',np.mean(arr_30[0]))
print('pf:',np.mean(arr_30[1]))
print('auc:',np.mean(arr_30[2]))
print('p:',np.mean(arr_30[3]))
print('F:',np.mean(arr_30[4]))


np.savetxt('nasa.csv', np.array(save_arr), delimiter = ',')
# np.savetxt('nasa_data.csv', np.array(data_set_arr), delimiter = ',')
