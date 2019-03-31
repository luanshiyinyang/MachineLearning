# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def get_data():
    train = pd.read_csv('data/train.csv', dtype={'Age': np.float64})
    test = pd.read_csv('data/test.csv', dtype={'Age': np.float64})
    return train, test


def harmonize_data(titanic):
    """
    预处理数据，随机森林不允许非数值、空置等
    :param titanic:
    :return:
    """
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    titanic.loc[titanic['Sex']=='male','Sex']=0
    titanic.loc[titanic['Sex']=='female','Sex']=1
    titanic['Embarked'] = titanic['Embarked'].fillna('S')
    titanic.loc[titanic['Embarked']=='S','Embarked']=0
    titanic.loc[titanic['Embarked']=='C','Embarked']=1
    titanic.loc[titanic['Embarked']=='Q','Embarked']=2
    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
    return titanic


def create_submission(alg, train, test, predictors, filename):
    """
    文件输出，一般竞赛平台要求提交数据集为id+label
    :param alg:
    :param train:
    :param test:
    :param predictors:
    :param filename:
    :return:
    """
    alg.fit(train[predictors], train['Survived'])
    predictions = alg.predict(test[predictors])
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv(filename, index=False)


if __name__ == '__main__':
    train, test = get_data()
    train_data = harmonize_data(train)
    test_data = harmonize_data(test)
    # 确定模型的特征
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # 建模
    alg = RandomForestClassifier(
        random_state=1,
        n_estimators=150,
        min_samples_split=4,
        min_samples_leaf=2
    )
    # 交叉验证
    scores = cross_val_score(
        alg,
        train_data[predictors],
        train_data['Survived'],
        cv=3
    )
    print(scores.mean())
    print(scores.std())
    # 预测结果输出
    create_submission(alg, train_data, test_data, predictors, 'data/result.csv')