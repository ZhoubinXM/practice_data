import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def data_loader():
    train_data = pd.read_csv('./data/train_data.csv', encoding='gbk')
    test_data = pd.read_csv('./data/test.csv', encoding='gbk')
    test_label = pd.read_csv('./data/test_label.csv', encoding='gbk')

    test_data["患有糖尿病标识"] = -1
    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    data['舒张压'] = data['舒张压'].fillna(-1)
    data['出生年份'] = 2022 - data['出生年份']

    def bmi(a):
        if a < 18.5:
            return 0
        elif 18.5 <= a <= 24:
            return 1
        elif 24 < a <= 27:
            return 2
        elif 27 < a <= 32:
            return 3
        else:
            return 4

    data['BMI'] = data['体重指数'].apply(bmi)

    def family_his(a):
        if a == '无记录':
            return 0
        elif a == '叔叔或姑姑有一方患有糖尿病' or a == '叔叔或者姑姑有一方患有糖尿病':
            return 1
        else:
            return 2

    data['糖尿病家族史'] = data['糖尿病家族史'].apply(family_his)

    def pressure(a):
        if a < 60:
            return 0
        elif 60 <= a <= 90:
            return 1
        else:
            return 2

    data['Pressure'] = data['舒张压'].apply(pressure)

    print("Load data done!")

    train = data[data['患有糖尿病标识'] != -1]
    test = data[data['患有糖尿病标识'] == -1]
    train_label = train['患有糖尿病标识']
    train = train.drop(['患有糖尿病标识', '编号'], axis=1)
    test = test.drop(['患有糖尿病标识', '编号'], axis=1)

    return train, train_label, test, test_label


def model(model_name):
    model_list = ['MLP', 'decision_tree']
    if model_name not in model_list:
        raise NotImplementedError("{} is not Implemented in this task.")
    if model_name == 'decision_tree':
        model = DecisionTreeClassifier()
    return model


def submit_result(pred_result, target):
    target['label'] = pred_result
    target.to_csv('./data/submit.csv', index=False)
    print("Generate submit result!")


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader()
    model = model('decision_tree')
    model.fit(train_data, train_label)
    y_pre = model.predict(test_data)
    submit_result(y_pre, test_label)
