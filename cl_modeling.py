"""
This class we can model
"""


class cl_modeling:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

    def train_predict_model(self, model):
        # Train the model
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)

        # import metrics
        from sklearn.metrics import roc_auc_score, confusion_matrix
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.metrics import precision_score, recall_score

        print("Precision: ", precision_score(self.y_test, y_pred))
        print("Recall: ", recall_score(self.y_test, y_pred))
        print("Accuracy: ", accuracy_score(self.y_test, y_pred))
        print("F1: ", f1_score(self.y_test, y_pred))
        print("AUC: ", roc_auc_score(self.y_test, y_pred_proba[:, 1]))
        print("Confusion Matrix: \n ", confusion_matrix(self.y_test, y_pred))

    def multi_default_models(self, models=None):
        if models:
            ob2 = cl_modeling(self.X_train, self.X_test, self.y_train, self.y_test)
            for model in models:
                print(model)
                ob2.train_predict_model(model)
                print()
        else:
            ob2 = cl_modeling(self.X_train, self.X_test, self.y_train, self.y_test)
            # Import model's lib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.tree import DecisionTreeClassifier
            from lightgbm import LGBMClassifier
            from xgboost import XGBRFClassifier

            # Setting Models
            nb = GaussianNB()
            rf = RandomForestClassifier(
                criterion="entropy", n_estimators=500, max_depth=6
            )
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            dt = DecisionTreeClassifier(max_depth=13, min_samples_leaf=10)
            xgb = XGBRFClassifier(max_depth=10, learning_rate=0.1)
            lgbm_rf = LGBMClassifier(
                boosting_type="rf",
                n_jobs=1,
                bagging_freq=3,
                bagging_fraction=0.3,
                importance_type="gain",
            )
            lgbm_dart = LGBMClassifier(
                boosting_type="dart", n_jobs=1, importance_type="gain"
            )
            lgbm = LGBMClassifier(n_jobs=1, importance_type="gain")

            # Evaluating
            model_list = [nb, lr, dt, rf, xgb, lgbm_rf, lgbm_dart, lgbm]
            for model in model_list:
                print(model)
                ob2.train_predict_model(model)
                print()
            ob2 = cl_modeling(self.X_train, self.X_test, self.y_train, self.y_test)
