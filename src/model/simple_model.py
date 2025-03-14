from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV


class SimpleModel:
    def __init__(self, args):
        self.model_type = args.model.lower()
        self.use_cv = getattr(args, 'use_cv', True)  # Default to True for backward compatibility
        
        # Define base models and their parameter grids
        if self.model_type == 'svm':
            self.base_model = SVC(probability=True)
            self.param_grid = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10.0],
            }
            # Default (best) parameters
            self.default_params = {'kernel': 'linear', 'C': 0.1}
            
        elif self.model_type == 'rf':
            self.base_model = RandomForestClassifier(n_estimators=200)
            self.param_grid = {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            }
            self.default_params = {'max_depth': None, 'min_samples_split': 2}
            
        elif self.model_type == 'knn':
            self.base_model = KNeighborsClassifier()
            self.param_grid = {
                'n_neighbors': [3, 5, 7],
            }
            self.default_params = {'n_neighbors': 5}
            
        elif self.model_type == 'dt':
            self.base_model = DecisionTreeClassifier()
            self.param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            }
            self.default_params = {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 5}
            
        elif self.model_type == 'ridge':
            self.base_model = RidgeClassifier()
            self.param_grid = {
                'alpha': [0.1, 1.0, 10.0],
            }
            self.default_params = {'alpha': 10.0}
            
        # Set up the model based on whether we're using cross-validation
        if self.use_cv:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy'
            )
        else:
            # Use base model with default parameters
            self.model = self.base_model.set_params(**self.default_params)

    def fit(self, X_train, y_train):
        if self.use_cv:
            print(f'\nPerforming grid search for {self.model_type.upper()} model...')
            self.model.fit(X_train, y_train)
            print(f'Best parameters: {self.model.best_params_}')
            print(f'Best cross-validation score: {self.model.best_score_:.3f}')
        else:
            print(f'\nTraining {self.model_type.upper()} model with default parameters...')
            print(f'Parameters: {self.default_params}')
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
