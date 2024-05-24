import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel
import pandas as pd

DEFAULT_PARAMS_PT = {
    'hidden_layers': [1],#, 2],#, 3],
    'hidden_units': [16],# , 64, 128],
    # 'learning_rate': [1e-4, 1e-3, 1e-2],
     'learning_rate': [1e-2],
    'batch_size': [16],#, 32, 64],
    'epochs': [5]
}

class PyTorchEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, model, criterion, optimizer, epochs=100, batch_size=32):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy().flatten()  # Ensure the output is a 1D array

class PyTorchRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        
        self.hidden_layers = kwargs.get('hidden_layers', 2)
        self.hidden_units = kwargs.get('hidden_units', 64)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        
        self.model = self.build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert processed DataFrame to PyTorch tensors
        self.X_train = torch.tensor(self.X_train.to_numpy(), dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
        
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size, shuffle=False)

        self.estimator = PyTorchEstimator(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        self.search = None  # Initialize the search attribute to None

    def build_model(self):
        layers = []
        input_dim = self.X_train.shape[1]
        
        layers.append(nn.Linear(input_dim, self.hidden_units))
        layers.append(nn.ReLU())
        
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(self.hidden_units, 1))
        return nn.Sequential(*layers)

    def tune_hyper_parameters(self):
        param_grid = {
            'hidden_layers': DEFAULT_PARAMS_PT['hidden_layers'],
            'hidden_units': DEFAULT_PARAMS_PT['hidden_units'],
            'learning_rate': DEFAULT_PARAMS_PT['learning_rate'],
            'batch_size': DEFAULT_PARAMS_PT['batch_size'],
            'epochs': DEFAULT_PARAMS_PT['epochs']
        }
        search = GridSearchCV(self.estimator, param_grid, cv=3, scoring=self.scoring, n_jobs=-1)
        search.fit(self.X_train.numpy(), self.y_train.numpy().ravel())
        self.search = search
        self.update_best_params(search.best_params_)

    def update_best_params(self, best_params):
        self.hidden_layers = best_params.get('hidden_layers', self.hidden_layers)
        self.hidden_units = best_params.get('hidden_units', self.hidden_units)
        self.learning_rate = best_params.get('learning_rate', self.learning_rate)
        self.batch_size = best_params.get('batch_size', self.batch_size)
        self.epochs = best_params.get('epochs', self.epochs)
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.estimator = PyTorchEstimator(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def train(self):
        if self.search:
            # Perform hyperparameter tuning
            self.tune_hyper_parameters()

        # Train the model
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
        return self.estimator
    
    def predict(self, X_data):
        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_data)
        return predictions.numpy().flatten()
    
    @property
    def default_params(self):
        return DEFAULT_PARAMS_PT
    
    @property
    def unnecessary_parameters(self):
        return ['scoring', 'split_column', 'create_encoding_rules', 'apply_encoding_rules', 'create_transformations', 'apply_transformations', 'test_size',
                'already_splitted_data']
