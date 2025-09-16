import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import wandb
from utils.csv_utils import read_aggregated_features



import random

class SVMModel:
	def __init__(self, args):
		self.args = args
		seed = getattr(args, 'seed', 42)
		np.random.seed(seed)
		random.seed(seed)
		self.load_data()
		kernel = getattr(args, 'svm_kernel', 'rbf')
		C = getattr(args, 'svm_c', 1.0)
		gamma = getattr(args, 'svm_gamma', 'scale')
		self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=seed)

	def load_data(self):
		df = read_aggregated_features(self.args)
		df['wavnet'] = df['wavnet'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
		df['mfcc'] = df['mfcc'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
		all_file_paths, wavenet_features, mfcc_features, classes = zip(*df.itertuples(index=False, name=None))
		features = np.array(wavenet_features)
		X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.15, random_state=42, shuffle=True)
		le = LabelEncoder()
		y_train = le.fit_transform(y_train)
		y_test = le.transform(y_test)
		self.X_train, self.X_test = X_train, X_test
		self.y_train, self.y_test = y_train, y_test
		self.le = le

	def train(self):
		wandb.init(project=self.args.wandb_project, config=vars(self.args))
		self.model.fit(self.X_train, self.y_train)
		train_acc = self.model.score(self.X_train, self.y_train)
		val_acc = self.model.score(self.X_test, self.y_test)
		wandb.log({"train_acc": train_acc, "val_acc": val_acc})
		wandb.finish()

	def evaluate(self):
		val_acc = self.model.score(self.X_test, self.y_test)
		return {"val_acc": val_acc}

	def save(self, path):
		import joblib
		joblib.dump(self.model, path)
