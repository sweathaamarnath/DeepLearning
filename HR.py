import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime

class HR_Data_Prep_Utility(object):
	"""HR_Data_Prep_Utility is used for preparing data for the ML"""

	def __init__(self, dataset, feature_col, target_col, fe_hashing_ratio):
		"""
		Initialize and builds the HR Dataset to be used in training a model
		
		Only the below features are supported
		['MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID', 'DeptID', 'PerfScoreID', 'PayRate', 'Termd', 'PositionID', 'State', 'DOB', 'CitizenDesc', 'HispanicLatino', 'RaceDesc', 'DateofHire', 'DateofTermination', 'ManagerName', 'RecruitmentSource', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'LastPerformanceReview_Date']

		:param dataset: pandas dataframe read from csv
		:param feature_col: column names of the features
		:param target_col: column name of the target
		"""
		super(HR_Data_Prep_Utility, self).__init__()
		self.emp_ds = dataset
		self.feature_col = feature_col
		self.target_col = target_col
		self.fe_hashing_ratio = fe_hashing_ratio
		self._cat_col = ['MarriedID', 'MaritalStatusID', 'GenderID','EmpStatusID', 'DeptID', 'Termd', 'PositionID','State', 'CitizenDesc', 'HispanicLatino', 'RaceDesc', 'ManagerName', 'RecruitmentSource']
		self._num_col = ['PayRate', 'PerfScoreID', 'Age', 'CurrentCmpyExp', 'EngagementSurvey', 'EmpSatisfaction','SpecialProjectsCount', 'DaysSinceLastRev']
		self._cat_col_onehot = ['MarriedID', 'MaritalStatusID', 'GenderID','EmpStatusID', 'Termd', 'CitizenDesc', 'HispanicLatino']
		self._cat_columns_feat_hash = ['DeptID', 'PositionID','State', 'RaceDesc', 'ManagerName', 'RecruitmentSource']


	def get_x_y_data(self):
		"""
		Description
		:param name: description
		:return: Description
		"""
		X, y = self._split_x_y(self.emp_ds)
		X = self._values_fix(X)
		X = self._add_features(X[self.feature_col])
		X = self._missing_values_fix(X)
		X = self._encode_category_features(X, reduction_ratio=self.fe_hashing_ratio)
		y = self._missing_values_fix(y)
		y = self._encode_category_features(y, reduction_ratio=self.fe_hashing_ratio)
		X = self._scale_data(X)
		return X, y


	def _fe_fill_missing_val(self, X, column_name, fe_type):
		"""
		Description
		:param name: description
		:return: Description
		"""
		if(fe_type is 'num'):
			X[column_name + '_missing'] =  np.zeros((len(X.index), 1))
			#X.iloc[(X.loc[X[column_name].isna() == True]).index, X.columns.get_loc(column_name + '_missing')] = 1
			X.loc[X[column_name].isna() == True, column_name + '_missing'] = 1
			#X.iloc[(X.loc[X[column_name].isna() == True]).index, X.columns.get_loc(column_name)] = 0
			X.loc[X[column_name].isna() == True, column_name] = 0
		elif(fe_type is 'str'):
			#X.iloc[(X.loc[X[column_name].isna() == True]).index, X.columns.get_loc(column_name)] = 'Missing'
			X.loc[X[column_name].isna() == True, column_name] = 'Missing'
		return X


	def _fe_category_feature_hashing(self, X, column_name, n_features):
		"""
		Description
		:param name: description
		:return: Description
		"""
		fh = FeatureHasher(n_features=n_features, input_type='string')
		x_features_arr = fh.fit_transform(X[column_name].astype('str')).toarray()
		column_names = np.array([])
		for i in range(n_features):
			column_names = np.append(column_names, column_name+'_'+str(i+1))
		return pd.concat([X, pd.DataFrame(x_features_arr, columns=column_names)], axis=1)


	def _fe_category_one_hot_encoder(self, X, column_name):
		"""
		Description
		:param name: description
		:return: Description
		"""
		x_features_arr = pd.get_dummies(X[column_name])
		x_features_arr.rename(columns=lambda x: column_name+'_' + str(x), inplace=True)
		return pd.concat([X, x_features_arr], axis=1)


	def _split_x_y(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		return X[self.feature_col], X[self.target_col]


	def _add_features(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		now = datetime.now()
		if set(['DateofHire','DateofTermination', 'Termd']).issubset(X.columns):
			X['DateofHire'] = pd.to_datetime(X['DateofHire'], format="%m/%d/%Y")
			X['DateofTermination'] = pd.to_datetime(X['DateofTermination'], format="%m/%d/%y")
			X.loc[X['Termd'] == 0, 'CurrentCmpyExp'] = X['DateofHire'].apply(lambda x: now.year - x.year)
			X.loc[X['Termd'] == 1, 'CurrentCmpyExp'] = (X['DateofTermination'] - X['DateofHire'])/np.timedelta64(1,'Y')
			X = X.drop(['DateofHire', 'DateofTermination'], axis=1)
		if 'LastPerformanceReview_Date' in X.columns:
			X['LastPerformanceReview_Date'] = pd.to_datetime(X['LastPerformanceReview_Date'], format="%m/%d/%Y")
			X['DaysSinceLastRev'] = X['LastPerformanceReview_Date'].apply(lambda x: (now - x).days)
			X = X.drop(['LastPerformanceReview_Date'], axis=1)
		if 'DOB' in X.columns:
			X['DOB'] = pd.to_datetime(X['DOB'], format="%d-%m-%Y")
			X['Age'] = X['DOB'].apply(lambda x: now.year - x.year)
			X = X.drop(['DOB'], axis=1)
		return X


	def _format_date_of_termination(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		pattern1_match = X['DateofTermination'].str.match(pat = '^(0[1-9]|1[012])/(0[1-9]|1[0-9]|2[0-9]|3[01])/([0-9]{2})$')
		dates_p1 = pd.to_datetime((X[pattern1_match==True])['DateofTermination'], format="%m/%d/%y")
		pattern2_match = X['DateofTermination'].str.match(pat = '^((19|2[0-9])[0-9]{2})/(0[1-9]|1[012])/(0[1-9]|[12][0-9]|3[01])$')
		dates_p2 = pd.to_datetime((X[pattern2_match==True])['DateofTermination'], format="%Y/%m/%d")
		combined_dates = dates_p1.append(dates_p2)
		X = X.drop(['DateofTermination'], axis=1)
		X.at[combined_dates.index, 'DateofTermination'] = combined_dates.values
		return X


	def _missing_values_fix(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		#Added features are missed from this.. 
		cat_columns = list(set(self.feature_col) & set(self._cat_col))
		num_columns = list(set(self.feature_col) & set(self._num_col)) + ['CurrentCmpyExp', 'DaysSinceLastRev', 'Age']
		for column in cat_columns:
			if column in X.columns:
				X = self._fe_fill_missing_val(X, column, 'str')
		for column in num_columns:
			if column in X.columns:
				X = self._fe_fill_missing_val(X, column, 'num')
		return X


	def _values_fix(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		if 'HispanicLatino' in X.columns:
			X.loc[X['HispanicLatino']=='yes', 'HispanicLatino'] = 'Yes'
			X.loc[X['HispanicLatino']=='no', 'HispanicLatino'] = 'No'
		return X


	def _encode_category_features(self, X, reduction_ratio):
		"""
		Description
		:param name: description
		:return: Description
		"""
		cat_columns_oh = list(set(self.feature_col) & set(self._cat_col_onehot))
		cat_columns_fh = list(set(self.feature_col) & set(self._cat_columns_feat_hash))
		for column in cat_columns_oh:
			if column in X.columns:
				X = self._fe_category_one_hot_encoder(X, column)
		for column in cat_columns_fh:
			if column in X.columns:
				#X = self._fe_category_feature_hashing(X, column, int(len(X[column].unique())*reduction_ratio))
				X = self._fe_category_one_hot_encoder(X, column)
		drop_encoded_fe = []
		for column in cat_columns_oh + cat_columns_fh:
			if column in X.columns:
				drop_encoded_fe.append(column)
		X = X.drop(drop_encoded_fe, axis=1)
		return X


	def _scale_data(self, X):
		"""
		Description
		:param name: description
		:return: Description
		"""
		scaler = StandardScaler()
		scaler.fit(X)
		return pd.DataFrame(scaler.transform(X))