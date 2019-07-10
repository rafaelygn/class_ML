class premodeling:

    def __init__(self, dataset, index_cols=None, ignore_cols=None, drop_cols=None):
        '''
        Here we define what will be index, cat_cols, cat_cols_used and drop columns
        '''
        # Importing libs
        import numpy as np
        import pandas as pd
        self.dataset = dataset.copy()
        # Set index columns
        if index_cols:
            self.dataset.set_index(index_cols, inplace=True)
        # Drop columns in drop_cols
        if drop_cols:
            self.dataset.drop(labels=drop_cols, axis=1, inplace=True)
        # Set importants features
        self.cat_cols = self.dataset.select_dtypes(
            include='object').columns.tolist()
        self.num_cols = self.dataset.select_dtypes(
            exclude='object').columns.tolist()
        if ignore_cols:
            # Use all categories unless ignore cols
            self.cat_cols_used = [
                col for col in self.cat_cols if col not in ignore_cols]
            self.num_cols_used = [
                col for col in self.num_cols if col not in ignore_cols]
        else:
            # Use all categories columns
            self.cat_cols_used = self.cat_cols
            self.num_cols_used = self.num_cols

    def fill_missing(self, na_cat='desconhecido', na_num='mean'):
        '''
        Preenche valores nulos para valores categóricos com 'desconhecido'
        e para valores numéricos com a média.

        Para escorar o teste, os valores das médias vão estar salvos
        no train_mean.
        '''
        self.train_mean = self.dataset[self.num_cols_used].mean()
        if na_cat:
            self.dataset[self.cat_cols_used] = self.dataset[self.cat_cols_used].fillna(
                value=na_cat, axis=1)
        if na_num:
            self.dataset[self.num_cols_used] = self.dataset[self.num_cols_used].fillna(
                value=self.dataset[self.num_cols_used].mean())
        return self.dataset

    def encoding_others(self, per_others=0, cat_cols=None, other_name='Outros',
                        ignore_cols=None, index_cols=None):
        '''
        Codificamos os valores categoricos com ocorrência menor que min_others
        por 'Outros'
        '''
        if cat_cols:
            # Use some categories columns
            self.cat_cols_used_other = [
                col for col in self.cat_cols_used if col in ignore_cols]
        else:
            # Use all categories columns
            self.cat_cols_used_other = self.cat_cols_used
        # Creating a log dictionary
        self.dict_log = {}
        # Mapping and saving in self.dict_log
        for col in self.cat_cols_used_other:
            var = self.dataset[col].value_counts(normalize=True)*100
            self.dict_log[col] = var[var > per_others].index.tolist()
        # What isn't in dict_log, replace by 'other_name'
        for col in self.dict_log.keys():
            self.dataset[col] = self.dataset[col].apply(
                lambda x: x if x in self.dict_log.get(col) else other_name)
        return self.dataset

    def encoding_OneHot(self, cat_cols=None, drop_cat=True):
        if cat_cols:
            # Use some categories columns
            self.cat_cols_used_encoding = [
                col for col in self.cat_cols_used if col in cat_cols]
        else:
            # Use all categories columns
            self.cat_cols_used_encoding = self.cat_cols_used
        import re
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(self.dataset[self.cat_cols_used_encoding])
        a_encoding = enc.transform(
            self.dataset[self.cat_cols_used_encoding]).toarray()
        # Take the columns Name
        enc_name = enc.get_feature_names().tolist()
        col_name = []
        for num, col in zip(range(len(self.dataset[self.cat_cols_used_encoding])), self.dataset[self.cat_cols_used_encoding]):
            partial_col_name = [name.replace(re.findall('x[0-9]+', name)[0], col)
                                for name in enc_name if re.findall('[0-9]+', name)[0] == str(num)]
            col_name += partial_col_name
        df_encoding = pd.DataFrame(a_encoding, columns=col_name)
        # return self.dataset, enc, a_encoding, col_name
        for col in col_name:
            self.dataset[col] = df_encoding[col].values
        if drop_cat:
            self.dataset.drop(
                labels=self.cat_cols_used_encoding, axis=1, inplace=True)
        return self.dataset, enc

    def encoding_mean_rate(self, target, cat_cols=None, min_obs=100):
        if cat_cols:
            # Use some categories columns
            self.cat_cols_used_meanrate = [
                col for col in self.cat_cols_used if col in cat_cols]
        else:
            # Use all categories columns
            self.cat_cols_used_meanrate = self.cat_cols_used

        for col in self.cat_cols_used_meanrate:
            df_mean = pd.DataFrame(self.dataset.groupby([col])[target].mean())
            df_mean.columns = [col+'_mean']
            df_count = pd.DataFrame(
                self.dataset.groupby([col])[target].count())
            df_count.columns = [col+'_count']
            df_result = df_mean.join(df_count)
            df_result = df_result[df_result[col+'_count'] > min_obs]
            self.dataset = self.dataset.join(df_result[col+'_mean'], on=[col])
            self.num_cols_used.append(col+'_mean')

    def encoding_label(self, target, str_target):
        n = len(self.dataset[target].unique())
        print(n)
        if n != 2:
            print('FAIL. nº de features: ', n)
            return None
        else:
            self.dataset[target].apply(lambda x: 1 if x == str_target else 0)
