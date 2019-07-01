class Exploring:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def corr_matrix(self, target=None, heatmap=None, **kwargs_plot):
        corr_matrix = self.dataset.corr()
        if target: 
            result = corr_matrix[target].sort_values(ascending=False)
            return pd.DataFrame(result)
        elif heatmap:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(corr_matrix, alpha=.9, cmap=plt.get_cmap("coolwarm"), **kwargs_plot)
            return None
        else:
            return corr_matrix
        
    def plot_corr_matrix(self, subset=None, **kwargs_plot):
        from pandas.plotting import scatter_matrix as scatter_matrix
        if subset:
            scatter_matrix(self.dataset[subset],figsize=(12,8), **kwargs_plot)
            plt.show()
            return None
        else:
            scatter_matrix(self.dataset,figsize=(12,8))
            plt.show()
            return None
    
    def plot_geo(self, x, y, var, **kwargs_plot):
        from matplotlib import pyplot as plt
        x_value = self.dataset[x]
        y_value = self.dataset[y]
        z_value = self.dataset[var]
        fig, ax = plt.subplots(figsize=(12,8))
        ax_color = ax.scatter(x_value, y_value, 
                    c=z_value,
                    label=var,
                    cmap=plt.get_cmap("jet"),
                    **kwargs_plot)
        cbar = fig.colorbar(ax_color)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.legend()
        plt.show()
        return None

    def plot_percentil(self, var1, var2, percentiles=[.25, .50, .75], table=None, **kwargs_plot):
        array_percentil = self.dataset[var1].describe(percentiles)
        index_percentil = list(str(int(per*100)).replace('0.', '')+'%' for per in percentiles)
        label = index_percentil + ['max']
        self.dataset['group_'+var1] = pd.cut(self.dataset[var1],
                                       bins=array_percentil[['min']+index_percentil+['max']],
                                       labels=label)
        df_mean  = pd.DataFrame(self.dataset.groupby(by='group_'+var1)[var2].mean())
        df_count = pd.DataFrame(self.dataset.groupby(by='group_'+var1)[var2].count())
        df_std   = pd.DataFrame(self.dataset.groupby(by='group_'+var1)[var2].std())
        
        df_final = df_mean.join(df_count, rsuffix='_count').join(df_std, rsuffix='_std')
        if table:
            return df_final
        else:
            df_final[var2+'_std_upper'] = df_final[var2] + df_final[var2+'_std']/2
            df_final[var2+'_std_lower'] = df_final[var2] - df_final[var2+'_std']/2
            x = np.arange(df_final.shape[0])
            y = df_final[var2]
            y2 = df_final[var2+'_count']
            std = df_final[var2+'_std']
            std_upper = df_final[var2+'_std_upper']
            std_lower = df_final[var2+'_std_lower']
            fig, ax1 = plt.subplots(figsize=(12,8))
            ax1.plot(x, y, color='black', alpha=0.6)
            ax1.plot(x, std_lower, color='black', alpha=0.3)
            #ax1.stackplot(x, std_upper, std_lower)
            ax1.plot(x, std_upper, color='black', alpha=0.3)
            ax1.set_xlabel('Percentil '+var1)
            ax1.set_ylabel('mean_'+var2)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Volumetria')
            ax2.bar(x, y2, width=0.2, alpha=.3, color='green')
            plt.xticks(x, label)
            return None

    def boxplot_normalized(self, drop_columns=None):
        df_numerical = self.dataset._get_numeric_data()
        df_box = (df_numerical - df_numerical.mean())/df_numerical.std()
        if drop_columns:
            df_box.drop(drop_columns, axis=1, inplace=True)
        fig, ax1 = plt.subplots(figsize=(12,8))
        sns.boxplot(data=df_box)
        plt.show()
        return None
        
    def data_info(self):
        info = pd.DataFrame()
        info["var"] = self.dataset.columns
        info["# missing"] = list(self.dataset.isnull().sum())
        info["% missing"] = info["# missing"] / self.dataset.shape[0]*100
        info["types"] = list(self.dataset.dtypes)
        info["unique values"] = list(len(self.dataset[var].unique()) for var in self.dataset.columns)
        return info
