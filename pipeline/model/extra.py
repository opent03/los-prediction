self.data=self.data[(self.data['los']>=include_time)]
self.hids=self.data['stay_id'].unique()
labels_csv=pd.DataFrame(columns=['stay_id','label'])
labels_csv['stay_id']=pd.Series(self.hids)
labels_csv['label']=0
for hid in self.hids:
        grp=self.data[self.data['stay_id']==hid]
        dataDic[hid]={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{},'ethnicity':grp['ethnicity'].iloc[0],'age':int(grp['Age']),'gender':grp['gender'].iloc[0],'label':int(grp['label'])}
        labels_csv.loc[labels_csv['stay_id']==hid,'label']=int(grp['label'])
labels_csv.to_csv('./data/csv/labels.csv',index=False)   

def mortality_length(self,include_time,predW):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time+predW)]
        self.hids=self.data['stay_id'].unique()

def create_label(cohort_output, out_name, include_time=24):
    data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
    data['intime'] = pd.to_datetime(data['intime'])
    data['outtime'] = pd.to_datetime(data['outtime'])
    data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
    data['los']=data['los'].astype(str)
    data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
    data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
    data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
    data=data.drop(columns=['days', 'dummy','hours','min','sec'])
    data=data[data['los']>0]
    data['Age']=data['Age'].astype(int)
    data = data[(data['los']>=include_time)]
    hids = data['stay_id'].unique()
    labels_csv = data.loc[:,['stay_id','label']]
    labels_csv.to_csv('./data/csv/'+out_name+'.csv',index=False)   

