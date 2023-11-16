#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np


# In[43]:


import pandas as pd


# In[44]:


myindex = ['USA','Canada','Mexico']


# In[45]:


mydata = [1776,1876,1821]


# In[46]:


myser = pd.Series(data=mydata)


# In[47]:


myser


# In[48]:


myser = pd.Series(data=mydata,index=myindex)


# In[49]:


myser


# In[50]:


myser['USA']


# In[51]:


ages = {'Sam':5,'Frank':10,'Spike':7}


# In[52]:


pd.Series(ages)


# In[53]:


q1 = {'Japan':80,'China':450,'India':200,'USA':250}
q2 = {'Brazil':100,'China':500,'India':210,'USA':260}


# In[54]:


sales_q1 = pd.Series(q1)


# In[55]:


sales_q2 = pd.Series(q2)


# In[56]:


sales_q1


# In[57]:


sales_q2


# In[58]:


sales_q1['Japan']


# In[59]:


sales_q1 + sales_q2


# In[60]:


sales_q1.add(sales_q2,fill_value=0)


# In[61]:


mydata = np.random.randint(0,101,(4,3))


# In[62]:


mydata


# In[63]:


myindex = ['CA','NY','AZ','TX']


# In[64]:


mycolumns = ['Jan','Feb','Mar']


# In[65]:


df = pd.DataFrame(mydata)


# In[66]:


df


# In[67]:


df = pd.DataFrame(data=mydata,index=myindex)


# In[68]:


df


# In[69]:


df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)


# In[70]:


df


# In[74]:


df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/tips.csv')


# In[75]:


df


# In[76]:


df.columns


# In[77]:


df.index


# In[78]:


df.head()


# In[79]:


df.tail()


# In[80]:


df.describe()


# In[81]:


df.describe().transpose()


# In[82]:


df['total_bill']


# In[83]:


mycols = ['total_bill','tip']
df[mycols]


# In[84]:


df['tip'] + df['total_bill']


# In[85]:


df['tip_percentage'] = 100 * df['tip']/df['total_bill']


# In[86]:


df.head()


# In[87]:


df['price_per_person'] = np.round(df['total_bill']/df['size'],2)


# In[88]:


df.drop('tip_percentage',axis=1)# droping columns


# In[89]:


df


# In[90]:


# setting an extra index
df = df.set_index("Payment ID")


# In[91]:


df.head()


# In[92]:


df = df.reset_index() # undoing the previous operation


# In[93]:


df.head()


# In[94]:


df.iloc[0]


# In[95]:


df.iloc[0:3]
one_row = df.iloc[1]


# In[96]:


df.loc[[0,8]]
one_row


# In[97]:


df = df.drop(1)


# In[98]:


df


# In[102]:


df


# In[103]:


df.iloc[0:4]


# In[104]:


import numpy as np
import pandas as pd


# In[105]:


df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/tips.csv')


# In[106]:


df.head()


# In[107]:


bool_series = df['total_bill'] > 40


# In[108]:


df[bool_series]


# In[109]:


df[df['total_bill'] > 40]


# In[110]:


df[df["sex"]=='Male']


# In[111]:


df[df['size'] > 3]


# In[112]:


# AND & --- BOTH COND NEED TO BE TRUE

# OR / ---- EITHER COND IS TRUE


# In[113]:


df[(df['total_bill'] > 30)|(df['sex'] == "Male")]


# In[114]:


df[(df['day']=='Sun')|(df['day']=='Sat')]


# In[115]:


options = ['Sat','Sun']


# In[116]:


df['day'].isin(options)


# In[117]:


df[df['day'].isin(['Sat','Sun'])]


# In[118]:


def yelp(price):
    if price < 10:
        return '$'
    elif price >= 10 and price < 30:
        return '$$'
    else:
        return '$$$'


# In[119]:


df['yelp'] = df['total_bill'].apply(yelp)


# In[120]:


df


# In[121]:


def quality(total_bill,tip):
    if tip/total_bill > 0.25:
        return "Generous"
    else:
        return "Other"


# In[122]:


quality(16.99,1.01)


# In[123]:


df[['total_bill','tip']].apply(lambda df:quality(df['total_bill'],df['tip']),axis=1)


# In[124]:


df['Quality'] = np.vectorize(quality)(df['total_bill'],df['tip'])


# In[125]:


df.head()


# In[126]:


import numpy as np
import pandas as pd
df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/tips.csv')


# In[127]:


df.describe().transpose()


# In[128]:


# Sorting columns
df.sort_values('tip',ascending=False)


# In[129]:


# Sorting by multiple columns
df.sort_values(['tip','size'])


# In[130]:


df['total_bill'].max()


# In[131]:


# returning the index location of max value in column 'total_bill'
df['total_bill'].idxmax()


# In[132]:


df.iloc[170]


# In[133]:


df.corr()


# In[134]:


# counting categorical columns
df['sex'].value_counts()


# In[135]:


df['day'].unique()


# In[136]:


df['day'].value_counts()


# In[137]:


# replacing values
df['sex'].replace('Female','F')


# In[138]:


df['sex'].replace(['Female','Male'],['F','M'])


# In[139]:


mymap = {'Female':'F','Male':'M'}


# In[140]:


df['sex'].map(mymap)


# In[141]:


simple_df = pd.DataFrame([1,2,2,2],['a','b','c','d'])


# In[142]:


simple_df


# In[143]:


simple_df.duplicated()


# In[144]:


simple_df.drop_duplicates()


# In[145]:


df[df['total_bill'].between(10,20,inclusive=True)]


# In[146]:


df.nlargest(10,'tip')


# In[147]:


df.sample(5)


# In[148]:


df.sample(frac=0.1) # return random 10% of the columns


# In[149]:


import numpy as np
import pandas as pd


# In[154]:


df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/movie_scores.csv')


# In[155]:


df.head()


# In[156]:


df.isnull()


# In[157]:


df.notnull()


# In[158]:


df[df['pre_movie_score'].isnull()]


# In[159]:


df


# In[160]:


df.dropna()


# In[161]:


df.dropna(thresh=4)# returns only the rows that have 4 non-null-values


# In[162]:


df.dropna(thresh=5)


# In[163]:


df.dropna(subset=['last_name'])


# In[164]:


help(df.fillna)


# In[165]:


df.fillna('NEW VALUE!')


# In[166]:


df['pre_movie_score']


# In[167]:


df['pre_movie_score'].fillna(df['pre_movie_score'].mean())


# In[168]:


df.fillna(df.mean())


# In[169]:



df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/mpg.csv')


# In[170]:


df


# In[171]:


df['model_year'].value_counts()


# In[172]:


df.groupby('model_year').mean()


# In[173]:


df.groupby(['model_year','cylinders']).mean()


# In[174]:


df.groupby(['model_year','cylinders']).mean().index


# In[175]:


year_cyl = df.groupby(['model_year','cylinders']).mean()


# In[176]:


year_cyl


# In[177]:


year_cyl.index.names


# In[178]:


year_cyl.index.levels


# In[179]:


year_cyl.loc[[70,82]]


# In[180]:


import numpy as np
import pandas as pd


# In[181]:


data_one = {'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3']}


# In[182]:


data_two = {'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']}


# In[183]:


one = pd.DataFrame(data_one)


# In[184]:


two = pd.DataFrame(data_two)


# In[185]:


one


# In[186]:


two


# In[187]:


# concatenating Series based on columns
pd.concat([one,two],axis=1)


# In[188]:


pd.concat([one,two])


# In[189]:


two.columns = one.columns


# In[190]:


two


# In[191]:


pd.concat([one,two])


# In[192]:


mydf = pd.concat([one,two])


# In[193]:


mydf


# In[194]:


mydf.index


# In[195]:


mydf.index = range(len(mydf))


# In[196]:


mydf


# In[197]:


registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bob','Claire','David']})


# In[198]:


logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bob']})


# In[199]:


registrations


# In[200]:


logins


# In[201]:


pd.merge(registrations,logins,how='inner',on='name')


# In[202]:


pd.merge(left=registrations,right=logins,how='left',on='name')


# In[203]:


pd.merge(left=registrations,right=logins,how='right',on='name')


# In[204]:


pd.merge(registrations,logins,how='outer',on='name')


# In[205]:


# combining Data Frames based on indexes
registrations = registrations.set_index('name')


# In[206]:


registrations


# In[207]:


logins


# In[208]:


import numpy as np
import pandas as pd


# In[209]:


email = 'jose@email.com'


# In[210]:


email.split('@')


# In[211]:


names = pd.Series(['andrew','bob','claire','david','5'])


# In[212]:


names


# In[213]:


names.str.upper()


# In[214]:


names.str.isdigit()


# In[215]:


tech_finance = ['GOOG,APPL,AMZN','JPM,BAC,GS']


# In[216]:


len(tech_finance)


# In[217]:


tickers = pd.Series(tech_finance)


# In[218]:


tickers


# In[219]:


tickers.str.split(',')


# In[220]:


tech = 'GOOG;APPL;AMZN'


# In[221]:


tech.split(';')


# In[222]:


tech.split(';')[0]


# In[223]:


tickers.str.split(',')[0]


# In[224]:


tickers.str.split(',').str[0]


# In[225]:


tickers = pd.Series(tech_finance)


# In[226]:


tickers.str.split(',',expand=True)


# In[227]:


messy_names = pd.Series(['andrew  ','bo;bo','   claire '])


# In[228]:


messy_names


# In[229]:


messy_names = messy_names.str.replace(';','')


# In[230]:


messy_names


# In[231]:


names = messy_names.str.strip()


# In[232]:


names


# In[233]:


cleaned_names = names.str.capitalize()


# In[234]:


cleaned_names


# In[235]:


messy_names = pd.Series(['andrew  ','bo;bo','   claire '])


# In[236]:


def cleanup(name):
    name = name.replace(";","")
    name = name.strip()
    name = name.capitalize()
    return name


# In[237]:


messy_names.apply(cleanup)


# In[238]:


from datetime import datetime
import pandas as pd
myyear = 2016
mymonth = 1
myday = 1
myhour = 3
mymin = 30
mysec = 15


# In[239]:


mydate = datetime(myyear,mymonth,myday)


# In[240]:


mydate


# In[241]:


mydatetime = datetime(myyear,mymonth,myday,myhour,mymin,mysec)


# In[242]:


mydatetime


# In[243]:


mydatetime.year


# In[244]:


myser = pd.Series(['November 3, 1990','2000-01-01',None])


# In[245]:


myser


# In[28]:


# timeser = pd.to_datetime(myser)
date = pd.to_datetime('July-1-2015')
date.day_name()


# In[12]:


timeser


# In[51]:


obvi_euro_date = '31-12-2000'


# In[52]:


pd.to_datetime(obvi_euro_date)


# In[53]:


euro_date = '10-12-2000'


# In[54]:


pd.to_datetime(euro_date,dayfirst=True)


# In[247]:


style_date = '01--Dec--2000'


# In[248]:


custom_date = "12th of Dec 2000"


# In[ ]:





# In[250]:


sales = pd.read_csv('RetailSales_BeerWineLiquor.csv')


# In[251]:


sales


# In[252]:


sales['DATE'] # this is not a date object


# In[253]:


sales['DATE'] = pd.to_datetime(sales['DATE'])


# In[254]:


sales['DATE']


# In[255]:


sales


# In[256]:


sales = sales.set_index("DATE")


# In[257]:


sales


# In[258]:


# resampling
sales.resample(rule='A').mean()


# In[259]:


import pandas as pd


# In[260]:


df = pd.read_csv('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/example.csv')


# In[261]:


df


# In[262]:


df.to_csv('newfile.csv',index=False)


# In[263]:


new = pd.read_csv("newfile.csv")


# In[264]:


new


# In[265]:


url = "https://en.wikipedia.org/wiki/World_population"


# In[266]:


tables = pd.read_html(url)


# In[267]:


len(tables)


# In[268]:


tables[4]


# In[269]:


excel_sheet_dict = pd.read_excel('C:/Users/PC/Desktop/Python for Machine learning/03-Pandas/my_excel_file.xlsx',sheet_name=None)


# In[270]:


excel_sheet_dict.keys()


# In[271]:


df=excel_sheet_dict['First_Sheet']


# In[272]:


df


# In[273]:


import pandas as pd
import numpy as np


# In[274]:


from sqlalchemy import create_engine


# In[275]:


temp_db = create_engine('sqlite:///:memory:')


# In[276]:


df = pd.DataFrame(data=np.random.randint(low=0,high=100,size=(4,4)),columns=['a','b','c','d'])


# In[277]:


df


# In[278]:


df.to_sql(name='new_table',con=temp_db)


# In[279]:


new_df = pd.read_sql(sql='new_table',con=temp_db)


# In[280]:


new_df


# In[281]:


result = pd.read_sql_query(sql="SELECT a,c FROM new_table",con=temp_db)


# In[282]:


result


# In[283]:


import numpy as np


# In[284]:


arr = np.array([[-7/3,-2/3],[1/3,-8/3]])


# In[285]:


print(arr)


# In[4]:


print ("exponential of array is/n")
print(np.exp(arr))


# In[ ]:




