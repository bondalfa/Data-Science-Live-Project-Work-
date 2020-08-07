# -*- coding: utf-8 -*-
"""visualization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/187Nz3C4jvcGnhIeqfHdrSlaFwpL35xKv
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ds = pd.read_csv("DS_DATESET.csv")

ds.head()

ds.isnull().sum()

ds = ds.drop(['link to Linkedin profile','Link to updated Resume (Google/ One Drive link preferred)','Certifications/Achievement/ Research papers'], axis = 1)

print(ds)

"""**Question 2.a**
*The number of students applied to different technologies.*
"""

a = sns.countplot(x='Areas of interest',data=ds, order = ds['Areas of interest'].value_counts().index)
plt.xticks(rotation=90)



"""# **Question 2.b**
 The number of students applied for Data Science who knew ‘’Python” and who *didn’t* *italicized text*
"""

dfA = ds[['Areas of interest','Programming Language Known other than Java (one major)']]

dfa1=dfA[(dfA['Areas of interest'] == 'Data Science ') & (dfA['Programming Language Known other than Java (one major)'] != 'Python')]

dfa1.head()

df2 = dfA.loc[(dfA['Areas of interest'] == 'Data Science ') & (dfA['Programming Language Known other than Java (one major)'] == 'Python')]

df2.head()

dfB = dfA.loc[(dfA['Areas of interest']== 'Data Science ') & (dfA['Programming Language Known other than Java (one major)'])]

dfB['Programming Language Known other than Java (one major)'] = dfB['Programming Language Known other than Java (one major)'].map(lambda dfB : 1 if dfB == 'Python' else 0)

ax = sns.countplot(x="Programming Language Known other than Java (one major)", data=dfB)
ax.set(xlabel='Students who know Python or Not', ylabel='No of students opting FOr DATA SCIENCE ')
plt.savefig('multipage_pdf.pdf')
plt.show()





"""# Question 2.c
The different ways students learned about this program
"""

dfc = ds['How Did You Hear About This Internship?'].value_counts().plot(kind = 'barh', title = "No. Students applied to different technologies")

sns.countplot(x='How Did You Hear About This Internship?',data=ds)
plt.xticks(rotation=90)
plt.savefig('multipage_pdf.pdf',bbox_inches='tight')

"""# Question 2.d
Students who are in the fourth year and have a CGPA greater than 8.0
"""

cgp = ds[['Which-year are you studying in?','CGPA/ percentage']]
cgp.head()

df_cg = cgp[(cgp['Which-year are you studying in?'] =='Fourth-year') & (cgp['CGPA/ percentage'])]

df_cg['CGPA/ percentage'] = df_cg['CGPA/ percentage'].map(lambda df_cg: 1 if df_cg > 8 else 0)

d = sns.countplot(x="CGPA/ percentage", data=df_cg)
ax.set(xlabel='CGPA greater than 8', ylabel='common ylabel')
plt.show()

"""# Question 2.e
*Students who applied for Digital Marketing with verbal and written communication score greater than 8*
"""

verb = ds
verb = verb[(verb['Areas of interest'] == 'Digital Marketing ')]
verb.head()

verb = verb[(verb['Areas of interest'] == 'Digital Marketing ')]
verb['Rate your verbal communication skills [1-10]'] = verb['Rate your verbal communication skills [1-10]'].map(lambda s: 1 if s > 8 else 0)
ax = sns.countplot(x="Rate your verbal communication skills [1-10]", data=verb)
ax.set(xlabel='Verbal skills greater than 8 (1 if greater than 8 else 0)', ylabel='no. of student applying for Digital Marketing')

"""# Question 2.f
Year-wise and area of study wise classification of students
"""

year = ds
f_year = year[year['Which-year are you studying in?'] == 'First-year']

s_year = ds
s_year = s_year[s_year['Which-year are you studying in?'] == 'Second-year']

t_year = ds
t_year = t_year[t_year['Which-year are you studying in?'] == 'Third-year']

fin_year = ds
fin_year = fin_year[fin_year['Which-year are you studying in?'] == 'Fourth-year']

f = sns.countplot(x="Major/Area of Study", data=f_year)
f = f.set(xlabel='Area of study', ylabel='No. of students')
plt.xticks(rotation=-45)
plt.title('Students in First Year')
f = sns.set_style('white')
plt.show()

f = sns.countplot(x="Major/Area of Study", data=s_year)
f = f.set(xlabel='Area of study', ylabel='No. of students')
plt.xticks(rotation=-45)
plt.title('Students in Second Year')
f = sns.set_style('white')
plt.show()

f = sns.countplot(x="Major/Area of Study", data=t_year)
f = f.set(xlabel='Area of study', ylabel='No. of students')
plt.xticks(rotation=-45)
plt.title('students in Third Year')
f = sns.set_style('white')
plt.show()

f = sns.countplot(x="Major/Area of Study", data=fin_year)
f = f.set(xlabel='Area of study', ylabel='No. of students')
plt.xticks(rotation=-45)
plt.title('students in Fourth Year')
f = sns.set_style('white')
plt.show()

"""#Question 2.g
City and college wise classification of students
"""

ds['City'].value_counts()

# question G
g = sns.countplot(x="City", data=ds)
g = g.set(xlabel='CITY', ylabel='No. of students')
plt.xticks(rotation=-45)
plt.title('NO. of students from the city')
g = sns.set_style('white')
plt.show()

"""#Question 2.h
Plot the relationship between the CGPA and the target variable
"""

sns.swarmplot(x='Label', y='CGPA/ percentage', data=ds)
plt.show()

"""# Question 2.i
Plot the relationship between the Area of Interest and the target variable
"""

i = ds.groupby(['Areas of interest', 'Label']).size().unstack().plot(kind='barh',title='\nThe relationship between the Area of Interest and the target variable' )

"""## Question 2.j
Plot the relationship between the year of study, major, and the target variable.
"""

is_2020 =  ds['Expected Graduation-year']==2020
df_2020 = ds[is_2020]
df_2020.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='bar', title = 'the relationship between the year of 2020(class of 2020), major, and the target variable')

is_2021 =  ds['Expected Graduation-year']==2021
df_2021 = ds[is_2021]
df_2021.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='bar',title = 'the relationship between the year of 2021(class of 2021), major, and the target variable')

is_2022 =  ds['Expected Graduation-year']==2022
df_2022 = ds[is_2022]
df_2022.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='bar', title = 'the relationship between the year of 2022(class of 2022), major, and the target variable')

is_2023 =  ds['Expected Graduation-year']==2023
df_2023 = ds[is_2023]
df_2023.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='bar', title = 'the relationship between the year of 2023(class of 2023), major, and the target variable')









import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

with PdfPages('visualization-output.pdf') as pdf:

  a = sns.countplot(x='Areas of interest',data=ds, order = ds['Areas of interest'].value_counts().index)
  plt.xticks(rotation=90)
  plt.title('Number of students applied to different technologies.')
  pdf.savefig(bbox_inches='tight')
  plt.close()
  

  b = sns.countplot(x="Programming Language Known other than Java (one major)", data=dfB)
  b.set(xlabel='Students who know Python or Not', ylabel='No of students opting FOr DATA SCIENCE ')
  plt.title('The number of students applied for Data Science who knew ‘’Python” and who didn’t KNEW')
  pdf.savefig(bbox_inches='tight')
  plt.close()
  
  c = sns.countplot(x='How Did You Hear About This Internship?',data=ds)
  plt.xticks(rotation=90)
  pdf.savefig(bbox_inches='tight')
  plt.close()
  
  d = sns.countplot(x="CGPA/ percentage", data=df_cg)
  d.set(xlabel='CGPA greater than 8', ylabel='common ylabel')
  pdf.savefig()
  plt.close()

  e = sns.countplot(x="Rate your verbal communication skills [1-10]", data=verb)
  e.set(xlabel='Verbal skills greater than 8 (1 if greater than 8 else 0)', ylabel='no. of student applying for Digital Marketing')
  pdf.savefig()
  plt.close()

  f = sns.countplot(x="Major/Area of Study", data=f_year)
  f = f.set(xlabel='Area of study', ylabel='No. of students')
  plt.xticks(rotation=-45)
  plt.title('Students in First Year')
  f = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()


  f = sns.countplot(x="Major/Area of Study", data=s_year)
  f = f.set(xlabel='Area of study', ylabel='No. of students')
  plt.xticks(rotation=-45)
  plt.title('Students in Second Year')
  f = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()


  f = sns.countplot(x="Major/Area of Study", data=t_year)
  f = f.set(xlabel='Area of study', ylabel='No. of students')
  plt.xticks(rotation=-45)
  plt.title('students in Third Year')
  f = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()

  f = sns.countplot(x="Major/Area of Study", data=fin_year)
  f = f.set(xlabel='Area of study', ylabel='No. of students')
  plt.xticks(rotation=-45)
  plt.title('students in Fourth Year')
  f = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()
    
  g = sns.countplot(x="City", data=ds)
  g = g.set(xlabel='CITY', ylabel='No. of students')
  plt.xticks(rotation=-45)
  plt.title('NO. of students from the city')
  g = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()

  fig, g1 = plt.subplots(figsize =(10,10))
  g1 = sns.countplot(x="College name", data=ds)
  g1 = g1.set(xlabel='College name', ylabel='No. of students')
  plt.xticks(rotation=90)
  plt.title('NO. of students per college')
  g1 = sns.set_style('white')
  pdf.savefig(bbox_inches='tight')
  plt.close()
  
  h= sns.swarmplot(x='Label', y='CGPA/ percentage', data=ds)
  plt.title('Relationship between the CGPA and the target variable')
  pdf.savefig(bbox_inches='tight')
  plt.close()

  is_2020 =  ds['Expected Graduation-year']==2020
  df_2020 = ds[is_2020]
  df_2020.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='barh', title = 'the relationship between the year of 2020(class of 2020), major, and the target variable')
  pdf.savefig(bbox_inches='tight')
  plt.close()


  is_2021 =  ds['Expected Graduation-year']==2021
  df_2021 = ds[is_2021]
  df_2021.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='barh',title = 'the relationship between the year of 2021(class of 2021), major, and the target variable')
  pdf.savefig(bbox_inches='tight')
  plt.close()


  is_2022 =  ds['Expected Graduation-year']==2022
  df_2022 = ds[is_2022]
  df_2022.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='barh', title = 'the relationship between the year of 2022(class of 2022), major, and the target variable')
  pdf.savefig(bbox_inches='tight')
  plt.close()

  is_2023 =  ds['Expected Graduation-year']==2023
  df_2023 = ds[is_2023]
  df_2023.groupby(['Major/Area of Study', 'Label']).size().unstack().plot(kind='barh', title = 'the relationship between the year of 2023(class of 2023), major, and the target variable')
  pdf.savefig(bbox_inches='tight')
  plt.close()
