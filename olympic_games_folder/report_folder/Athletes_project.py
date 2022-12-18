#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
from functools import partial
from scipy.stats import shapiro 
from scipy.stats import ttest_ind
from scipy.stats import chisquare 


# ## 1. Read files to one table

# In[2]:


path = "/home/natalya/Stat_BI_2022/athlete_events"
extension = "csv"


# In[3]:


def read_results(path, extension):
    files = glob.glob(os.path.join(path, "*." + extension))
    if extension == "csv":
        pd_function = partial(pd.read_csv, sep = ",")    
    elif extension == "tsv":
        pd_function = partial(pd.read_csv, sep = "\t")   
    elif extension == "xslx":
        pd_function = pd.read_excel
    dfs = []
    for f in files:
        df = pd_function(f)
        dfs.append(df)
    athlete_events = pd.concat(dfs)
    return athlete_events


# In[4]:


athlete_events = read_results(path, extension)


# ## 2. Check the data

# In[5]:


athlete_events


# In[6]:


sum(athlete_events.duplicated())


# There are duplicated rows. Let's delete them - they can distort our results.

# In[7]:


athlete_events = athlete_events.drop_duplicates()


# Let's check if there is any NAs.

# In[8]:


athlete_events.isna().sum()


# Yes, there are some. We could leava them as it is (because usually various function can deal with NA). Numeric variables we could replace with mean value (but it's strange to replace object variables with smth). Sometimes NAs are replaced witht zeros (but here it would be wrong).
# 
# In this case I'd prefer to drop row with NAs, because we have plenty of data, which is enough for our exploratory analysis. Of course, we shouldn't drop NAs in column 'Medal'.

# In[9]:


athlete_events.columns


# In[10]:


athlete_events = athlete_events.dropna(subset=['Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games',
       'Year', 'Season', 'City', 'Sport', 'Event'])


# In[11]:


athlete_events.shape[0]


# In[12]:


athlete_events.isna().sum()


# Let's reset row indecies.

# In[13]:


athlete_events.index = list(range(athlete_events.shape[0]))


# In[14]:


athlete_events.tail()


# Let's check data types.

# In[15]:


athlete_events.dtypes


# In[16]:


athlete_events['Year'] = athlete_events['Year'].astype('int64')


# In[17]:


athlete_events.dtypes


# And some descriptive statistics

# In[18]:


athlete_events.describe()


# It's impossible to be 340 cm high...

# In[19]:


athlete_events.query("Height == 340")


# In[20]:


athlete_events.query("ID == 23549")


# Looks like she was actually 176 high.

# In[21]:


outlier_line_idx = athlete_events.query("Height == 340").index


# In[22]:


athlete_events.loc[outlier_line_idx, "Height"] = 176


# In[23]:


athlete_events["Height"].describe()


# In[24]:


athlete_events.query("Height == 226")


# In[25]:


athlete_events.query("ID == 132627")


# Ok, this guy was really that tall.

# In[26]:


athlete_events[["Sex", "Season", "Medal"]].apply(pd.unique)


# What is G gender?

# In[27]:


athlete_events.query("Sex == 'G'")


# Looks like he is a man.

# In[28]:


index_g = athlete_events.query("Sex == 'G'").index


# In[29]:


athlete_events.loc[index_g, "Sex"] = "M"


# In[30]:


athlete_events["Sex"].unique()


# ## 3. Youngest sportswoman and sportsman

# In[31]:


answer = athlete_events.groupby('Sex').agg({'Age': 'min'})
answer


# In[32]:


m = int(answer.loc['M'][0])
f = int(answer.loc['F'][0])


# In[33]:


print(f'The youngest sportswoman was {f} years old and the youngest sportsman was {m} years old.')


# ## 4. Mean and sd of Height

# In[34]:


athlete_events.groupby('Sex').agg({'Height': ['mean', 'std']})


# ## 5. Mean and sd of Height of female tennis players

# In[35]:


mean = athlete_events.query("Sex == 'F' & Year == 2000 & Sport == 'Tennis'")["Height"].mean()
round(mean, 1)


# In[36]:


std = athlete_events.query("Sex == 'F' & Year == 2000 & Sport == 'Tennis'")["Height"].std()
round(std, 1)


# ## 6. Heaviest athlet in 2006

# In[37]:


idx_line = athlete_events.query("Year == 2006")["Weight"].idxmax()
athlete_events.loc[idx_line, "Sport"]


# ## 7. Gold medals women from 1980 to 2010

# In[38]:


athlete_events.query("Sex == 'F' & 1980 <= Year <= 2010 & Medal == 'Gold'").shape[0]


# ## 8. John Aalberg

# In[39]:


athlete_events.query("Name == 'John Aalberg'")["Year"].nunique()


# ## 9. Age groups in 2008

# In[40]:


subset_2008 = athlete_events.query("Year == 2008")


# In[41]:


age_points = [15, 25, 35, 45, 55]


# In[42]:


for i in range(len(age_points) - 2):
    l_b = age_points[i]
    r_b = age_points[i+1]
    number = subset_2008.query("@l_b <= Age < @r_b")["ID"].nunique()
    print(f"Nummber of sportsmen and sportswomen aged from {l_b} up to {r_b} is {number}.")
l_b = age_points[-2]
r_b = age_points[-1]
number = subset_2008.query("@l_b <= Age < @r_b")["ID"].nunique()
print(f"Nummber of sportsmen and sportswomen aged from {l_b} to {r_b} (including) is {number}.")


# There were a lot of people aged 25-35. There was few people aged 45-55.

# ## 10. Number of sports in 1994 and 2002

# In[43]:


sports_2002 = athlete_events.query("Year == 2002")["Sport"].nunique()
sports_1994 = athlete_events.query("Year == 1994")["Sport"].nunique()
difference = sports_2002 - sports_1994
print(f"""Nummber of various sports in 2002 was {sports_2002} and in 1994 was {sports_1994}. 
In 2002 there were {difference} sports more.""")


# ## 11. TOP 3 Teams

# In[44]:


subset_medals = athlete_events.dropna()


# In[45]:


subset_medals.groupby(['Season', 'Medal'])['NOC'].agg(pd.Series.mode)


# ## 12. Standartization

# In[46]:


height_mu = athlete_events["Height"].mean()
height_std = athlete_events["Height"].std()


# In[47]:


athlete_events["Height_z_scores"] = (athlete_events["Height"] - height_mu) / height_std


# In[48]:


athlete_events.head()


# ## 13. min-max scale

# Min-max normalization gets all the scaled data in the range (0, 1)

# In[49]:


height_min = athlete_events["Height"].min()
height_max = athlete_events["Height"].max()


# In[50]:


height_max, height_min


# In[51]:


athlete_events["Height_min_max_scaled"] = (athlete_events["Height"] - height_min)/(height_max - height_min)


# In[52]:


athlete_events.head()


# ## 14. Height, weight, age of men and women in winter season

# In[53]:


winter_subset = athlete_events.query("Season == 'Winter'")[['Sex', 'Age', 'Height', 'Weight']]


# In[54]:


winter_f_subset = winter_subset.query('Sex == "F"')


# In[55]:


winter_m_subset = winter_subset.query('Sex == "M"')


# In[56]:


winter_f_subset.var()


# In[57]:


winter_m_subset.var()


# The number of observations is quite big, so we ca use T-test. We can use independant T-test because our two samples are independant.

# In[58]:


ttest_ind(winter_f_subset['Age'], winter_m_subset['Age'])


# In[59]:


rcParams['figure.figsize'] = 11.7,8.27
rcParams['font.size'] =  22


# In[60]:


age_fig = sns.boxplot(x=winter_subset['Sex'], y=winter_subset['Age'])
plt.axhline(60, linewidth=2, color='black', xmin=0.25, xmax=0.75)
plt.ylim(7, 70)
plt.text(0.25, 62, "p-value << 0.05", fontsize=20, color='black');


# In[61]:


#age_fig.figure.savefig("age.png")


# In[62]:


ttest_ind(winter_f_subset['Height'], winter_m_subset['Height'])


# In[63]:


height_fig = sns.boxplot(x=winter_subset['Sex'], y=winter_subset['Height'])
plt.axhline(210, linewidth=2, color='black', xmin=0.25, xmax=0.75)
plt.ylim(130, 220)
plt.text(0.25, 215, "p-value << 0.05", fontsize=20, color='black');


# In[64]:


#height_fig.figure.savefig("height.png")


# In[65]:


ttest_ind(winter_f_subset['Weight'], winter_m_subset['Weight'], equal_var=False)


# In[66]:


weight_fig = sns.boxplot(x=winter_subset['Sex'], y=winter_subset['Weight'])
plt.axhline(130, linewidth=2, color='black', xmin=0.25, xmax=0.75)
plt.ylim(30, 140)
plt.text(0.25, 132, "p-value << 0.05", fontsize=20, color='black');


# In[67]:


#weight_fig.figure.savefig("weight.png")


# Men were older, taller and heavier.

# ## 15. Team and medal

# Well, of course they are connected, because different teams have different number of medals...

# In[68]:


gold_madals = athlete_events.query("Medal == 'Gold'")


# In[69]:


gold_madals = gold_madals.groupby('Team').agg({'Medal': 'value_counts'})


# In[70]:


gold_madals = gold_madals.rename(columns={'Medal':'Gold_medals'})


# In[71]:


gold_madals = gold_madals.reset_index().drop('Medal', axis=1)


# In[72]:


gold_madals


# In[73]:


all_medals = subset_medals.groupby('Team').agg({'Medal': 'count'})


# In[74]:


all_medals = all_medals.reset_index().rename(columns={'Medal':'Total_medals'})


# In[75]:


all_medals


# In[76]:


merged_medals = all_medals.merge(gold_madals, how='outer').fillna(0)


# In[77]:


merged_medals['Gold_medals'] = merged_medals['Gold_medals'].astype('int64')


# In[78]:


merged_medals


# In[79]:


sorted_medals = merged_medals.sort_values('Total_medals', ascending=False)


# In[80]:


sorted_medals


# In[81]:


plt.scatter(merged_medals['Total_medals'], merged_medals['Gold_medals'])
plt.xlabel("Number of all medals")
plt.ylabel("Number of gold medals")

plt.text(sorted_medals.iloc[0, 1] - 1000, sorted_medals.iloc[0, 2] - 100, sorted_medals.iloc[0, 0])
plt.text(sorted_medals.iloc[1, 1] - 900, sorted_medals.iloc[1, 2] - 100, sorted_medals.iloc[1, 0])
plt.text(sorted_medals.iloc[2, 1] - 700, sorted_medals.iloc[2, 2] - 100, sorted_medals.iloc[2, 0])


# ## 16. Some hypotheses

# ### Difference in height in two different sports

# In[82]:


athlete_events['Sport'].unique()


# H0: there is no difference in height of sportsmen in Judo and Basketball.
# 
# H1: Height of sportsmen in Judo and Basketball is different.

# In[83]:


judo_bas_m = athlete_events.query('Sport == "Judo" or Sport == "Basketball"').query('Sex == "M"')


# In[84]:


judo_height = judo_bas_m.query('Sport == "Judo"')['Height']
judo_height.var()


# In[85]:


basketball_height = judo_bas_m.query('Sport == "Basketball"')['Height']
basketball_height.var()


# In[86]:


ttest_ind(judo_height, basketball_height)


# In[87]:


sns.boxplot(data=judo_bas_m, x="Sport", y="Height")
plt.axhline(235, linewidth=2, color='black', xmin=0.25, xmax=0.75)
plt.ylim(130, 250)
plt.text(0.25, 240, "p-value << 0.05", fontsize=20, color='black');


# Basketball players are higher.

# ### Weight in two differnt sports

# H0: There is no difference in weight of sportsmen in Judo and Boxing.
# 
# H1: Weight of sportsmen in Judo and Boxing is different.

# In[88]:


judo_box_m = athlete_events.query('Sport == "Judo" or Sport == "Boxing"').query('Sex == "M"')


# In[89]:


judo_weight = judo_box_m.query('Sport == "Judo"')['Weight']
judo_weight.var()


# In[90]:


boxing_weight = judo_box_m.query('Sport == "Boxing"')['Weight']
boxing_weight.var()


# In[91]:


ttest_ind(judo_weight, boxing_weight, equal_var=False)


# In[92]:


sns.boxplot(data=judo_box_m, x="Sport", y="Weight")
plt.axhline(225, linewidth=2, color='black', xmin=0.25, xmax=0.75)
plt.ylim(30, 250)
plt.text(0.25, 230, "p-value << 0.05", fontsize=20, color='black');


# Judo sportsmen are heavier (I thought the opposite...)

# ### How number of women and men changed

# Let's check to intervals: in 20th century and in 21st.

# In[93]:


athlete_events_21 = athlete_events.query("2000 < Year <= 2010")


# In[94]:


athlete_events_20 = athlete_events.query("1950 < Year <= 1960")


# In[95]:


athlete_events_21['Sex'].value_counts()


# In[96]:


athlete_events_20['Sex'].value_counts()


# In[97]:


m_20 = athlete_events_20["Sex"].value_counts()['M']
f_20 = athlete_events_20["Sex"].value_counts()['F']
total_20 = athlete_events_20.shape[0]

m_21 = athlete_events_21["Sex"].value_counts()['M']
f_21 = athlete_events_21["Sex"].value_counts()['F']
total_21 = athlete_events_21.shape[0]


# In[98]:


f_part = (f_20 +  f_21) / (total_20 + total_21)
m_part = (m_20 +  m_21) / (total_20 + total_21)


# In[99]:


m_20_exp = total_20 * m_part
f_20_exp = total_20 * f_part
m_21_exp = total_21 * m_part
f_21_exp = total_21 * f_part


# In[100]:


chisquare([m_20, f_20, m_21, f_21], f_exp=[m_20_exp, f_20_exp, m_21_exp, f_21_exp])


# Looks like the ratio of men and women really changed.

# In[101]:


year_count = athlete_events.groupby('Year').agg({'Sex': 'value_counts'})
year_count = year_count.rename(columns={'Sex':'Count'}).reset_index()


# In[102]:


year_count.head()


# In[103]:


year_total = year_count.groupby('Year').agg({'Count': 'sum'}).reset_index()


# In[104]:


sns.lineplot(data=year_total, x="Year", y="Count", color='black')


# This is how number of all participants changed: it became sharply bigger in 1950s. I suppose that fluctuations in the end reflect the division to winter and summer games to different years.

# In[105]:


sns.lineplot(data=year_count, x="Year", y="Count", hue="Sex")


# We can see, that number of men doesn't considerably change since 1960s, but number of women is slightly growing.
