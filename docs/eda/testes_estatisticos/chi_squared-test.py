import pandas as pd
from scipy.stats
import chi2

df_chi = pd.read_csv('chi-test.csv')
contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Shopping?"])

print('contingency_table :-\n',contingency_table)#Observed Values

Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom:-",ddof)
alpha = 0.05

chi_square = sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

chi_square_statistic = chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value: ',critical_value)#p-value

p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value: ', p_value)
print('Significance level: ', alpha)
print('Degree of Freedom: ', ddof)
print('chi-square statistic: ', chi_square_statistic)
print('critical_value: ', critical_value)
print('p-value: ', p_value)

if chi_square_statistic >= critical_value:
    print("Reject H0, There is a relationship between 2 categorical variables")
else:
    print("Retain H0, There is no relationship between 2 categorical variables")
    
if p_value <= alpha:
    print("Reject H0, There is a relationship between 2 categorical variables")
else:
    print("Retain H0, There is no relationship between 2 categorical variables")