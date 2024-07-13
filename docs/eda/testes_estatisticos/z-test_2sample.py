from statsmodels.stats import weightstats as stests

ztest, pval = stests.ztest(df['bp_before'],
                            x2=df['bp_after'],
                            value=0,
                            alternative='two-sided')

print(float(pval))

if pval < 0.05:
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")