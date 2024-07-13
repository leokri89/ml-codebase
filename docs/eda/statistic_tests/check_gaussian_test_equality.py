from scipy import stats
import numpy as np
import pandas as pd

"""
Essa python deve verificar se duas distribuições são Gaussianas ou não
e verifica se elas são estatisticamente iguais
"""

def generate_data(loc1=0, loc2=0, scale1=1, scale2=1, n=100):
    non_gaussian_1 = stats.skewnorm.rvs(1, loc=loc1, scale=scale1, size=n)
    non_gaussian_2 = stats.skewnorm.rvs(1, loc=loc2, scale=scale2, size=n)

    gaussian_1 = np.random.normal(loc=loc1, scale=scale1, size=n)
    gaussian_2 = np.random.normal(loc=loc2, scale=scale2, size=n)

    return pd.DataFrame({'sample_non_gaussian_1': non_gaussian_1,'sample_non_gaussian_2':non_gaussian_2,
                       'sample_gaussian_1':gaussian_1,'sample_gaussian_2':gaussian_2})


def test_mannwhitneyu(sample1, sample2):
    print('Testing choice: Mann-Whitney U rank test')
    sts, p = stats.mannwhitneyu(sample1,
                                sample2,
                                use_continuity=True,
                                alternative='two-sided',
                                axis=0,
                                method='auto')
    if p > 0.05:
        print('Two Distribution looks equal (p>0.05), pvalue: {:.3f}'.format(p))
    else:
        print('Two Distribution does not looks equal (p<=0.05), pvalue: {:.3f}'.format(p))


def test_t(sample1, sample2):
    print('Testing choice: t-test')
    sts, p = stats.ttest_ind(sample1, sample2)
    if p > 0.05:
        print('Two Distribution looks equal (p>0.05), pvalue: {:.3f}'.format(p))
    else:
        print('Two Distribution does not looks equal (p<=0.05), pvalue: {:.3f}'.format(p))


def check_gaussian_shapiro(sample1, sample2):
    _, p1_shapiro = stats.shapiro(sample1)
    _, p2_shapiro = stats.shapiro(sample2)
    if p1_shapiro > 0.05 and p2_shapiro > 0.05:
        print('Samples looks Gaussian via Shapiro: Sample 1 - {}, Sample 2 - {}'.format(p1_shapiro, p2_shapiro))
        return True
    print('Samples does not look Gaussian via Shapiro: Sample 1 - {}, Sample 2 - {}'.format(p1_shapiro, p2_shapiro))
    return False


def check_gaussian_dagostino(sample1, sample2):
    _, p1_dagostinos = stats.normaltest(sample1)
    _, p2_dagostinos = stats.normaltest(sample2)
    if p1_dagostinos > 0.05 and p2_dagostinos > 0.05:
        print('Samples looks Gaussian via D`Agostino`s: Sample 1 - {}, Sample 2 - {}'.format(p1_dagostinos, p2_dagostinos))
        return True
    print('Samples does not look Gaussian via D`Agostino`s: Sample 1 - {}, Sample 2 - {}'.format(p1_dagostinos, p2_dagostinos))
    return False


def check_gaussian_anderson(sample1, sample2):
    print(stats.anderson(sample1))
    print(stats.anderson(sample2))
    #if 'need to fill':
    #    print('Samples looks Gaussian via Anderson')
    #    return True
    #print('Samples does not look Gaussian via Anderson')
    #return False


def check_gaussian_kstest(sample1, sample2):
    print(stats.ks_2samp(sample1, sample2))


loc1=50
loc2=50
scale1=10
scale2=10
n=1000

df = generate_data(loc1, loc2, scale1, scale2)

samples_col = ['sample_gaussian_1','sample_gaussian_2']
#samples_col = ['sample_non_gaussian_1','sample_non_gaussian_2']

sns.displot(df[samples_col])

sample1 = df[samples_col[0]]
sample2 = df[samples_col[1]]

test_1 = check_gaussian_shapiro(sample1, sample2)
test_2 = check_gaussian_dagostino(sample1, sample2)
#test_3 = check_gaussian_anderson(sample1, sample2)
#ks_test = check_gaussian_kstest(sample1, sample2)

if test_1 or test_2:
    test_t(sample1, sample2)
else:
    test_mannwhitneyu(sample1, sample2)