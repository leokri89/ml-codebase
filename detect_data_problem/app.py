
import re

#import click
import numpy as np
from scipy.stats import poisson

def fn_trata_string(string, sep):
    string = re.sub('[^0-9 \.\,]','',string)
    while ',' in string or '  ' in string:
        string = string.replace(',','.').replace('  ',' ').strip()
    return [int(float(value)) for value in string.split(sep)]

def cast_series(X, eval, sep=' '):
    try:
        int_serie = fn_trata_string(X, sep)
    except:
        print('Falha para converter serie')

    try:
        eval = int(float(eval.replace(',','.')))
    except:
        print('Falha para converter o valor avaliado')
    return int_serie, eval

def get_data(serie):
    serie = np.array(serie)
    lbound = round(np.quantile(serie,0.10))
    ubound = round(np.quantile(serie,0.90))
    mu = round(serie[(serie > lbound) & (serie < ubound)].mean())
    serie_wo_outlier = serie[(serie > lbound - mu) & (serie < ubound + mu)]
    std_dev = round(serie_wo_outlier.std())
    print('\n' + ('-'*80))
    print(f'Serie: {serie}')
    print(f'Low Bound: {lbound}, Upper Bound: {ubound}')
    print(f'Serie without outliers: {serie_wo_outlier}')
    print(f'Mean: {mu}, Standard Deviation: {std_dev}')
    print('-'*80)
    return {'lbound':lbound,'ubound':ubound,'mu':mu,'serie':serie_wo_outlier,'std_dev':std_dev}

def get_error_probability(series, eval_value, mu, threshold=[0.65, 0.8]):
    prc_serie = poisson.pmf(series, mu)
    res = poisson.pmf(eval_value, mu)

    err_pct = float(1 - (res / prc_serie.max()))
    threshold = np.sort(threshold)
    
    if err_pct < threshold[0]:
        aviso = 'Baixa probabilidade de erro'
    elif err_pct < threshold[1]:
        aviso = 'Media probabilidade de erro'
    else:
        aviso = 'Alta probabilidade de erro'

    frmt_res = round(res * 100,2)
    frmt_err_pct = round(err_pct * 100,2)

    print(f'\nProbabilidade eventual: {frmt_res}%')
    print(f'Probabilidade de erro: {frmt_err_pct}%')
    print(f'Aviso: {aviso}')
    return {'percentage_event':frmt_res,'err_percentage':frmt_err_pct, 'aviso': aviso}