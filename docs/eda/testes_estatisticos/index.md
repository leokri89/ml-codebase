## **One sample t-test**
The One Sample t Test determines whether the sample mean is statistically different from a known or hypothesised population mean. The One Sample t Test is a parametric test.

**Example**: you have 10 ages and you are checking whether avg age is 30 or not.
```python title="docs\eda\testes_estatisticos\t-test_1sample.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\t-test_1sample.py"
```

## **Two sampled T-test**
The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test

```python title="docs\eda\testes_estatisticos\t-test_2sample.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\t-test_2sample.py"
```

## **Paired sampled t-test**
The paired sample t-test is also called dependent sample t-test. It’s an uni variate test that tests for a significant difference between 2 related variables. An example of this is if you where to collect the blood pressure for an individual before and after some treatment, condition, or time point.

**H0:** means difference between two sample is 0

**H1:** mean difference between two sample is not 0

```python title="docs\eda\testes_estatisticos\t-test_paired_sample.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\t-test_paired_sample.py"
```

## **When you can run a Z Test.**

Several different types of tests are used in statistics (i.e. f test, chi square test, t test). You would use a Z test if:

- Your sample size is greater than 30. Otherwise, use a t test.

- Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point.

- Your data should be normally distributed. However, for large sample sizes (over 30) this doesn’t always matter.

- Your data should be randomly selected from a population, where each item has an equal chance of being selected.
Sample sizes should be equal if at all possible.

```python title="docs\eda\testes_estatisticos\z-test_1sample.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\z-test_1sample.py"
```

## **Two-sample Z test**
In two sample z-test, similar to t-test here we are checking two independent data groups and deciding whether sample mean of two group is equal or not.

**H0: mean of two group is 0**

**H1: mean of two group is not 0**

```python title="docs\eda\testes_estatisticos\z-test_2sample.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\z-test_2sample.py"
```

## **ANOVA (F-TEST)**
The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the same time. For example, if we wanted to test whether voter age differs based on some categorical variable like race, we have to compare the means of each level or group the variable. We could carry out a separate t-test for each pair of groups, but when you conduct many tests you increase the chances of false positives. The analysis of variance or ANOVA is a statistical inference test that lets you compare multiple groups at the same time.

**F = Between group variability / Within group variability**

Unlike the z and t-distributions, the F-distribution does not have any negative values because between and within-group variability are always positive due to squaring each deviation.

## **Anova - One Way F-test**
It tell whether two or more groups are similar or not based on their mean similarity and f-score.

Example : there are 3 different category of plant and their weight and need to check whether all 3 group are similar or not

```python title="docs\eda\testes_estatisticos\anova_one_way_f-test.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\anova_one_way_f-test.py"
```

## **Two Way F-test**
Two way F-test is extension of 1-way f-test, it is used when we have 2 independent variable and 2+ groups. 2-way F-test does not tell which variable is dominant. if we need to check individual significance then Post-hoc testing need to be performed.

Now let’s take a look at the Grand mean crop yield (the mean crop yield not by any sub-group), as well the mean crop yield by each factor, as well as by the factors grouped together

```python title="docs\eda\testes_estatisticos\two_way_f-test.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\two_way_f-test.py"
```

## **Chi-Square Test**
The test is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

For example, in an election survey, voters might be classified by gender (male or female) and voting preference (Democrat, Republican, or Independent). We could use a chi-square test for independence to determine whether gender is related to voting preference

```python title="docs\eda\testes_estatisticos\chi_squared-test.py" linenums="1"
--8<-- "docs\eda\testes_estatisticos\chi_squared-test.py"
```
