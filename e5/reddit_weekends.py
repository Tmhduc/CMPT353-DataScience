from scipy import stats
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def filter_data(file):
    counts = pd.read_json(file, lines=True)
    counts['date'] = pd.to_datetime(counts['date'])
    # year_series = counts['date'].dt.year
    filtered_counts = counts[(counts['date'].dt.year.isin([2012, 2013])) & (counts['subreddit'] == 'canada')]
    return filtered_counts

def seperate_weekdays_weekends(filtered_counts):
    filtered_counts['weekday'] = filtered_counts['date'].dt.weekday
    weekdays = filtered_counts[filtered_counts['weekday'] < 5]['comment_count']
    weekends = filtered_counts[filtered_counts['weekday'] >= 5]['comment_count']
    return weekdays, weekends

def T_test(weekdays, weekends):
    ttest = stats.ttest_ind(weekdays, weekends).pvalue
    initial_weekday_normality = stats.normaltest(weekdays).pvalue
    initial_weekend_normality = stats.normaltest(weekends).pvalue
    initial_levene_p = stats.levene(weekdays, weekends).pvalue
    return ttest, initial_weekday_normality, initial_weekend_normality, initial_levene_p

def log_transformation(weekdays, weekends):
    log_weekdays = np.log(weekdays)
    log_weekends = np.log(weekends)
    transformed_weekday_normality_p = stats.normaltest(log_weekdays).pvalue
    transformed_weekend_normality_p = stats.normaltest(log_weekends).pvalue
    transformed_levene_p = stats.levene(log_weekdays, log_weekends).pvalue
    return transformed_weekday_normality_p, transformed_weekend_normality_p, transformed_levene_p

def sqrt_transformation(weekdays, weekends):
    sqrt_weekdays = np.sqrt(weekdays)
    sqrt_weekends = np.sqrt(weekends)
    transformed_weekday_normality_p = stats.normaltest(sqrt_weekdays).pvalue
    transformed_weekend_normality_p = stats.normaltest(sqrt_weekends).pvalue
    transformed_levene_p = stats.levene(sqrt_weekdays, sqrt_weekends).pvalue
    return transformed_weekday_normality_p, transformed_weekend_normality_p, transformed_levene_p

def central_limit_theorem(filtered_counts):
    # print(filtered_counts['date'].apply(lambda x: x.isocalendar()))
    filtered_counts['iso_year'], filtered_counts['iso_week'], _ = zip(*filtered_counts['date'].apply(lambda x: x.isocalendar()))
    filtered_counts['weekday_col'] = filtered_counts['weekday'] < 5
    grouped = filtered_counts.groupby(['iso_year', 'iso_week', 'weekday_col'])['comment_count'].mean().reset_index()
    
    weekdays_mean = grouped[grouped['weekday_col'] == True]['comment_count']
    weekends_mean = grouped[grouped['weekday_col'] == False]['comment_count']
    
    weekly_weekday_normality_p = stats.normaltest(weekdays_mean).pvalue
    weekly_weekend_normality_p = stats.normaltest(weekends_mean).pvalue
    weekly_levene_p = stats.levene(weekdays_mean, weekends_mean).pvalue
    weekly_ttest_p = stats.ttest_ind(weekdays_mean, weekends_mean).pvalue
    
    return weekly_weekday_normality_p, weekly_weekend_normality_p, weekly_levene_p, weekly_ttest_p

def non_parametric_test(weekdays, weekends):
    utest_p = stats.mannwhitneyu(weekdays, weekends, alternative="two-sided").pvalue
    return utest_p

def main():
    file = sys.argv[1]
    filtered_counts = filter_data(file)
    weekdays, weekends = seperate_weekdays_weekends(filtered_counts)
    ttest_p, initial_weekday_normality_p, initial_weekend_normality_p, initial_levene_p = T_test(weekdays, weekends)
    transformed_weekday_normality_p, transformed_weekend_normality_p, transformed_levene_p = sqrt_transformation(weekdays, weekends)
    weekly_weekday_normality_p, weekly_weekend_normality_p, weekly_levene_p, weekly_ttest_p = central_limit_theorem(filtered_counts)
    utest_p = non_parametric_test(weekdays, weekends)
    average_weekday_comments = weekdays.mean()
    average_weekend_comments = weekends.mean()

    if average_weekday_comments > average_weekend_comments:
        print("More Reddit comments are posted on weekdays in /r/canada.")
    else:
        print("More Reddit comments are posted on weekends in /r/canada.")

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = ttest_p,
        initial_weekday_normality_p = initial_weekday_normality_p,
        initial_weekend_normality_p = initial_weekend_normality_p,
        initial_levene_p = initial_levene_p,
        transformed_weekday_normality_p = transformed_weekday_normality_p,
        transformed_weekend_normality_p = transformed_weekend_normality_p,
        transformed_levene_p = transformed_levene_p,
        weekly_weekday_normality_p = weekly_weekday_normality_p,
        weekly_weekend_normality_p = weekly_weekend_normality_p,
        weekly_levene_p = weekly_levene_p,
        weekly_ttest_p = weekly_ttest_p,
        utest_p = utest_p,
    ))

if __name__ == "__main__":
    main()
    
