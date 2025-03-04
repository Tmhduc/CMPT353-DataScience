import sys
import pandas as pd
import scipy.stats as stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)

def load_data(filename):
    return pd.read_json(filename, orient='records', lines=True)

def analyze_search_usage(df):
    df = df.copy()
    df.loc[:, 'group'] = df['uid'] % 2
    df.loc[:, 'searched'] = df['search_count'] > 0
    
    print(df)
    contigency = pd.crosstab(df['group'], df['searched'])
    chi2, more_users_p, dof, expected = stats.chi2_contingency(contigency)
    
    treatment = df[df['group'] == 1]['search_count']
    control = df[df['group'] == 0]['search_count']
    U, more_searches_p = stats.mannwhitneyu(treatment, control)
     
    return more_users_p, more_searches_p

def is_instructor(df):
    return df[df['is_instructor'] == True]

def main():

    if len(sys.argv) != 2:
        print("Usage: python3 ab_analysis.py searches.json")
        sys.exit(1)

    searchdata_file = sys.argv[1]
    data_df = load_data(searchdata_file)

    # data_df['group'] = data_df['uid'] % 2
    # data_df['searched'] = data_df['search_count'] > 0
    # print(pd.crosstab(data_df['group'], data_df['searched']))
    # print(data_df[data_df['search_count'] > 0])
    
    more_users_p, more_searches_p = analyze_search_usage(data_df)
    
    instructors_df = is_instructor(data_df)
    more_instr_p, more_instr_searches_p  = analyze_search_usage(instructors_df)
    
    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p= more_users_p,
        more_searches_p= more_searches_p,
        more_instr_p= more_instr_p,
        more_instr_searches_p= more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
