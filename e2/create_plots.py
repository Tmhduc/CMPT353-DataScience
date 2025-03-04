import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    if len(sys.argv) != 3:
        sys.exit(1)
    
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    df1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
                    names=['lang','page','views','bytes'])

    df2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
                    names=['lang','page','views','bytes'])

    print(df1)
    print(df2)
    plt.figure(figsize=(10,5))
    
    # plot 1
    plt.subplot(1,2,1)
    df1_sorted = df1.sort_values(by=['views'], ascending=False)
    plt.plot(df1_sorted['views'].values)
    plt.title("Distribution of Views")
    plt.xlabel("Rank")
    plt.ylabel("Views")

    # plot 2
    plt.subplot(1,2,2)
    common_pages = df1.index.intersection(df2.index)
    combine_views = pd.DataFrame({
        'views_hour1': df1.loc[common_pages, 'views'],
        'views_hour2': df2.loc[common_pages, 'views']
    })
    
    print(combine_views)
    plt.scatter(combine_views['views_hour1'], combine_views['views_hour2'])
    plt.title("Hourly Views Comparison")
    plt.xlabel("Views in Hour 1")
    plt.ylabel("Views in Hour 2")
    plt.xscale('log')
    plt.yscale('log')
    
    # Show both plot
    plt.savefig("wikipedia.png")
    plt.show()

if __name__ == "__main__":
    main()