import pandas as pd
import matplotlib.plot as plt

# Read DataFrame
pd.read_csv('tse.csv', index_col='date', parse_dates=['date'])
pd.read_excel()
pd.read_html()
pd.read_json()

# Examining DataFrame
df.info()
df.head()
df.tail()
df.index
df.columns 

# Reindex
df.reindex()
df.set_index([column1, column2])
df.sort_index()
df.sort_values()
df.column.astype('category')
df.asfreq('D')
df.to_timestamp()

# Indexing and Slicing
df[column]
df.column
df.loc[row_name, column_name]
df.iloc[row_number, column_number]
df[[column]]

# Filtering
df[condition] # df[df.salt > 60]
df.dropna()

# Pivoting
df.pivot(index=, columns=, values=)
df.unstack(level=)
df.stack(level=)
df.swaplevel(0, 1)
pd.melt(df, id_vars=, var_name=, value_name=)
df.pivot_table(index=, columns=, values=, aggfunc=,)

# Groupby
df.groupby()
.agg(['max','sum'])
.transform(zscore)
.apply()

# Counting
df.value_counts()
df.idxmax(axis='columns')

# Arithmetic
df.add()
df.sub()
df.mul()
df.divide()

df.shift()
df.pct_change()
df.diff()

# Plot
df.plot()
plt.show()


.resample() + transformation method

pd.date_range(start=, end=, periods=, freq=)

# Concatenating and merging 
.append()
pd.concat(keys=, axis=, join=)
.join()
pd.merge(on=, suffixes=, how=)
pd.merge_ordered()
pd.merge_asof()



