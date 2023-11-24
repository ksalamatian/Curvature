import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('out.csv')
df.set_index("Unnamed: 0", inplace=True)
subdf=df.loc[["US","FR","DE","GB", "CH", "RU","UA","IR","IL","SA","QA","JP","KR","CN","AU", "NZ"]]
bar=sns.scatterplot(data=subdf, x="outavg", y="inavg")
bar.set(xlabel='inavg', ylabel='outavg')
#sns.scatterplot(data=subdf,x="ratiosum",y="ratioavg")
for i in range(subdf.shape[0]):
    plt.text(x=subdf.outavg[i]+0.02,y=subdf.inavg[i]+0.02,s=subdf.index[i],
          fontdict=dict(color='red',size=8),
          bbox=dict(facecolor='white',alpha=0.0))
sns.lineplot(x=[0, 2.0],y= [0, 2.0])
plt.xscale('linear')
plt.yscale('linear')
fig=bar.get_figure()
plt.show()
fig.savefig("fig.png")
