import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

# Read in .csv and select every 5th state/action observation
df = pd.read_csv('cartpole_phy_q_obs_800.csv', names=['state1', 'state2', 'state3', 'state4', 'action'])
df = df[(df.index % 5) == 0].reset_index()
features = ['state1','state2','state3','state4']

x = df.loc[:, features].values
y = df.loc[:,['action']].values
x = StandardScaler().fit_transform(x)

# Compute 3-component PCA on state observations
pca = PCA(n_components=3)
prin_comps = pca.fit_transform(x)
prin_df = pd.DataFrame(data=prin_comps)

# Combine principal component dataframe with action dataframe, renmae columns
plot_df = pd.concat([prin_df, df[['action']]], axis=1)
plot_df.rename(columns={plot_df.columns[0]: 'PC1', plot_df.columns[1]: 'PC2', plot_df.columns[2]: 'PC3', 'action': 'action'}, 
        inplace=True)

# Plot 3D scatter of principal component projections colored
# according to action (0 or 1)
fig = px.scatter_3d(plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color=plot_df['action'].astype('string'),
        labels={'0':'PC1','1':'PC2','2':'PC3'},
        title= 'Total Explained Variance: {0}'.format(pca.explained_variance_ratio_.sum()*100)
        )
fig.update_traces(marker={'size': 1})
fig.show()

"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0.0, 1.0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = plot_df['action'] == target
    ax.scatter(plot_df.loc[indicesToKeep, 'PC1'], 
            plot_df.loc[indicesToKeep, 'PC2'],
            plot_df.loc[indicesToKeep, 'PC3'],
            c = color,
            s = 10)
ax.legend(targets)
ax.grid()
plt.show()
"""
