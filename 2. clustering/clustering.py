from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

questions=[
    "First impression",
    "Speaks English",
    "Students are comfortable",
    "Students are stressed out",
    "Clear about course objectives",
    "Satisfied with course contents"
]

options=[
    ["Poor", "Not Good", "Fair", "Very Good", "Excellent"],
    ["Never", "Seldom", "Sometimes", "Majority of the time", "Always"],
    ["Strong disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
    ["Strong disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
    ["Strong disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
    ["Strong disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
]

def plot_data(data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    for i in range(0, reduced_data.shape[0]):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c='r', marker='.')
    plt.show()

def plot_cluster(km,maxK):
    font0 = FontProperties()
    font0.set_size(10)
    n=int(input("Enter cluster #:"))

    n=n-1
    if (n >= maxK or n<0):
        print("Enter a valid cluster size")
        return

    c = km.cluster_centers_
    fig, axes = plt.subplots(2, 3)
    for j in range(len(questions)):
        k = j * 5
        Y = c[n][k:k+5]
        X = np.arange(0,5)
        x=int(j%2)
        y=int(j/2)
        axes[x,y].bar(X,Y)
        axes[x, y].set_title(questions[j])
        objects=options[j][:]
        axes[x, y].set_xticklabels(objects, fontproperties=font0, minor=False, rotation=25)
        axes[x, y].set_ylim([0, 100])
    plt.suptitle('Faculty Group ' + str(n+1), fontsize=16)
    plt.show()





def print_labels(km,df,k):
    print(km.labels_)
    l=[]
    for i in range(k):
        l.append("")                        #Empty line for each group
    for i in range(km.labels_.shape[0]):    #for each point/ observation in cluster
        j=km.labels_[i]                     #Find the label of the observation
        if(l[j]):
            l[j]+= ", " + str(df["Course"][i])
        else:
            l[j]= "Group "+ str(j+1) + " comprises: " + str(df["Course"][i])
    print(l)


print("*******************************************************************************************")
print("*       Welcome to the three Prong solution to Faculty Evaluation - Clustering    *")
print("*******************************************************************************************")
print("Reading data")

df = pd.read_csv('data.csv')
df1 = df.drop(df.columns[[0, 1]], axis=1)
data = df1.as_matrix()

print("Finding the appropriate number of clusters")

K = range(2, 6)
maxK=-1
max=-1
scs=[]
km= KMeans(n_clusters=2)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sc=metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')
    scs.append(sc)
    if maxK==-1 or sc>max:
        maxK=k
        km=kmeans
        max=sc


print("\n\nWe found ", maxK, " groups of clusters (faculty members) according to the faculty evaluation response")

cmd=""
while (True):
    cmd = input("#:")
    if (cmd == "data"):
        plot_data(data)

    elif (cmd=="cluster"):
        plot_cluster(km,maxK)

    elif (cmd=="list"):
        print_labels(km,df,maxK)

    if(cmd=="quit"):
        break