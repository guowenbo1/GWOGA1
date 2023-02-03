
import pandas as pd
def ReadData(filePath,a):

    data = pd.read_csv(filePath,header=0)
    # data = data.loc[data.loc[:,'gene_name'].isin(gene),:] #selected same gene with gse15222
    #data = data.dropna(axis=0,how='any')
    selected_gene = data.index.isin(a)

    data.index = data['gene_name'].values
    data = data.drop(labels='gene_name',axis=1)
    print(data.iloc[selected_gene,:])
    geneName = data.index
    labels = data.columns.values #0 没病，1 有病
    geneData = data.T
    geneData = pd.DataFrame(geneData,columns=geneName)
    # print(geneData)
    label_list = []
    for i in labels:
        if 'CON' in i:
            label_list.append(0)
        else:
            label_list.append(1)
    return geneName,geneData,label_list

a = [1,3,4]
gse15222 = 'gse15222.csv'
geneName,geneData,label = ReadData(gse15222,a)
