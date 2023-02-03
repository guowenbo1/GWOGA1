import pandas as pd

def same_gene():
    gse15222_path = 'gse15222.csv'
    ngs_path = 'voom_normalization_log2.csv'
    gse15222 = pd.read_csv(gse15222_path)
    ngs_data = pd.read_csv(ngs_path)
    gse15222_gene = gse15222['SYMBOL'].str.split('_',expand=True).iloc[:,0]
    ngs_gene = ngs_data['gene_name']
    inter_gene = set(ngs_gene).intersection(gse15222_gene)
    return inter_gene



