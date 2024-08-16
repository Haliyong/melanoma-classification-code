from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def create_association_set(dataset):
    def set_creator(first_column, second_column):
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        one = encoder.fit_transform(dataset[[first_column]])
        one_df = pd.DataFrame.sparse.from_spmatrix(one, columns=encoder.get_feature_names_out([first_column]))
    
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        two = encoder.fit_transform(dataset[[second_column]])
        two_df = pd.DataFrame.sparse.from_spmatrix(two, columns=encoder.get_feature_names_out([second_column]))
        
        return pd.concat([one_df, two_df ], axis=1)
    
    return set_creator

def create_association_rules(dataset, name):
    frequent_sets = apriori(dataset, min_support=0.1, use_colnames=True)
    rules=association_rules(frequent_sets, metric="confidence", min_threshold=0.7)
    rules.to_csv(f'{name}.csv', index=False)
    
# Import and check data
dataset=pd.read_csv('metadata/train-metadata.csv', dtype=str, low_memory=False)
#print(dataset.head(5))
set_creator = create_association_set(dataset)

# Check if there are patterns for gender/sex and location
location_association = set_creator('anatom_site_general', 'sex')
#print(location_association.head(5))
create_association_rules(location_association, 'location')

# Check if there are patterns with image and lighting types
image_association = set_creator('image_type', 'tbp_tile_type')
create_association_rules(image_association, 'image')

# Check if there are patterns for location and the confidence of the lesion being a nevus
nevus_association = set_creator('tbp_lv_location', 'tbp_lv_nevi_confidence')
create_association_rules(nevus_association, 'nevus')