"""
这个文件是用来生成data。pkl文件的，
生成后可以直接运行HGCL-main\dataset\Yelp\Generate_ICI.py
以及HGCL-main\dataset\Yelp\GenerateDistanceMat.py
运行完后就可以实现HGCL了
"""

from sklearn.utils import shuffle
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations
from collections import defaultdict

"""city1"""
# 读取CSV文件
csv_file_path = 'dataraw/city_train.csv'
train_df = pd.read_csv(csv_file_path)
train_df = shuffle(train_df)

# 获取train中唯一的项目和类别
unique_projects_train = train_df['venueid_id'].unique()
unique_categories_train = train_df['cate_id'].unique()
unique_users_train = train_df['user'].unique()

# 总文件
csv_file_path = 'dataraw/city1.csv'
all_df = pd.read_csv(csv_file_path)
# 获取总项目和类别的数量
num_projects = len(all_df['venueid_id'].unique())
num_categories = len(all_df['cate_id'].unique())
num_users = len(all_df['user'].unique())


"""创建项目与类别的dict"""
# 创建一个字典，用于存储类别及相应的项目列表
category_dict = {}
# 遍历DataFrame，将类别及相应的项目列表存储到字典中
for _, row in train_df.iterrows():
    category = row['cate_id']
    project = row['venueid_id']
    if category not in category_dict:
        category_dict[category] = [project]
    else:
        if project not in category_dict[category]:
            category_dict[category].append(project)
print("项目与类别的dict done")

"""创建项目与类别的关系矩阵"""
project_category_matrix = csr_matrix((num_projects, num_categories), dtype=int)
# 填充关系矩阵
for _, row in train_df.iterrows():
    project_index = row['venueid_id']
    category_index = row['cate_id']
    project_category_matrix[project_index, category_index] = 1
print("项目与类别的关系矩阵 done")
###

""""用户与项目之间的"""
# 创建用户与项目的关系矩阵
user_project_matrix = csr_matrix((num_users, num_projects), dtype=int)
# # 填充关系矩阵
for _, row in train_df.iterrows():
    user_index = row['user']
    project_index = row['venueid_id']
    user_project_matrix[user_index, project_index] = 1
print("用户与项目 done")
####


"""用户与用户的关系"""
###创建一个字典，用于存储相同项目的用户组合
project_user_dict = defaultdict(list)
# 遍历DataFrame，将相同项目的用户组合存储到字典中
for _, row in train_df.iterrows():
    project_user_dict[row['venueid_id']].append(row['user'])
# 创建一个列表，用于存储用户组合
user_combinations = []
# 遍历字典，获取用户组合
for users in project_user_dict.values():
    users = set(users)
    if len(users) > 1:
        user_combinations.extend(combinations(users, 2))
# 获取用户之间的关系矩阵
user_matrix = np.zeros((num_users,num_users))
# 填充关系矩阵
for user1, user2 in user_combinations:
    user_matrix[user1, user2] = 1
    user_matrix[user2, user1] = 1
user_matrix = csr_matrix(user_matrix, dtype='int')
print("用户与用户 done")
###


"""testdata"""
test_data = []
# 读取test.csv文件
test_file_path = 'dataraw/city_test.csv'
test_df = pd.read_csv(test_file_path)
all_projects = [i for i in range(0,num_projects)]
# 提取用户、项目和类别，形成列表
test_Data = test_df[['user', 'venueid_id']].values.tolist()
n = len(test_Data)
item_num = num_projects
for i in range(n):
    u = test_Data[i][0]
    v = test_Data[i][1]
    test_data.append([u, v])
    # Negative Sample
    user_interacted_projects = train_df[train_df['user'] == u]['venueid_id'].unique()
    user_interacted_projects_test = test_df[test_df['user'] == u]['venueid_id'].unique()
    # 获取该用户未交互的项目（从all文件中随机选择99个）
    non_interacted_projects = np.setdiff1d(all_projects, user_interacted_projects, assume_unique=True)#找出了数组arr1中不同于数组arr2的元素,
    random_non_interacted_projects = np.random.choice(non_interacted_projects, 99, replace=True)
    for j in random_non_interacted_projects:
        test_data.append([u, j])

print("test data done")
data = (user_project_matrix, test_data, user_matrix, project_category_matrix, category_dict)
with open("data.pkl", 'wb') as fs:
    pickle.dump(data, fs)
print('Done')

