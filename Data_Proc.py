import json
import time

import pandas as pd
import numpy as np
from collections import Counter
import tqdm
import os


def fd():
    path = ['Sports_and_Outdoors.csv', 'Electronics.csv', 'Toys_and_Games.csv', 'Home_and_Kitchen.csv',
            'Kindle_Store.csv', 'Software.csv', 'Tools_and_Home_Improvement.csv', 'AMAZON_FASHION.csv',
            'Clothing_Shoes_and_Jewelry.csv', 'Books.csv', 'Grocery_and_Gourmet_Food.csv',
            'Cell_Phones_and_Accessories.csv', 'Musical_Instruments.csv', 'Luxury_Beauty.csv', 'Office_Products.csv',
            'CDs_and_Vinyl.csv', 'Patio_Lawn_and_Garden.csv', 'Industrial_and_Scientific.csv', 'Prime_Pantry.csv',
            'Appliances.csv', 'Automotive.csv', 'Arts_Crafts_and_Sewing.csv', 'Video_Games.csv',
            'Magazine_Subscriptions.csv', 'Gift_Cards.csv','Movies_and_TV.csv']
    filePath = 'ratings-full'
    p = os.listdir(filePath)
    for it in p:
        if it not in path:
            continue
        user = []
        data = pd.read_csv(os.path.join(filePath, it), names=['item', 'user', 'rating', 'timestamp'],
                           usecols=[0, 1, 2, 3], chunksize=10000000,
                           engine='python')
        for i, dt in enumerate(data):
            user = np.concatenate((user, np.unique(dt['user'])), axis=0)
            user = np.unique(user)
        tb = pd.DataFrame(data=user)
        pth = os.path.join('domain_user', it)
        tb.to_csv(pth, index=None, header=None)


def fd_unk():
    filePath = 'domain_user'
    # p = os.listdir(filePath)
    p = ['Clothing_Shoes_and_Jewelry', 'Books', 'Movies_and_TV', 'CDs_and_Vinyl']
    users = []
    for it, vl in enumerate(p):
        data = pd.read_csv(os.path.join(filePath, vl + '.csv'), names=['user'], engine='python')
        users.append([data['user']])
    for it in range(4):

        data = users[it]
        if it == 0:
            commons = data
        else:
            commons = np.intersect1d(commons, data)

    tb = pd.DataFrame(data=commons)
    pth = os.path.join('overlaps', 'CD_BK.csv')
    tb.to_csv(pth, index=None, header=None)


def get_core():
    # 处理4个域的物品core值计算
    p = ['Clothing_Shoes_and_Jewelry.csv', 'Books.csv', 'Movies_and_TV.csv', 'CDs_and_Vinyl.csv']
    for it in p:
        data = pd.read_csv(os.path.join('ratings-full', it), names=['item', 'user', 'rating', 'timestamp'],
                           usecols=[0, 1, 2, 3], chunksize=10000000,
                           engine='python')
        items = []
        for i, dt in enumerate(data):
            items += list(dt['item'])
        item_counts = Counter(items)
        tb = pd.DataFrame.from_dict(item_counts, orient='index')
        tb.to_csv(os.path.join('item_core', it), header=None)
        print(f'finish {it}, {len(item_counts)} items')


def get_user_core():
    # 处理4个域的用户交互频率计算
    p = ['Clothing_Shoes_and_Jewelry.csv', 'Books.csv', 'Movies_and_TV.csv', 'CDs_and_Vinyl.csv']
    for it in p:
        data = pd.read_csv(os.path.join('ratings-full', it), names=['item', 'user', 'rating', 'timestamp'],
                           usecols=[0, 1, 2, 3], chunksize=10000000,
                           engine='python')
        users = []
        for i, dt in enumerate(data):
            users += list(dt['user'])
        user_counts = Counter(users)
        tb = pd.DataFrame.from_dict(user_counts, orient='index')
        tb.to_csv(os.path.join('user_core', it), header=None)
        print(f'finish {it}, {len(user_counts)} users')


def get_data():
    # p = ['Movies_and_TV.csv', 'CDs_and_Vinyl.csv', 'Books.csv', 'Home_and_Kitchen.csv', 'Electronics.csv', 'Clothing_Shoes_and_Jewelry.csv']
    p = ['Clothing_Shoes_and_Jewelry.csv', 'Books.csv', 'Movies_and_TV.csv', 'CDs_and_Vinyl.csv']
    # p = ['Tools_and_Home_Improvement.csv', 'Toys_and_Games.csv', 'Automotive.csv', 'Pet_Supplies.csv', 'Kindle_Store.csv', 'Office_Products.csv', 'Patio_Lawn_and_Garden.csv', 'Grocery_and_Gourmet_Food.csv']
    select_user = []
    select_item = []
    core = [48, 96, 32, 48]  # 对应4个域的用户core阈值
    # core = [24, 32, 32, 32, 48, 32, 32, 32]
    for i, d in enumerate(p):
        dt = pd.read_csv(os.path.join('user_core', d), names=['id', 'core'], header=None)
        user = dt.loc[dt['core'] >= core[i]]
        user = user['id']
        user = list(user)
        select_user.append(user)
    for i, d in enumerate(p):
        dt = pd.read_csv(os.path.join('item_core', d), names=['id', 'core'], header=None)
        item = dt.loc[dt['core'] >= 10]
        item = item['id']
        item = list(item)
        select_item.append(item)
    all_user = []
    for i, user in enumerate(select_user):
        all_user.extend(user)
        all_user = list(set(all_user))
    tb = pd.DataFrame(data=all_user)
    tb.to_csv(os.path.join('core16', '8user.csv'), header=None, index=None)
    for j, it in enumerate(p):
        user = []
        data = pd.read_csv(os.path.join('ratings-full', it), names=['item', 'user', 'rating', 'timestamp'],
                           usecols=[0, 1, 2, 3], chunksize=10000000,
                           engine='python')
        for i, dt in enumerate(data):
            pos = dt.loc[dt['user'].isin(select_user[j])]
            pos = pos.loc[dt['item'].isin(select_item[j])]
            tb = pd.DataFrame(data=pos.values)
            user.extend(list(tb.iloc[:, 0]))
            user = list(np.unique(user))
            pth = os.path.join('processed_data', it.split('.')[0])
            if not os.path.exists(pth): os.mkdir(pth)
            pth = os.path.join(pth, str(i) + it)
            tb.to_csv(pth, index=None, header=None)
            print(f'finish {str(i) + it}')
        print(len(user))


def union():
    p = 'processed_data'
    #p = 'processed_overlap_data'
    dir = ['Clothing_Shoes_and_Jewelry', 'Books', 'Movies_and_TV', 'CDs_and_Vinyl']
    # dir = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Movies_and_TV', 'CDs_and_Vinyl']
    # dir = ['Tools_and_Home_Improvement', 'Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products', 'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
    for it in dir:
        files = os.listdir(os.path.join(p, it))
        ratings = []
        dir_path = os.path.join(p, it)
        for f in files:
            pth = os.path.join(dir_path, f)
            data = pd.read_csv(pth, names=['item', 'user', 'rating', 'timestamp'], engine='python', header=None)
            ratings.append(data.values)
        ratings = np.vstack(ratings)
        ratings = pd.DataFrame(ratings)
        ratings.to_csv(os.path.join(dir_path, it + '.csv'), header=None, index=None)


def hash_id():
    # path = 'core16'
    # files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Tools_and_Home_Improvement', 'CDs_and_Vinyl',
    #          'Movies_and_TV', 'Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products', 'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
    # files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors',
    #          'Cell_Phones_and_Accessories', 'Movies_and_TV', 'CDs_and_Vinyl']
    #path = 'processed_overlap_data'
    path = 'processed_data'
    files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Movies_and_TV', 'CDs_and_Vinyl']
    data = []
    user_path = os.path.join('core16', '8user.csv')
    users = pd.read_csv(user_path, header=None, names=['id'])
    # user_8 = pd.read_csv(user_path, header=None, names=['id'])
    # user_path = os.path.join(os.path.join(path, 'hash-16'), '16user.csv')
    # user_16 = pd.read_csv(user_path, header=None, names=['id'])

    # all_user = list(user_8['id']) + list(user_16['id'])
    # all_user = list(set(all_user))
    all_user = list(users['id'])
    user_dic = dict(zip(all_user, [i for i in range(len(all_user))]))
    for i, it in enumerate(files):
        dt = pd.read_csv(os.path.join(os.path.join(path, it), it + '.csv'), header=None,
                         names=['item', 'user', 'rating', 'timestamp'],
                         engine='python')
        items = np.unique(dt['item'].values)
        item_dic = dict(zip(items, [i for i in range(len(items))]))
        ratings = []
        for vl in dt.values:
            ratings.append([item_dic[vl[0]], user_dic[vl[1]], vl[2], vl[3]])
        dt = pd.DataFrame(ratings)
        p = os.path.join('core16', 'hash-8')  # 修正输出目录，与split_data()的输入目录一致
        if not os.path.exists(p):
            os.makedirs(p)
        dt.to_csv(os.path.join(p, it + '.csv'), header=None, index=None)
        print(f'finish {it}')


def split_data(file):
    print(file)
    path = 'core16/hash-8'
    # files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Tools_and_Home_Improvement', 'CDs_and_Vinyl',
    #          'Movies_and_TV', 'Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products', 'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
    all_user = []
    for i, it in enumerate(file):
        dt = pd.read_csv(os.path.join(path, it + '.csv'), header=None, names=['item', 'user', 'rating', 'timestamp'],
                         engine='python')
        items = np.unique(dt['item'].values)
        item_dic = dict(zip(items, [i for i in range(len(items))]))
        users = np.unique(dt['user'].values)
        user_dic = dict(zip(users, [i for i in range(len(users))]))
        all_user.extend(list(users))
        all_user = list(set(all_user))
        user_data = {}
        for u in tqdm.tqdm(users):
            u = int(u)
            rt = dt.loc[dt['user'] == u]
            test_rt = rt.loc[rt['timestamp'].idxmax()]
            T, I = test_rt['timestamp'], test_rt['item']
            train_rt = rt.loc[rt['item'] != I]['item']
            user_data[u] = [list(train_rt.map(item_dic)), item_dic[I]]
        user_data['item_num'] = len(items)
        user_data['user_num'] = len(users)
        pth = os.path.join(os.path.join('core16', 'split-8'), it.split('.')[0] + '.json')
        with open(pth, 'w') as f:
            json.dump(user_data, f)
        try:
            print(file[i] + 'finished')
        except:
            continue
    # with open(os.path.join(os.path.join(path, 'split-16'), 'all.csv'), 'w') as f:
    #     json.dump(all_user, f)

def negative_sample():
    path = 'core16/split-8'  # 修正为与split_data()输出目录一致
    # files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Tools_and_Home_Improvement', 'CDs_and_Vinyl',
    #          'Movies_and_TV', 'Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products', 'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
    # another_name = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Tools', 'CDs', 'Movies', 'Toys', 'Automotive', 'Pet', 'Kindle', 'Office', 'Patio', 'Grocery']
    # files = ['Movies_and_TV', 'Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products',
    #          'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
    # another_name = ['Movies', 'Toys', 'Automotive', 'Pet', 'Kindle', 'Office', 'Patio', 'Grocery']
    files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Movies_and_TV', 'CDs_and_Vinyl']
    another_name = ['Clothing', 'Books', 'Movies', 'CDs']
    # files = ['Clothing_Shoes_and_Jewelry', 'Books','Movies_and_TV','CDs_and_Vinyl']
    # another_name = ['Clothing', 'Books', 'Movies', 'CDs']
    all_user = []
    for it in files:
        with open(os.path.join(path, it + '.json')) as f:
            data = json.load(f)
            all_user.extend(data.keys())
            all_user = list(set(all_user))
            f.close()
    all_dic = dict(zip(all_user, [i for i in range(len(all_user))]))
    # user_dicts = {i: {} for i in range(len(all_user))}
    user_dicts = [{} for i in range(len(all_user))]
    user_train_data = [[[] for j in range(4)] for i in range(len(all_user))]
    server_test_data = [[] for _ in range(4)]
    num_items = []
    num_users = []
    for i, it in enumerate(files):
        with open(os.path.join(path, it + '.json')) as f:
            data = json.load(f)
            item_num = data['item_num']
            user_num = data['user_num']
            num_items.append(item_num)
            num_users.append(user_num)
            f.close()
        gen = []
        total_num = 99
        cnt = 0
        for (id, rating) in data.items():
            if id == 'item_num' or id == 'user_num':
                continue
            all_id = all_dic[id]
            user_dicts[all_id][another_name[i]] = cnt
            tot = rating[0] + [rating[1]]
            while len(gen) < total_num:
                negative = np.random.randint(0, item_num)
                if negative not in tot and negative not in gen:
                    gen.append(negative)
            for ng in gen:
                server_test_data[i].append([cnt, ng, 0])
            server_test_data[i].append([cnt, rating[1], 1])
            user_train_data[all_id][i] = rating[0]
            cnt += 1
    save_dic = dict(zip(['client_train_data', 'server_evaluate_data', 'num_items', 'num_users', 'user_dic'],
                        [user_train_data, server_test_data, num_items, num_users, user_dicts]))
    with open('core16/split-8/implicit.json', 'w') as f:  # 55518
        json.dump(save_dic, f)
    print('finish')

def get_central():
    with open('core16/split-8/implicit.json', 'r') as f:
        d = json.load(f)
        f.close()
    train_data = [[], []]
    for i in range(len(d['user_dic'])):
        domains = list(d['user_dic'][i].keys())
        data = d['client_train_data'][i]
        if 'Books' in domains:
            for it in data[1]:
                train_data[0].append([d['user_dic'][i]['Books'], it])
        if 'CDs' in domains:
            for it in data[3]:
                train_data[1].append([d['user_dic'][i]['CDs'], it])
    test_data = [d['server_evaluate_data'][1], d['server_evaluate_data'][3]]
    save_dic = dict(zip(['server_train_data', 'server_evaluate_data', 'num_items', 'num_users', 'user_dic'],
                        [train_data, test_data, d['num_items'], d['num_users'], d['user_dic']]))
    with open('core16/split-8/implicit_central.json', 'w') as f:
        json.dump(save_dic, f)
    print('finish')


def get_overlap_data():
    # 处理重叠用户数据
    overlap_users = pd.read_csv('overlaps/CD_BK.csv', header=None, names=['user'])
    overlap_users = overlap_users['user'].tolist()

    # 筛选重叠用户的数据
    domains = ['Books.csv', 'CDs_and_Vinyl.csv']
    processed_overlap = {}

    for domain in domains:
        data = pd.read_csv(os.path.join('ratings-full', domain), names=['item', 'user', 'rating', 'timestamp'],
                          usecols=[0, 1, 2, 3], chunksize=10000000, engine='python')

        overlap_data = []
        for chunk in data:
            # 筛选重叠用户
            filtered_chunk = chunk[chunk['user'].isin(overlap_users)]
            if not filtered_chunk.empty:
                overlap_data.append(filtered_chunk)

        if overlap_data:
            combined_data = pd.concat(overlap_data, ignore_index=True)
            domain_name = domain.split('.')[0]
            processed_overlap[domain_name] = combined_data

            # 保存处理后的重叠数据
            output_dir = os.path.join('processed_overlap_data', domain_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            combined_data.to_csv(os.path.join(output_dir, domain_name + '.csv'), header=None, index=None)

    print(f'Processed overlap data for {len(processed_overlap)} domains')
    return processed_overlap


def split_overlap_data():
    # 分割重叠数据的训练和测试集
    domains = ['Books', 'CDs_and_Vinyl']
    overlap_data = {}

    for domain in domains:
        data_path = os.path.join('processed_overlap_data', domain, domain + '.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, header=None, names=['item', 'user', 'rating', 'timestamp'])

            # 为每个用户分割数据
            user_data = {}
            users = data['user'].unique()

            for user in users:
                user_ratings = data[data['user'] == user].sort_values('timestamp')
                if len(user_ratings) > 1:
                    # 最新的交互作为测试，其余作为训练
                    test_item = user_ratings.iloc[-1]['item']
                    train_items = user_ratings.iloc[:-1]['item'].tolist()
                    user_data[int(user)] = [train_items, test_item]

            overlap_data[domain] = user_data
            print(f'Split data for {domain}: {len(user_data)} users')
        else:
            print(f'Warning: {data_path} not found')

    return overlap_data


def overlap_negative_sample():
    # 为重叠数据生成负样本
    overlap_data = split_overlap_data()
    domains = ['Books', 'CDs_and_Vinyl']
    domain_short = ['Books', 'CDs']

    # 获取所有物品ID
    all_items = {}
    for i, domain in enumerate(domains):
        data_path = os.path.join('processed_overlap_data', domain, domain + '.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, header=None, names=['item', 'user', 'rating', 'timestamp'])
            all_items[domain_short[i]] = data['item'].unique().tolist()

    # 生成负样本
    overlap_samples = {}
    for domain in domain_short:
        if domain in overlap_data and domain in all_items:
            samples = []
            domain_data = overlap_data[domain]
            item_list = all_items[domain]
            num_items = len(item_list)
            item_set = set(item_list)

            for user_id, (train_items, test_item) in domain_data.items():
                # 为测试物品生成负样本
                user_items = set(train_items + [test_item])
                negative_items = []

                while len(negative_items) < 99:  # 生成99个负样本
                    neg_item = np.random.choice(item_list)
                    if neg_item not in user_items and neg_item not in negative_items:
                        negative_items.append(neg_item)

                # 添加正样本和负样本
                samples.append([user_id, test_item, 1])  # 正样本
                for neg_item in negative_items:
                    samples.append([user_id, neg_item, 0])  # 负样本

            overlap_samples[domain] = samples
            print(f'Generated {len(samples)} samples for {domain} domain')

    # 保存重叠数据的负样本
    with open('overlap_negative_samples.json', 'w') as f:
        json.dump(overlap_samples, f)

    return overlap_samples


def get_domain_user():
    another_name = ['Clothing', 'Books', 'Movies', 'CDs']
    # another_name = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Tools', 'CDs', 'Movies', 'Toys',
    #                 'Automotive', 'Pet', 'Kindle', 'Office', 'Patio', 'Grocery']
    # another_name = ['Clothing', 'Books', 'Movies', 'CDs']
    with open('core16/split-8/implicit.json', 'r') as f:
        d = json.load(f)
        f.close()
    domain_user = {it: [] for it in another_name}

    for i in range(len(d['user_dic'])):
        domains = list(d['user_dic'][i].keys())
        for it in another_name:
            if it in domains:
                domain_user[it].append(i)
    with open('core16/split-8/domain_user.json', 'w') as f:
        json.dump(domain_user, f)
        f.close()

#fd()
#get_core()      # 计算物品core值
#get_user_core() # 计算用户core值
#get_data()      # 筛选数据并生成数据块
#union()         # 合并数据块
#hash_id()       # ID重编码 → core16/hash-8/
#split_data(['Clothing_Shoes_and_Jewelry', 'Books', 'Movies_and_TV', 'CDs_and_Vinyl'])  # 分割数据
#negative_sample()
get_domain_user()
# split_overlap_data()
# overlap_negative_sample()
#get_domain_user()


# # async split


# files = ['Clothing_Shoes_and_Jewelry', 'Books', 'Home_and_Kitchen', 'Electronics', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Movies_and_TV', 'CDs_and_Vinyl',
#               'Tools_and_Home_Improvement','Toys_and_Games', 'Automotive', 'Pet_Supplies', 'Kindle_Store', 'Office_Products', 'Patio_Lawn_and_Garden', 'Grocery_and_Gourmet_Food']
# split_data(files[:7])

