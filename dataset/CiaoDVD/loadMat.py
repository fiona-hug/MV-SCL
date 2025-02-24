# coding:UTF-8

import scipy.io as scio
import scipy.sparse as sp
import numpy as np
import random
import pickle
import datetime


# random.seed(123)
# np.random.seed(123)

def splitData(ratingMat, trustMat, categoryMat):
    # #filter user interaction number less than 2
    train_row, train_col, train_data = [], [], []
    test_row, test_col, test_data = [], [], []

    ratingMat = ratingMat.tocsr()
    userList = np.where(np.sum(ratingMat != 0, axis=1) >= 2)[0]

    for i in userList:
        uid = i
        tmp_data = ratingMat[i].toarray()

        _, iidList = np.where(tmp_data != 0)
        random.shuffle(iidList)  # shuffle
        test_num = 1
        train_num = len(iidList) - 1

        # positive data
        train_row += [i] * train_num
        train_col += list(iidList[:train_num])
        train_data += [1] * train_num

        test_row += [i] * test_num
        test_col += list(iidList[train_num:])
        test_data += [1] * test_num

        # negetive data
        # neg_iidList = np.where(np.sum(tmp_data==0))
        # neg_iidList = random.sample(list(neg_iidList),99)
        # test_row += i * 99
        # test_col += neg_iidList

    train = sp.csc_matrix((train_data, (train_row, train_col)), shape=ratingMat.shape)
    test = sp.csc_matrix((test_data, (test_row, test_col)), shape=ratingMat.shape)

    # print('train_num = %d, train rate = %.2f'%(train.nnz,train.nnz/ratingMat.nnz))
    # print('test_num = %d, test rate = %.2f'%(test.nnz,test.nnz/ratingMat.nnz))
    with open('./train.csv', 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open('./test.csv', 'wb') as fs:
        pickle.dump(test.tocsr(), fs)


def filterData(ratingMat, trustMat, categoryMat):
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    ratingMat = ratingMat.tocsr()
    trustMat = trustMat.tocsr()
    categoryMat = categoryMat.tocsr()

    a = np.sum(np.sum(train != 0, axis=1) == 0)
    b = np.sum(np.sum(train != 0, axis=0) == 0)
    c = np.sum(np.sum(trustMat, axis=1) == 0)
    while a != 0 or b != 0 or c != 0:
        if a != 0:
            idx, _ = np.where(np.sum(train != 0, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            trustMat = trustMat[idx][:, idx]
        elif b != 0:
            _, idx = np.where(np.sum(train != 0, axis=0) != 0)
            train = train[:, idx]
            test = test[:, idx]
            categoryMat = categoryMat[idx]
        elif c != 0:
            idx, _ = np.where(np.sum(trustMat, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            trustMat = trustMat[idx][:, idx]
        a = np.sum(np.sum(train != 0, axis=1) == 0)
        b = np.sum(np.sum(train != 0, axis=0) == 0)
        c = np.sum(np.sum(trustMat, axis=1) == 0)

    with open('./train.csv', 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open('./test.csv', 'wb') as fs:
        pickle.dump(test.tocsr(), fs)
    return ratingMat, trustMat, categoryMat


def splitAgain(ratingMat, trustMat, categoryMat):
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    train = train.tolil()
    test = test.tolil()

    idx = np.where(np.sum(test != 0, axis=1).A == 0)[0]  # A是array,matrix转换为array格式
    for i in idx:
        uid = i
        tmp_data = train[i].toarray()
        _, iidList = np.where(tmp_data != 0)
        sample_iid = random.sample(list(iidList), 1)
        test[uid, sample_iid] = 1
        train[uid, sample_iid] = 0

    with open('./train.csv', 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open('./test.csv', 'wb') as fs:
        pickle.dump(test.tocsr(), fs)


def testNegSample(ratingMat, trustMat, categoryMat):
    with open('./train.csv', 'rb') as fs:
        train = pickle.load(fs)
    with open('./test.csv', 'rb') as fs:
        test = pickle.load(fs)

    train = train.todok()
    test_u = test.tocoo().row
    test_v = test.tocoo().col
    test_data = []
    n = test_u.size
    for i in range(100):
        u = test_u[i]
        v = test_v[i]
        test_data.append([u, v])
        # Negative sample
        for t in range(99):
            j = np.random.randint(test.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(test.shape[1])
            test_data.append([u, j])

    with open('./test_Data.csv', 'wb') as fs:
        pickle.dump(test_data, fs)


if __name__ == '__main__':
    # with open('./train.csv', 'rb') as fs:
    #     train = pickle.load(fs)
    # with open('./test.csv', 'rb') as fs:
    #     test = pickle.load(fs)
    # with open('./test_Data.csv', 'rb') as fs:
    #     test_Data = pickle.load(fs)

    print(datetime.datetime.now())
    cv = 1
    # raw data
    ratingsMat = 'dataraw/rating.mat'
    trustMat = 'dataraw/trust.mat'

    # userid, productid, categoryid, rating, helpfulness
    # ratings = scio.loadmat(ratingsMat)['rating'][:22000]
    # column1 trusts column 2.
    ratings = scio.loadmat(ratingsMat)['rating']
    userNum = ratings[:, 0].max() + 1
    itemNum = ratings[:, 1].max() + 1
    trust = scio.loadmat(trustMat)['trustnetwork']
    # trust=[]
    # for data in trust1:
    #     if data[0]<=ratings[:, 0].max() and data[1]<=ratings[:, 0].max():
    #         trust.append(data)
    # trust = np.array(trust)
    # generate train data and test data
    trustMat = sp.dok_matrix((userNum, userNum))
    categoryMat = sp.dok_matrix((itemNum, 1))
    ratingMat = sp.dok_matrix((userNum, itemNum))

    ##generate ratingMat and categoryMat
    for i in range(ratings.shape[0]):
        data = ratings[i]
        uid = data[0]
        iid = data[1]
        typeid = data[2]
        categoryMat[iid, 0] = typeid
        ratingMat[uid, iid] = 1
    ##generate trust mat
    for i in range(trust.shape[0]):
        data = trust[i]
        trustid = data[0]
        trusteeid = data[1]
        trustMat[trustid, trusteeid] = 1

    splitData(ratingMat, trustMat, categoryMat)
    ratingMat, trustMat, categoryMat = filterData(ratingMat, trustMat, categoryMat)
    splitAgain(ratingMat, trustMat, categoryMat)
    ratingMat, trustMat, categoryMat = filterData(ratingMat, trustMat, categoryMat)
    testNegSample(ratingMat, trustMat, categoryMat)

    ##生成categoryDict
    categoryDict = {}
    categoryData = categoryMat.toarray().reshape(-1)
    for i in range(categoryData.size):
        iid = i
        typeid = categoryData[i]
        if typeid in categoryDict:
            categoryDict[typeid].append(iid)
        else:
            categoryDict[typeid] = [iid]

    print(datetime.datetime.now())

    with open('./train.csv', 'rb') as fs:
        trainMat = pickle.load(fs)
    with open('./test_Data.csv', 'rb') as fs:
        test_Data = pickle.load(fs)

    with open('trust.csv', 'wb') as fs:
        pickle.dump(trustMat, fs)

    data = (trainMat, test_Data, trustMat, categoryMat, categoryDict)

    with open("data.pkl", 'wb') as fs:
        pickle.dump(data, fs)

    print('Done')