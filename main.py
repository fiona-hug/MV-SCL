import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader  
import torch.optim as optim
import pickle
import random
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from ToolScripts.TimeLogger import log
from ToolScripts.BPRData import BPRData  
import ToolScripts.evaluate as evaluate

from model  import MODEL
from args  import make_args
print(t.cuda.device_count())

modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False



class Hope():
    def __init__(self, args, data, distanceMat, itemMat):
        self.args = args 
        self.userDistanceMat, self.itemDistanceMat, self.uiDistanceMat = distanceMat
        self.userMat = (self.userDistanceMat != 0) * 1
        self.itemMat = (itemMat != 0) * 1
        self.uiMat = (self.uiDistanceMat != 0) * 1  
       
        self.trainMat, testData, _, _, _ = data
        self.test_data_raw = testData
        self.userNum, self.itemNum = self.trainMat.shape
        train_coo = self.trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        assert np.sum(train_r == 0) == 0
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist()#将用户和项目组成两列
        test_data = testData
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, 1, True)
        test_dataset =  BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0) 
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=1024*1000, shuffle=False,num_workers=0)
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []
    
    def prepareModel(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        self.model = MODEL(
                           self.args,
                           self.userNum,
                           self.itemNum,
                           self.userMat,self.itemMat, self.uiMat,
                           self.args.hide_dim,
                           self.args.Layers).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def predictModel(self,user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1)
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    def adjust_learning_rate(self):
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)

    def getModelName(self):
        title = "SR-HAN" + "_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr +\
        "_hide_dim_" + str(self.args.hide_dim) +\
        "_lr_" + str(self.args.lr) +\
        "_reg_" + str(self.args.reg) +\
        "_topK_" + str(self.args.topk)+\
        "-ssl_ureg_" + str(self.args.ssl_ureg) +\
        "-ssl_ireg_" + str(self.args.ssl_ireg)
        return ModelName

    def saveHistory(self): 
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        ModelName = self.getModelName()
        with open(r'./History/' + self.args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self): 
        ModelName = self.getModelName()
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        savePath = r'./Model/' + self.args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.curEpoch,
            'args': self.args,
            'opt': self.opt,
            'history':history
            }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + self.args.dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']
        history = checkpoint['history']
        self.train_losses = history['loss']
        self.test_hr = history['hr']
        self.test_ndcg = history['ndcg']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))
    
    # Contrastive Learning
    def ssl_loss(self, data1, data2,   index):
        index=t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = t.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = t.exp(pos_score / self.args.ssl_temp)
        all_score  = t.sum(t.exp(all_score / self.args.ssl_temp), dim = 1)
        ssl_loss  = (-t.sum(t.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss

    def cosine_similarity(self,embeddings1, embeddings2):
        """
        计算两个嵌入矩阵的余弦相似度
        参数：
            embeddings1: torch.Tensor, 第一个嵌入矩阵，形状为(n, d)
            embeddings2: torch.Tensor, 第二个嵌入矩阵，形状为(n, d)

        返回值：
            torch.Tensor, 余弦相似度矩阵，形状为(n, n)
        """
        # 计算嵌入矩阵的范数
        norm1 = t.norm(embeddings1, dim=1, keepdim=True)
        norm2 = t.norm(embeddings2, dim=1, keepdim=True)
        # 计算点积
        dot_product = t.mm(embeddings1, embeddings2.t())
        # 计算余弦相似度
        distance = dot_product / (norm1 * norm2.t())

        # 计算欧氏距离
        # e1 = embeddings1.unsqueeze(1)  # 变为(n, 1, 32)
        # e2 = embeddings2.unsqueeze(0)  # 变为(1, n, 32)
        # 计算所有对的欧式距离
        # distance = t.sqrt(t.sum((e1 - e2) ** 2, dim=2))

        # 计算所有对的曼哈顿距离
        # e1 = embeddings1.unsqueeze(1)  # 变为(n, 1, 32)
        # e2 = embeddings2.unsqueeze(0)  # 变为(1, n, 32)
        # distance = t.sum(t.abs(e1 - e2), dim=2)

        #皮尔逊相关系数
        # mean1 = embeddings1.mean(dim=1, keepdim=True)
        # mean2 = embeddings2.mean(dim=1, keepdim=True)
        # adjusted_e1 = embeddings1 - mean1
        # adjusted_e2 = embeddings2 - mean2
        # numerator = t.mm(adjusted_e1, adjusted_e2.t())
        # denominator = t.sqrt(t.sum(adjusted_e1 ** 2, dim=1)[:, None] * t.sum(adjusted_e2 ** 2, dim=1))
        # distance = numerator / denominator

        # 马氏距离
        # mean_vector1 = t.mean(embeddings1, dim=0)
        # mean_vector2 = t.mean(embeddings2, dim=0)
        # # 计算中心化向量
        # centered_e1 = embeddings1 - mean_vector1
        # centered_e2 = embeddings2 - mean_vector2
        # # 计算两个样本集的协方差矩阵
        # covariance_matrix1 = t.matmul(centered_e1.t(), centered_e1) / (centered_e1.size(0) - 1)
        # covariance_matrix2 = t.matmul(centered_e2.t(), centered_e2) / (centered_e2.size(0) - 1)
        # # 计算协方差矩阵的平均值
        # average_covariance_matrix = (covariance_matrix1 + covariance_matrix2) / 2
        # # 计算马氏距离
        # distance = t.sqrt(t.diagonal(
        #     t.matmul(t.matmul(centered_e1, t.inverse(average_covariance_matrix)), centered_e2.t())))

        return distance


    def soft_loss(self,embeddings_1, embeddings_2, index):
        """
        计算软对比损失中的权重项

        参数:
            i: int, 第一个样本的索引
            i_prime: int, 第二个样本的索引
            alpha: float, 系数
            tau_I: float, 温度参数
            D: torch.Tensor, 样本之间的距离矩阵

        返回值:
            torch.Tensor, 权重值
        """
        index = t.unique(index)
        embeddings1 = embeddings_1[index]
        embeddings2 = embeddings_2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        distance = self.cosine_similarity(norm_embeddings1, norm_embeddings2)  # 获取样本之间的距离
        weight = 2 * self.args.alpha * t.exp(-self.args.tau_inst * distance) # 计算权重

        pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = t.mm(norm_embeddings1, norm_embeddings2.T)
        # 计算负样本得分，通过减去正样本得分获得
        neg_score = all_score - t.diag(pos_score)

        neg_score  =  weight * t.exp(neg_score / self.args.ssl_temp)
        all_score  = t.sum(t.exp(all_score / self.args.ssl_temp), dim = 1)
        ssl_loss_neg  = (-t.sum(t.log(neg_score / ((all_score))))/(len(index)*(len(index)-1)))
        soft_ssl_loss = ssl_loss_neg
        return soft_ssl_loss




    # Model train
    def trainModel(self):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample() ##训练数据的负抽样
        step_num = 0 # count batch num
        for user, item_i, item_j in self.train_loader:  #user用户，i真实数据，j负样本
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()  
            step_num += 1
            self.train= True
            itemindex = t.unique(t.cat((item_i, item_j)))#拼接后去除重复值
            userindex = t.unique(user)
            """这里分别是EMuu，EMii，EFu，EFi，Eu，Ei"""
            self.userEmbed, self.itemEmbed, self.ui_userEmbedall, self.ui_itemEmbedall, self.ui_userEmbed, self.ui_itemEmbed, metaregloss = self.model( self.train, userindex, itemindex, norm=1)
            
            # Contrastive Learning of collaborative relations
            ssl_loss_user = self.ssl_loss(self.ui_userEmbed, self.userEmbed, user)    
            ssl_loss_item = self.ssl_loss(self.ui_itemEmbed, self.itemEmbed, item_i)


            #soft constractive leaning
            soft_ssl_loss_u = self.soft_loss(self.ui_userEmbed, self.userEmbed, user)
            soft_ssl_loss_i = self.soft_loss(self.ui_itemEmbed, self.itemEmbed, item_i)

            ssl_loss = self.args.ssl_ureg * ssl_loss_user + self.args.ssl_ireg * ssl_loss_item+self.args.ssl_uneg * soft_ssl_loss_u+self.args.ssl_ineg * soft_ssl_loss_i
            # prediction
            pred_pos, pred_neg = self.predictModel(self.ui_userEmbedall[user],  self.ui_itemEmbedall[item_i],  self.ui_itemEmbedall[item_j])
            bpr_loss = - nn.LogSigmoid()(pred_pos - pred_neg).sum()  
            epoch_loss += bpr_loss.item()
            regLoss = (t.norm(self.ui_userEmbedall[user])**2 + t.norm( self.ui_itemEmbedall[item_i])**2 + t.norm( self.ui_itemEmbedall[item_j])**2) 
            loss = ((bpr_loss + regLoss * self.args.reg ) / self.args.batch) + ssl_loss*self.args.ssl_beta + metaregloss*self.args.metareg
            
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),  max_norm=20, norm_type=2)
            self.opt.step()
        return epoch_loss 
    """原文件中的"""
    # def testModel(self):
    #     HR=[]
    #     NDCG=[]
    #
    #     with t.no_grad():
    #         uid = np.arange(0,self.userNum)
    #         iid = np.arange(0,self.itemNum)
    #         self.train = False
    #         _,_, self.ui_userEmbed, self.ui_itemEmbed,_,_,_= self.model( self.train,uid,iid,norm=1)
    #         ###这是对test中HR和DCGN
    #         for test_u, test_i in self.test_loader:
    #             test_u = test_u.long().cuda()
    #             test_i = test_i.long().cuda()
    #             pred = self.predictModel( self.ui_userEmbed[test_u], self.ui_itemEmbed[test_i], None, isTest=True)
    #             batch = int(test_u.cpu().numpy().size/100)
    #             for i in range(batch):
    #                 # 从预测中提取批次得分
    #                 batch_socres=pred[i*100:(i+1)*100].view(-1)
    #                 # 获取具有最高分的前 k 个索引
    #                 _,indices=t.topk(batch_socres,self.args.topk)
    #                 # 从测试集中提取相应的项目
    #                 tmp_item_i=test_i[i*100:(i+1)*100]
    #                 # 基于索引取前 k 个项目
    #                 recommends=t.take(tmp_item_i,indices).cpu().numpy().tolist()
    #                 # 真实项目（批次中的第一个项目）
    #                 gt_item=tmp_item_i[0].item()
    #
    #                 HR.append(evaluate.hit(gt_item,recommends))
    #                 NDCG.append(evaluate.ndcg(gt_item,recommends))
    #     return np.mean(HR),np.mean(NDCG)

    """为了评价指标改的"""
    def testModel(self):
        HR=[]
        NDCG=[]
        rec_item_list = []
        rec_user_list = []
        true_interactions = []
        precisions = []
        with t.no_grad():
            uid = np.arange(0,self.userNum)
            iid = np.arange(0,self.itemNum)
            self.train = False
            _,_, self.ui_userEmbed, self.ui_itemEmbed,_,_,_= self.model(self.train,uid,iid,norm=1)

            """rec的评价指标"""
            for i in self.rec:
                rec_item_list += self.rec[i]
                rec_user_list += [i] * 100
                true_interactions.append(list([1]+[0]*99))
            test_u =t.tensor(rec_user_list).long().cuda()
            test_i =t.tensor(rec_item_list).long().cuda()
            pred = self.predictModel(self.ui_userEmbed[test_u], self.ui_itemEmbed[test_i], None, isTest=True)
            batch = int(test_u.cpu().numpy().size / 100)
            for i in range(batch):
                # 从预测中提取批次得分
                batch_socres = pred[i * 100:(i + 1) * 100].view(-1)
                # 获取具有最高分的前 k 个索引
                _, indices = t.topk(batch_socres, self.args.topk)
                # 从测试集中提取相应的项目
                tmp_item_i = test_i[i * 100:(i + 1) * 100]
                # 基于索引取前 k 个项目
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                _, indices_all = t.topk(batch_socres, 100)
                # 真实项目
                gt_item = tmp_item_i[0].item()

                HR.append(evaluate.hit(gt_item, recommends[:10]))
                NDCG.append(evaluate.ndcg(gt_item, recommends[:10]))

                precisions.append([len({gt_item}.intersection(recommends[: k])) / k for k in
                                       [1, 2, 3, 4, 5, 10, 20]])  ##.intersection返回多个集合（集合的数量大于等于2）的交集，
        return np.mean(HR),np.mean(NDCG),np.mean(np.array(precisions), axis=0)

    def rec(self):
        rec = {}
        for data in self.test_data_raw:
            if int(data[0]) not in rec:
                rec[(int)(data[0])] = [(int)(data[1])]
            else:
                if(len(rec[(int)(data[0])])==100):
                    continue
                rec[(int)(data[0])].append((int)(data[1]))
            if (len(rec[(int)(data[0])])==100 and len(rec)==100):
                break
        return rec

    def run(self):
        self.rec = self.rec()
        self.prepareModel()
        self.curEpoch = 0
        for e in range(args.epochs+1):
            self.curEpoch = e
            # train
            epoch_loss = self.trainModel()
            self.train_losses.append(epoch_loss)

            # test
            HR, NDCG, precisions = self.testModel()
            print(HR, '\t', NDCG, end='\t')
            for i in range(len(precisions)):
                if i == len(precisions) - 1:
                    print("%.4f" % precisions[i], end='\n')
                else:
                    print("%.4f" % precisions[i], end='\t')
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            self.adjust_learning_rate()
       
if __name__ == '__main__':
    # hyper parameters
    args = make_args()
    args.dataset = 'Yelp'
    print(args)

    # train & test data
    with open(r'dataset/'+args.dataset+'/data.pkl', 'rb') as fs:
        data = pickle.load(fs)
    with open(r'dataset/'+ args.dataset + '/distanceMat_addIUUI.pkl', 'rb') as fs:
        distanceMat = pickle.load(fs) 
    with open(r"dataset/" + args.dataset + "/ICI.pkl", "rb") as fs:
        itemMat = pickle.load(fs)

    # model instance
    hope = Hope(args, data, distanceMat, itemMat)
    modelName = hope.getModelName()
    print('ModelName = ' + modelName)    
    hope.run()
   

    

  

