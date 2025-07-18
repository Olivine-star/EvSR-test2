

class Metric(object):
    def __init__(self):
        self.reset()

    def updateIter(self, loss_time, loss_ecm, loss_polarity, loss_total, num, IS=0, OS=0, GS=0):
        self.LossTime += loss_time
        self.LossEcm += loss_ecm
        self.LossPolarity += loss_polarity
        self.Num += num
        self.Loss += loss_total
        self.IS += IS
        self.OS += OS
        self.GS += GS

    # def updateIter_p(self, loss_time, loss_ecm, loss_total, loss_polarity, num, IS=0, OS=0, GS=0):
    #     self.LossTime += loss_time
    #     self.LossEcm += loss_ecm
    #     self.LossPolarity += loss_polarity
    #     self.Loss += loss_total
        
    #     self.Num += num
    #     self.IS += IS
    #     self.OS += OS
    #     self.GS += GS



    def reset(self):
        self.LossTime = 0
        self.LossEcm = 0
        self.LossPolarity = 0.0  # 新增的极性损失
        self.Loss = 0
        
        self.Num = 0
        self.IS = 0
        self.OS = 0
        self.GS = 0

    def getAvg(self):
        avgLossTime = self.LossTime / self.Num
        avgLossEcm = self.LossEcm / self.Num
        avgLossPolarity = self.LossPolarity / self.Num
        avgLoss = self.Loss / self.Num
        avgIS = self.IS / self.Num
        avgOS = self.OS / self.Num
        avgGS = self.GS / self.Num

        return avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS





# class Metric(object):
#     def __init__(self):
#         self.reset()

#     def updateIter(self, loss_time, loss_ecm, loss_total, num, IS=0, OS=0, GS=0):
#         self.LossTime += loss_time
#         self.LossEcm += loss_ecm
#         self.Num += num
#         self.Loss += loss_total
#         self.IS += IS
#         self.OS += OS
#         self.GS += GS


#     def reset(self):
#         self.LossTime = 0
#         self.LossEcm = 0
#         self.Loss = 0
#         self.Num = 0
#         self.IS = 0
#         self.OS = 0
#         self.GS = 0

#     def getAvg(self):
#         avgLossTime = self.LossTime / self.Num
#         avgLossEcm = self.LossEcm / self.Num
#         avgLoss = self.Loss / self.Num
#         avgIS = self.IS / self.Num
#         avgOS = self.OS / self.Num
#         avgGS = self.GS / self.Num

#         return avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS