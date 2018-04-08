import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        """
        关于这个技巧，我试过了，如果直接renorm(2,1,1)的话虽然大部分vector的norm会归一
        但是偶尔会有一些不稳定的情况发生，因为pytorch的设置是max_norm
        这里如果先归一到一个非常小的数值，再放大回来，就不会出现不稳定了
        """
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
        """
        这个是直接求cos(mx)的公式
        cos(2x) = 2*cos(x)**2 -1
        cos(3x) = 4*cos(x)**3 - 3*x
        ......
        """

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
        # xlen和wlen分别求得输入的每一个feature（每一行）和每一个class的Weight（每一列）的norm
        cos_theta = x.mm(ww) # size=(B,Classnum)
        #每一行每一列相乘
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        #除以每一行和每一列的norm，就是每一个feature和每一个class的weight之间的角度了
        cos_theta = cos_theta.clamp(-1,1) #clamp这一步我理解是为了数值稳定

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            #套用前面的公司Sun
            theta = Variable(cos_theta.data.acos())
            """
            arccos反余弦函数
            其实就是cos的反函数
            这里根据前面求得的cos值去取角度
            """
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1) #cos(原始的角度)*feature的norm
        phi_theta = phi_theta * xlen.view(-1,1) #cos(m*原始的角度)*feature的norm
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        # 留意这里这个it，后面有用到
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        #转成byte tensor，没转之前有-0,0,1这种值，转了之后就只有0,1了。适合做掩码
        index = Variable(index)
        # index指的是所有label标记出来的prediction,或者说所有预测对的输出feature

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum) # 这里乘个1.0有意义吗 ？
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        """
        终于开始套论文里公式7了,
        注意这里只需要在预测对的地方把cos_theta换成phi_theta，
        再跑softmax公式即可
        """
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
        """
        这个lamb的trick论文里没说啊，应该是个hacky way吧
        it代表iter的次数，效果是前3000次iter中国lamb的值缓慢地从1500减小到5
        这就意味着cos_theta不是完全替换成phi_theta,
        而是有个缓冲地逐渐替换，而且最多也就是替换了1/6
        """
        """
        不好意思，论文里有说，在appendix的最后一段G里面。。。。
        其实就是一个退火算法，
        在训练开始的时候，lamb很大，基本相当于原来的softmax在训练
        然后逐步地减小lamb，但是也不用太小，
        作者说一般减到5就可以有很好的效果了，再小的话训练就比较难了

        其实我觉得作者这样很鸡贼啊，因为如果lambda等于5的话，
        相当于你并没有把softmax给完全替换掉，跟论文通篇讲的调调是不一样的
        而且这样我甚至可以说你这只是给softmax加一个regularization嘛
        所以他才故意把这一段放在论文末尾吧

        The optimization of the A-Softmax loss is similar to the L-Softmax loss [16].
        We use an annealing optimization strategy
        to train the network with A-Softmax loss. 
        To be simple, the annealing strategy is essentially supervising the newtork from
        an easy task (i.e., large λ) gradually to a difficult task (i.e., small λ). 
        Specifically, we let 公式 and
        start the stochastic gradient descent initially with a very large λ
        (it is equivalent to optimizing the original softmax). 
        Then we gradually reduce λ during training. 
        Ideally λ can be gradually reduced to zero, 
        but in practice, a small value will usually suffice. 
        In most of our face experiments, 
        decaying λ to 5 has already lead to impressive results. 
        Smaller λ could potentially yield a better performance but is also more difficult to train.
        """
        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        # 咦，居然在这里撞见focal loss ?
        loss = loss.mean()

        return loss


class sphere20a(nn.Module):
    def __init__(self,classnum=10574,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = AngleLinear(512,self.classnum)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        if self.feature: return x

        x = self.fc6(x)
        return x
