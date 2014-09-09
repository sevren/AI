from numpy import *
class ARGEL:
    def __init__(self, input,hidden,output,lr):
        #Number of nodes
        self.input=input
        self.hidden=hidden
        self.output=output
        
        #learning rate
        self.lr=lr

        #creating matrixes
        
        #self.w_i2h = random.uniform(-0.25,0.25,(self.input+1,self.hidden))
        #self.w_h2o = random.uniform(-0.25,0.25,(self.hidden+1,self.output))

        self.w_i2h = ([[-0.00565005, -0.02988793], [-0.00142538 , 0.21216198],[ 0.18549249, -0.24295674]])
        
        self.w_h2o=([[-0.13972243, 0.11203751],[-0.11530539,  0.0808334 ], [-0.06020249,  0.22309515]])


        
        print 'w_i2h:', self.w_i2h
        print
        print 'w_h2o:', self.w_h2o

        self.ia = zeros(self.input+1)
        self.ha = zeros(self.hidden+1)
        self.opr = zeros(self.output)
        self.oa = zeros(self.output)
        
        self.hnet = zeros((self.hidden,1))
        self.onet = zeros((self.output,1))
        
        self.global_error =0.0

        print
        print '====\\===='
        print

    def feedforward(self,inputs):

        #set each input pattern (-1 so that the last input stays as 1)
        for i in range(self.input):
            self.ia[i]=inputs[i]
        self.ia[self.input]=1.0

        print 'inputs:', self.ia
        print

        #calculate the hidden net values
        for j in range(self.hidden):
            sum=0.0
            for i in range(self.input+1):
                sum+=self.ia[i]*self.w_i2h[i][j]
            self.hnet[j]=sum
        
        print 'hnet:', self.hnet

        #calculate the hidden activation values
        for j in range(self.hidden):
            self.ha[j]=1.0/(1.0+exp(-self.hnet[j]))
        self.ha[self.hidden]=1.0
        
        print
        print 'ha:', self.ha
        print


        for j in range(self.output):
            sum=0.0
            for i in range(self.hidden+1):
                sum+=self.ha[i]*self.w_h2o[i][j]
            self.onet[j]=sum
            
        print 'onet:', self.onet

        ex_sum=0.0
        for i in range(self.output):
            ex_sum+=exp(self.onet[i])
    
        #calculate the output probabilities
        for j in range(self.output):
            self.opr[j]=exp(self.onet[j])/ex_sum

        print 'probabilities:', self.opr

        
        #selection wheel
        print 'roulette wheel selection'
        print
        
        max=0.0
        for i in range(len(self.opr)):
            max+=self.opr[i]

        

        r = random.uniform(0,max)
        c = 0.0
        print 'r:', r
        print
        index=0
        for i in range(len(self.opr)):
            c+=self.opr[i]
            if c > r:
                index=i
                break
        print 'selected index:', index
        print

        #get the index  and replace it with 1, all others with zero
        self.oa[index]=1.
        print self.oa
        print
        return index
    

    def update(self,reward, index):

        #compute if there is a reward
        print 'oa:', self.oa.T , 'reward:', reward
        print
        print (self.oa.T == reward)
        if (self.oa.T == reward).all():
            print self.opr.flat[index]
            self.global_error= 1.-self.opr.flat[index]
        else:
            self.global_error = -1.
        print
        print 'global_error:', self.global_error
        print

        #calculate change in weights for output to hidden
        d_w_h2o=zeros((self.hidden+1,self.output))
        for i in range(self.output):
            for j in range(self.hidden+1):
                d_w_h2o[j][i]=self.lr*self.ha[j]*self.oa[i]*self.compute_delta()
                
        print d_w_h2o

        fb=zeros(self.hidden+1)
        #calculate feedback for hidden to input
        for i in range(self.output):
            sum=0.0
            for j in range(self.hidden+1):
                sum+=self.oa[i]*self.w_h2o[j][i]
                fb[j]=(1.-self.ha[j])*sum
        print
        print 'feedback:', fb
        print
        
        #calculate change in weights for the hidden to the input
        d_w_i2h=zeros((self.input+1,self.hidden))
        for i in range(self.hidden):
            for j in range(self.input+1):
                d_w_i2h[j][i]=self.lr*self.ia[j]*self.ha[i]*self.compute_delta()*fb[i]
        
        print d_w_i2h
        print
        
        #updates weights

        for i in range(self.output):
            for j in range(self.hidden+1):
                self.w_h2o[j][i]+=d_w_h2o[j][i]
                
        for i in range(self.hidden):
            for j in range(self.input+1):
                self.w_i2h[j][i]+=d_w_i2h[j][i]
       
        
    def compute_delta(self):
        if self.global_error >=0:
            return min(50./self.lr,(self.global_error/(1.-self.global_error)))
        else:
            return self.global_error

    def clear_oa(self):
         self.oa = zeros(self.output)
         
def setup():
    pattern = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
    reward = [[1.,0.],[0.,1.],[0.,1.],[1.,0.]]
    
    input =2
    hidden =2

    output=2
    
    lr = 0.35
    num_epochs=1000
    index=0
    
    nn = ARGEL(input,hidden,output,lr)
    
    for i in range(num_epochs):
        for p in range(len(pattern)):
        #r= random.randint(0,2)
            index=nn.feedforward(pattern[p])
            nn.update(reward[p],index)
            nn.clear_oa()
        print 'epoch', i
        print

    print 'test'
    print

    for p in range(len(pattern)):
        nn.feedforward(pattern[p])
        #nn.update(reward[p])
        nn.clear_oa()
    #    print
    
            
if __name__ == '__main__':
    setup()