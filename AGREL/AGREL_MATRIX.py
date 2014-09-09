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
        
        self.w_i2h = matrix(random.uniform(-0.25,0.25,(self.input+1,self.hidden+1)))
        self.w_h2o = matrix(random.uniform(-0.25,0.25,(self.hidden+1,self.output)))
        
        
        #connect the bias units together but make sure there have weight 0 to not affect the calculations
        
        print 'w_i2h:', self.w_i2h
        print
        print 'w_h2o:', self.w_h2o

        self.ia = matrix(ones((1,self.input+1)))
        self.ha = matrix(zeros((1,self.hidden+1)))
        self.opr = matrix(zeros((self.output,1)))
        self.oa = matrix(zeros((self.output,1)))
        
        self.hnet = matrix(zeros((self.hidden,1)))
        self.onet = matrix(zeros((self.output,1)))
        
        self.global_error =0.0

        print
        print '====\\===='
        print

    def feedforward(self,inputs):


        #set each input pattern (-1 so that the last input stays as 1)
        self.ia[:,:-1]=inputs
        
        print 'ia:',self.ia
        print
        #feed input into the connection from input to hidden neurons
        self.hnet = dot(self.ia,self.w_i2h)
        self.ha=1.0/(1.0+exp(-self.hnet))
        #bias unit must always have one
        self.ha[-1,-1]=1.
      
        print 'ha:', self.ha
        print
        #feed input from hidden to output using a softmax to get the probability of choosing the winning unit
        self.onet=dot(self.ha,self.w_h2o)
        print 'onet:', self.onet
        self.opr = exp(self.onet)/sum(exp(self.onet))
        print
        print 'probabilities', self.opr
        #print 'sum:', sum(self.opr)
        
        print#selection wheel
        print 'roulette wheel selection'
        print
        
        max=0.0
        for i in range(len(self.opr.flat)):
            max+=self.opr.flat[i]

        r = random.uniform(0,max)
        c = 0.0
        print 'r:', r
        print
        index=0
        for i in range(len(self.opr.flat)):
            c+=self.opr.flat[i]
            if c > r:
                index=i
                break
        print 'selected index:', index
        print
               
            
        
        
        #get the index  and replace it with 1, all others with zero
        self.oa.flat[index]=1.
        print self.oa
        print
        return index
    

    def update(self,reward, index):

        #compute if there is a reward
        print 'oa:', self.oa.T , 'reward:', reward
        print
        print (self.oa.T == reward).all()
        if (self.oa.T == reward).all():
            print self.opr.flat[index]
            self.global_error= 1.-self.opr.flat[index]
        else:
            self.global_error = -1.
        print
        print 'global_error:', self.global_error
        
        #calculate the change in weights between hidden and output
        d_w_h2o=matrix(zeros((self.hidden+1,self.output)))
        d_w_h2o=(self.lr*multiply(self.ha,self.oa)*self.compute_delta()).T
        
        fb=matrix(zeros((self.hidden+1,1)))
        #calculate feedback for hidden to input
        fb=multiply((1.-self.ha),cumsum(dot(self.oa.T,self.w_h2o.T)))
        print
        print 'feedback:', fb
        print
        
        #calculate the change in weights between the input and hidden
        d_w_i2h=matrix(zeros((self.input+1,self.hidden+1)))
        d_w_i2h=multiply(self.lr*self.compute_delta()*(self.ia.T*self.ha),fb)
      

        #update the weights

        self.w_h2o=self.w_h2o+d_w_h2o
        self.w_i2h=self.w_i2h+d_w_i2h
        
    def compute_delta(self):
        if self.global_error >=0:
            return min(50./self.lr,(self.global_error/(1.-self.global_error)))
        else:
            return self.global_error

    def clear_oa(self):
         self.oa = matrix(zeros((self.output,1)))
         
def setup():
    pattern = matrix([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    reward = matrix([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
    
    input =2
    hidden =7
    output=2
    
    lr = 0.05
    num_epochs=5000
    index=0
    
    nn = ARGEL(input,hidden,output,lr)
    
    for i in range(num_epochs):
        for p in range(len(pattern)):
            index=nn.feedforward(pattern[p])
            nn.update(reward[p],index)
            nn.clear_oa()
        print 'epoch', i
        print

    for p in range(len(pattern)):
        nn.feedforward(pattern[p])
        nn.clear_oa()
        print
    
            
if __name__ == '__main__':
    setup()
                               
                               