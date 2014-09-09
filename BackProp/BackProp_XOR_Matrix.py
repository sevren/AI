import numpy
from numpy import *

import matplotlib.pyplot as plt

class ANN:
    def __init__ (self,inputs,hidden,output,lr,momentum):
        #set number of units
        self.input = inputs
        self.hidden = hidden
        self.output = output

        #set learning rate & momentum
        self.lr=lr
        self.momentum=momentum
        
        #initalize the weights
        #self.w_i2h =0.4*numpy.random.random_sample((self.hidden,self.input+1))-0.2
        #self.w_h2o=0.4*numpy.random.random_sample((self.output,self.hidden+1))-0.2
        numpy.random.seed()
        self.w_i2h = numpy.random.random((self.hidden,self.input+1))
        self.w_h2o =numpy.random.random((self.output,self.hidden+1))
        #self.w_i2h = numpy.random.randn(self.hidden,self.input+1)
        #self.w_h2o = numpy.random.randn(self.output,self.input+1)
        

        #set the net matrixes
        self.hnet = numpy.matrix(numpy.zeros(shape=(self.hidden,1)))
        self.onet = numpy.matrix(numpy.zeros(shape=(self.output,1)))
     
        #set activation matrixes & set the bias "1"
        self.ia=numpy.matrix(numpy.ones(shape=(self.input+1,1)))
        self.ha=numpy.matrix(numpy.ones(shape=(self.hidden+1,1)))
        self.oa=numpy.matrix(numpy.zeros(shape=(self.output,1)))
       
        self.w_i2h=numpy.asmatrix(self.w_i2h)
        self.w_h2o=numpy.asmatrix(self.w_h2o)
        
        self.hDelta = numpy.matrix(numpy.zeros(shape=(self.hidden)))
        self.oDelta = numpy.matrix(numpy.zeros(shape=(self.output)))
        
        
    def feed_forward(self,inputs):
        self.ia.A[:-1,0]=inputs
        self.hnet = numpy.dot(self.w_i2h,self.ia)
        #self.ha[:-1,:]=1.0/(1+numpy.exp(-self.hnet))
        self.ha[:-1,:]=numpy.tanh(self.hnet)
    

        self.onet=numpy.dot(self.w_h2o,self.ha)
        #self.oa=1.0/(1+numpy.exp(-self.onet))
        self.oa=numpy.tanh(self.onet)
       
    
    def backprop(self,target):
        error = target-self.oa
    
        self.oDelta = (1.0-numpy.multiply(numpy.tanh(self.onet),numpy.tanh(self.onet)))*error 
        self.hDelta=numpy.multiply((1.0-numpy.multiply(numpy.tanh(self.hnet),numpy.tanh(self.hnet))),numpy.dot(self.w_h2o[:,:-1].T,self.oDelta))

        #self.hDelta=numpy.multiply(numpy.multiply((1.0-numpy.tanh(self.hnet)),numpy.tanh(self.hnet)),numpy.dot(self.w_h2o[:,:-1].T,self.oDelta))
        self.w_i2h=self.w_i2h+self.lr*numpy.dot(self.hDelta,self.ia.T)
        self.w_h2o=self.w_h2o+self.lr*numpy.dot(self.oDelta,self.ha.T)

        return (target-self.oa)
        
    def Out(self):
        return self.oa
    
    #return the sum of the squared errors to see how the learning is going
    def Error(self,target):
        return 0.5*numpy.sum(target-self.oa)**2
      
    def set_i2h(self,i2h):
        self.w_i2h=i2h
    def set_h2o(self,h2o):
        self.w_h2o=h2o

    def returns_i2h(self):
        return self.w_i2h
    
    def returns_h2o(self):
        return self.w_h2o
    
    def print_weights(self):
        print self.w_i2h
        print
        print self.w_h2o

def setup():
    #encode the xor problem
    pattern = [ [0,0], [0,1], [1,0], [1,1]]
    target = [[0],[1],[1],[0]]
    input = 2
    hidden = 7
    output = 1
    eta =0.05
    alpha=0.00

    num_epochs=1000
    epochs = numpy.arange(num_epochs)
    best_rms =100000000000000000000

    best_w_i2h = numpy.zeros(shape=(input+1,hidden+1))
    best_w_h2o = numpy.zeros(shape=(hidden+1,output))
    

    fig =plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Error')
    ax.set_xlabel('Epochs')
    ax.set_title(' RMS Error vs Epochs -Learning rate: '+str(eta)+' momentum '+ str(alpha) )
    leg = ax.legend((hidden, 'Hidden units'))
    for run in range(0,10):
    #create an instance of the artificial neural network class
        nn =ANN(input,hidden,output,eta,alpha)
        
    #setup the required arrays for the plots
        error_plot=[0.0]*num_epochs       
        rms_error=0.0
    
    #train the network
        for i in range(num_epochs):
            error = 0.0
            for p in range (len(pattern)):
                nn.feed_forward(pattern[p])
                nn.backprop(target[p])
                error+= nn.Error(target[p])
                rms_error =numpy.sqrt(error/(len(pattern)*output))
                #rms_error = error
                #print 'error is: ', rms_error
            if rms_error < best_rms:
                best_rms = rms_error
                print 'best rms error is',  best_rms
                best_w_i2h = nn.returns_i2h()
                best_w_h2o = nn.returns_h2o()
                    #print 'best weights i2h'
                    #print best_w_i2h
                    #print
                    #print 'best weights h2o'
                    #print best_w_h2o
            error_plot[i]=rms_error

               
            
            #show the plots for the errors vs epochs
        
        ax.plot(epochs, error_plot)

                   
            
    plt.show()
    print
    nn2=ANN(input,hidden,output,eta,alpha)
    nn2.set_i2h(best_w_i2h)
    nn2.set_h2o(best_w_h2o)
    for i in range(len(pattern)):
        nn2.feed_forward(pattern[i])
        print pattern[i], '->', nn2.Out()[0]
        #if nn2.Out()[0] > 0.7: 
        #    print '1'
        #else:
        #    print '0'
            
        #print 'retest'
        #nn2.feed_forward(pattern[0])
        #print pattern[0], '->', nn2.Out()[0]
        #nn2.feed_forward(pattern[3])
        #print pattern[3], '->', nn2.Out()[0]
    #fig2=plt.figure(2)
    #ax = fig2.add_subplot(111)
    #ax.set_ylabel('Error')
    #ax.set_xlabel('Epochs')
    #ax.set_title('Error vs Epochs -Learning rate: '+str(eta)+'momentum '+ str(alpha))
    #plt.show()
#auto execute from python ann.py on the command line
if __name__ == '__main__':
    setup()
