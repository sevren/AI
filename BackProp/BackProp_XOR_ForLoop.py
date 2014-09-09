import math
import random
import numpy
import numpy.matlib
import matplotlib.pyplot as plt

class ANN:
    def __init__(self,inputs,hidden,output,lr,momentum):
        #set the number of units
        self.input = inputs
        self.hidden = hidden
        self.output = output

        #set the learning rate
        self.learning_rate=lr
        #set the momenutm rate
        self.momentum = momentum

        #set the randomized weights (b-a)*random((rows,columns))-a to scale the random values
        #self.w_i2h = 0.4*numpy.random.random_sample((self.input,self.hidden))-0.2
        #self.w_h2o = 0.4*numpy.random.random_sample((self.hidden,self.output))-0.2
        
        self.w_i2h = numpy.random.randn(self.input,self.hidden)
        self.w_h2o=numpy.random.randn(self.hidden,self.output)

        
        #set the array's used for the activation values
        self.ia=numpy.ones(shape=(self.input))
        self.ha=numpy.ones(shape=(self.hidden))
        self.oa=numpy.ones(shape=(self.output))
        
        #hold previous weights(for calculating gradient descent with momentum)
        self.prev_w_i2h = numpy.zeros(shape=(self.input, self.hidden))
        self.prev_w_h2o = numpy.zeros(shape=(self.hidden, self.output))
        
    def feed_forward(self,inputs):
        
        #set the input activations to the inputs from the patterns
        for i in range(self.input):
            self.ia[i]=inputs[i]
        
        #First calculate net value of the input to the hidden
        #net is the sum of the weights * the previous activations
        #finally calculate the actual activation value for each hidden node using a transfer function. do not use 1/1+exp^net because there is a problem with the calculations use math.tanh instead.
        for h in range(self.hidden):
            net = 0.0
            for i in range(self.input):
                net +=self.w_i2h[i][h]*self.ia[i]
            self.ha[h]=numpy.tanh(net)
            #self.ha[h]=1.0/(1+numpy.exp(-net))
        #finally calculate the net value of the hidden to the output
        for o in range(self.output):
            net=0.0
            for h in range(self.hidden):
                net+=self.w_h2o[h][o]*self.ha[h]
            self.oa[o]=numpy.tanh(net)
            #self.oa[o]=1.0/(1+numpy.exp(-net))
            #print self.oa[o]
    
    def backprop(self,target):
        
        #compute the errors in the output units
        #target-actual * the derivative of the sigmoid transfer function
        Output_error = numpy.zeros(shape=(self.output))
        for o in range(self.output):
            Output_error[o] = (target[o]-self.oa[o])*(1-self.oa[o]*self.oa[o])
    
        #compute the errors in the hidden units using the output error from the output units
        Hidden_error = numpy.zeros(shape=(self.hidden))    
        for o in range(self.output):
            sum =0.0
            for h in range(self.hidden):
                sum +=(Output_error[o]*self.w_h2o[h][o])
                Hidden_error[h]=sum*(1-self.ha[h]*self.ha[h])
                
        #calculate the change in weight for hidden to output units
        #actually update the weights without using momentum term
        delta_h2o=numpy.zeros(shape=(self.hidden,self.output))
        for h in range(self.hidden):
            for o in range(self.output):
                delta_h2o[h][o]+=self.ha[h]*Output_error[o]
                self.w_h2o[h][o]=self.w_h2o[h][o]+self.learning_rate*delta_h2o[h][o] + self.momentum *self.prev_w_h2o[h][o]
                self.prev_w_h2o[h][o]=delta_h2o[h][o]

        #calculate the change in weight for the input to the hidden
        #actually updateh the weights without using momentum
        delta_i2h=numpy.zeros(shape=(self.input,self.hidden))
        for i in range(self.input):
            for h in range(self.hidden):
                delta_i2h[i][h]+=self.ia[i]*Hidden_error[h]
                self.w_i2h[i][h]=self.w_i2h[i][h]+self.learning_rate*delta_i2h[i][h] + self.momentum *self.prev_w_i2h[i][h]
                self.prev_w_i2h[i][h]=delta_i2h[i][h]

    #return the activation value for the output units
    def Out(self):
        return self.oa
    
    def print_weights(self):
        print self.w_i2h
        print
        print self.w_h2o
    
    #return the sum of the squared errors to see how the learning is going
    def Error(self,target):
        error =0.0
        for i in range(len(target)):
            print target[i]-self.oa[i]
            error+=0.5*(target[i]-self.oa[i])**2
        return error

#setup the problem
def setup():
    #encode the xor problem
    pattern = [ [0,0], [0,1], [1,0], [1,1]]
    target = [[0],[1],[1],[0]]
    
    #create an instance of the artificial neural network class
    nn =ANN(2,5,1,0.05,0.0)
    num_epochs=1000
  
    #setup the required arrays for the plots
    error_plot=[0.0]*num_epochs
    epochs = numpy.arange(num_epochs)
    rms_error=0.0

    #print weights
    nn.print_weights()
    print

    #train the network
    for i in range(num_epochs):
        error = 0.0
        for p in range (len(pattern)):    
        #rnd = random.randint(0,3)
            nn.feed_forward(pattern[p])
            nn.backprop(target[p])
            error+= nn.Error(target[p])
            rms_error =math.sqrt(error/4) 
            print 'error is: ', rms_error
            error_plot[i]=rms_error
    
    nn.print_weights()
    print
    #test the network
    for i in range(len(pattern)):
        nn.feed_forward(pattern[i])
        print pattern[i], '->', nn.Out()[0]
        #if nn.Out()[0] > 0.7: 
        #    print '1'
        #else:
        #    print '0'
   
    print 'retest'
    nn.feed_forward(pattern[0])
    print pattern[0], '->', nn.Out()[0]
    nn.feed_forward(pattern[3])
    print pattern[3], '->', nn.Out()[0]
            
    #show the plots for the errors vs epochs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, error_plot)
    plt.show()

#auto execute from python ann.py on the command line
if __name__ == '__main__':
    setup()
