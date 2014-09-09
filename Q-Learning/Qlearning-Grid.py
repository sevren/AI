import numpy as np
import sys
from pylab import plot, show, imshow, figure, cm

#RL-Algorithms
class Qlearning:
    def __init__(self,gamma,R,nactions,epsilon,alpha,lmbda,problem_type):
        self.nactions=nactions
        self.gamma=gamma
        self.epsilon=epsilon
        self.alpha=alpha
        self.lmbda=lmbda
        self.problem_type=problem_type
        
        #this is for continuous problems
        if self.problem_type=='Continuous':
            self.Q=np.zeros((R.shape[0]*R.shape[1],nactions))
            self.ET=np.zeros((R.shape[0]*R.shape[1],nactions))
        else:
            self.Q=np.zeros((nactions,R.shape[0],R.shape[1]))
            self.Q1=np.ones((nactions,R.shape[0],R.shape[1]))*np.inf
            

    def egreedy(self,currentState):
        if self.problem_type =='Continuous':
            if np.random.rand()<self.epsilon:
                return np.random.random_integers(0,self.nactions-1)
            else:
                return self.bestAction(currentState)
        else:
            if np.random.rand()>self.epsilon:
                return self.bestAction(currentState)
            else:
                return  np.random.random_integers(0,self.nactions-1)
            
    def egreedyAction(self,currentState,Epsilon):
        if Epsilon==0:
            return self.bestAction(currentState)

    #Returns the index of the best action
    def bestAction(self,currentState):
        if self.problem_type=='Continuous':
            #Qmax = Agent.Q[currentState,0]
            #bestAction=0
            
           # for a in xrange(1,self.nactions):
                #print currentState
           #     if Agent.Q[currentState,a] >Qmax:
           #         Qmax=Agent.Q[currentState,a]
           #         bestAction =a
           # return bestAction
            #print self.Q
            return np.where(self.Q[currentState,:]==np.max(self.Q[currentState,:]))[0][0]
        else:
            curRow=currentState[0]
            curCol=currentState[1]
            #now look at all the Q-values and choose the max one
            return np.where(self.Q[:,curRow,curCol]==np.max(self.Q[:,curRow,curCol]))[0][0]


    def update(self,state,action,r,qMax):
        if self.problem_type=='Continuous':
            self.Q[state,action] = self.Q[state,action]+self.alpha*r+self.gamma*qMax-self.Q[state,action]
        else:
            self.Q[action,state[0],state[1]] = self.Q[action,state[0],state[1]]+self.alpha*r+self.gamma*qMax -self.Q[action,state[0],state[1]]

#RL-Problems
class Grid(object):
    def __init__(self,sizex,sizey,nactions,noise, rewards):
        self.sizeRows=sizex
        self.sizeCols=sizey
        self.nactions=nactions
        self.noise=noise
        self.world=np.zeros((self.sizeRows,self.sizeCols))
        self.rewardSetup(rewards)
        self.start_state=[2,0]
        self.problem_type='Discrete'
        
    def rewardSetup(self,rewards):
        #rewards are always in the form [[row,col,value]] --may be multiple rewards in world
        for i in range(len(rewards)):
            self.world[rewards[i][0],rewards[i][1]]=rewards[i][2]
    
    def doAction(self,currentState,index):
        curRow=currentState[0]
        curCol=currentState[1]
    
        if index ==0:
        #go left
            curCol-=1
        elif index == 1:
        #go right
            curCol +=1
        elif index ==2:
        #go up
            curRow -=1
        elif index==3:
        #go down
            curRow+=1

    #check for boundries
        if curCol >3:
            curCol = 3
        if curCol <0:
            curCol=0
        if curRow >2:
            curRow=2
        if curRow <0:
            curRow =0
            
        #this is for the rock in the grid
        if curRow ==1 and curCol==1:
        #reset to the last known position because no movement can occur
            curRow=currentState[0]
            curCol=currentState[1]

        return [curRow,curCol]
    
    def giveReward(self,currentState):
        return self.world[currentState[0],currentState[1]]

    def reached_terminal_state(self,currentState):
        if (currentState[0] ==0 and currentState[1]==3) or (currentState[0]==1 and currentState[1]==3):
            return True
        else:
            return False

    def visual(self,state):
        sys.stdout.write(' # # # # # # # # # #\n')
        for i in range(self.sizeRows):
            sys.stdout.write(' # ')
            for j in range(self.sizeCols):
                if i==0 and j==3:
                    sys.stdout.write(' +100 ')
                elif i==1 and j==3:
                    sys.stdout.write(' -100 ')
                elif i==state[0] and j==state[1]:
                    sys.stdout.write(' @ ')
                elif i==1 and j==1:
                    sys.stdout.write(' # ')
                else:
                    sys.stdout.write(' 0 ')
            sys.stdout.write(' # \r\n')
        sys.stdout.write(' # # # # # # # # # #\n')

        
class windyGrid(Grid):
    def __init__(self,sizex,sizey,nactions,noise,rewards):
        super(windyGrid,self).__init__(sizex,sizey,nactions,noise,rewards)
        self.start_state=[4,0]
        self.wind =np.asarray(np.mat("0 0 0 0 0 0; 0 0 1 1 1 0; 0 0 1 1 1 0; 0 0 1 1 1 0; 0 0 1 1 0 0"))

    def doAction(self,currentState,index):
        curRow = currentState[0]
        curCol = currentState[1]
        if self.wind[curRow,curCol]==0:
            #no wind - normal movement
        
            if index == 0:
                #go left
                curCol-=1
            elif index ==1:
                #go right
                curCol+=1
            elif index ==2:
                #go up
                curRow -=1
            elif index==3:
                #go down
                curRow+=1
                
        else:
            #there is wind so special actions
            if index==0:
                #go up one and to the left
                curRow-=1
                curCol-=1
            elif index==1:
                #go up one and to the right
                curRow -=1
                curCol+=1
            elif index==2:
                #go two up
                curRow -=2
            elif index==3:
                pass #no movement down during wind
            
        #boundry check
        if curCol <0:
            curCol=0
        if curCol > self.sizeCols-1:
            curCol = self.sizeCols-1
        if curRow < 0:
            curRow=0
        if curRow > self.sizeRows-1:
            curRow = self.sizeRows-1
            
        return [curRow, curCol]

    def reached_terminal_state(self,currentState):
        if (currentState[0] ==4 and currentState[1]==4):
            return True
        else:
            return False
    def visual(self,state):

        sys.stdout.write(' # # # # # # # # # # #\n')
        for i in range(self.sizeRows):
            sys.stdout.write(' # ')
            for j in range(self.sizeCols):
                if i==4 and j==4:
                    sys.stdout.write(' G ')
                elif i==state[0] and j==state[1]:
                    sys.stdout.write(' @ ')
                elif (i==1 or i==2 or i==3 ) and (j==2 or j==3 or j==4):
                    sys.stdout.write(' w ')
                elif (i==4 and (j==2 or j==3)):
                    sys.stdout.write(' w ')
                else:
                    sys.stdout.write(' 0 ')
            sys.stdout.write(' # \r\n')
        sys.stdout.write(' # # # # # # # # # # #\n')

class MountainCar(object):
    def __init__(self):
        self.car_pos=-0.5
        self.car_vel=0.0
        self.goal_pos=0.45
        self.boundryl=-1.2
        self.boundryr=0.6
        self.boundrylspeed=-0.07
        self.boundryrspeed=0.07
        self.gravity = -0.0025
        self.hillFreq=3.0
        self.acceleration=0.001
        self.nactions=3
        self.start_state=[self.car_pos,self.car_vel]
        self.npartitions=30
        self.world=np.zeros((np.linspace(-1.2,0.6,self.npartitions).size,np.linspace(-0.07,0.07,self.npartitions).size))
        self.posrange = 0.6--1.2
        self.velrange=0.07--0.07
        self.posdiv=self.posrange/self.npartitions
        self.veldiv=self.velrange/self.npartitions
        self.problem_type='Continuous'
        
        
    def reached_terminal_state(self,currentState):
        if self.car_pos >0.3:
            print self.car_pos
        if self.car_pos >=self.goal_pos:
            print 'reached goal'
            return True
        else:
            False
            
    def giveReward(self,currentState):
        if currentState[0] <self.goal_pos:
            return -1
        else:
            return 100

    def doAction(self,currentState,action):
        currentPos=currentState[0]
        currentSpeed=currentState[1]
        if action==0:
            force=-1.
        elif action==1:
            force=0.
        elif action==2:
            force=1.
        
        delta_speed =currentSpeed +(self.acceleration*force)+(self.gravity *np.cos(self.hillFreq*currentPos))
        #check for boundries
        if delta_speed < self.boundrylspeed:
            delta_speed=self.boundrylspeed
        if delta_speed > self.boundryrspeed:
            delta_speed =self.boundryrspeed
            
            
        currentPos = currentPos+delta_speed
        currentSpeed =delta_speed
        if currentPos <=self.boundryl:
            currentPos=self.boundryl
            currentSpeed=0.0
        self.car_pos=currentPos
        self.car_vel=currentSpeed
        
        return [currentPos, currentSpeed]
    
    def getState(self,currentState):
        currentPos=currentState[0]
        currentSpeed=currentState[1]
        posIndex = int(np.abs(np.ceil(self.boundryl-np.abs(currentPos)/self.posdiv)))
        velIndex = int(np.abs(np.ceil(self.boundrylspeed-np.abs(currentSpeed)/self.veldiv)))
        
        stateIndex=int(self.npartitions*posIndex+velIndex)
        return stateIndex
        
    
    
def driver(Problem,Agent):
    if Problem.problem_type=='Continuous':
        #for mountain car the state space is divided into 30 equal paritions
        npartions=30
        episodes=5000
        steps=5000
        n_solved=0
        for episodes in xrange(episodes):
            print episodes
            if Agent.epsilon > 0.:
                if episodes %100 ==0:
                    Agent.epsilon =Agent.epsilon*Agent.epsilon
            #this descretizes the state
            s=Problem.start_state
            disc_s=Problem.getState(Problem.start_state)
            a= Agent.egreedy(disc_s)
            step=0
            while(not Problem.reached_terminal_state(s) and (step < steps)):
                #get and take action a and observe r, s'
                s_prime= Problem.doAction(s,a)
                disc_s_prime=Problem.getState(s_prime)
                #choose action a' from s'
                a_prime=Agent.egreedy(s_prime)
                a_star=Agent.egreedyAction(disc_s_prime,0)
                r=Problem.giveReward(s_prime)
    
                delta= r +Agent.gamma*Agent.Q[disc_s_prime,a_star]-Agent.Q[disc_s,a]
                Agent.ET[disc_s,a]=1 #replacing traces
                for s in range(900):
                    for a in range(Problem.nactions):
                        Agent.Q[s,a] =Agent.Q[s,a]+Agent.alpha*delta*Agent.ET[s,a]
                        if a_prime == a_star:
                            Agent.ET[s,a] = Agent.gamma*Agent.lmbda*Agent.ET[s,a]
                        else:
                            Agent.ET[s,a]=0
                s=s_prime
                a=a_prime
                step+=1
            if Problem.car_pos >=Problem.goal_pos:
                n_solved+=1
                print s
            if n_solved==50:
                break

        print 'set exploration to zero'
        print 'testing'
        Agent.epsilon=0.0
        s=Problem.start_state
        discS=Problem.getState(s)
        while(Problem.car_pos  <0.50):
            a=Agent.egreedyAction(discS,0)
            if a == 0:
                print 'reverse'
            elif a==1:
                print 'coast'
            elif a==2:
                print 'forward'
            s_prime=Problem.doAction(s,a)
            print Problem.car_pos
            s=s_prime
            discS=Problem.getState(s_prime)
            
                
                
    else:
        #inital state
        state=Problem.start_state
        #rain for 50000 episodes
        for episode in xrange(50000):
        #check if reached terminal state & give final reward and update
            if Problem.reached_terminal_state(state):
                r=Problem.giveReward(state)
                Agent.update(newState,action,r,qMax)
                #reset
                state=Problem.start_state
            #choose an action
            action=Agent.egreedy(state)
       
            #must be the max Q value for the next state over all possible actions
            newState=Problem.doAction(state,action)
            qMax= np.max(Agent.Q[:,newState[0],newState[1]])
            r=Problem.giveReward(state)
            Agent.update(state,action,r,qMax)
            state=newState
                
        #normalization of the Q-matrix    
        g=np.amax(Agent.Q)
        if g>0:
            Agent.Q=100*Agent.Q/g
            Agent.Q1=100*Agent.Q1/g
        print Agent.Q

        #Testing
        print 'Setting exploration to 0.0'
        Agent.epsilon=0.0
        print 'Optimal Path is:'
        print
        state=Problem.start_state
        while(Problem.reached_terminal_state(state)==False):
            Problem.visual(state)
            print
            action=Agent.bestAction(state)
            if action ==0:
                print 'Action taken: left'
            elif action ==1:
                print 'Action taken: right'
            elif action==2:
                print 'Action taken: up'
            elif action==3:
                print 'Action taken: down'
            newState=Problem.doAction(state,action)
            state=newState
        Problem.visual(state)
        
        
    
        
        
    
    
if __name__ == "__main__":
    #===Gridworlds====
    #setup the rewards for the Original Gridworld
    #rewards=[[0,3,100],[1,3,-100]]
    #Problem = Grid(3,4,4,0.0,rewards)
    #Agent=Qlearning(0.99,Problem.world,Problem.nactions,0.9,0.5,0.0,Problem.problem_type)
    #create windy grid world
    #rewards=[[4,4,100]]
    #Problem = windyGrid(5,6,4,0.0,rewards)   
    #Agent=Qlearning(0.85,Problem.world,Problem.nactions,0.9,0.5,0.0,Problem.problem_type)

    #===Mountain Car====

    #setup for the mountain car q-learning
    Problem=MountainCar()
    Agent=Qlearning(1.0,Problem.world,Problem.nactions,0.99999,0.1,0.0,Problem.problem_type)

    #===CartPole====

    #setup for the cart pole q-learning
    #Problem=CartPole()
    #Agent=Qlearning(0.99,Problem.world,Problem.nactions,0.9,0.5,0.0,Problem.problem_type)
    
    driver(Problem,Agent)