#Gowtham Rang
from itertools import product
from math import log
from collections import defaultdict
from random import shuffle


#Bayesian Nets
#describe the graph model as a dict and initialize test and train data sets and order
#automatically factorizes and constructs CPTs and infers HD/ computes Joints and 

################################################################################################################
class BayesNets:
    
    def __init__(self,
        trainfile='data-train-1.txt',
        testfile='data-test-1.txt',
        relationship={'A':['CH','HR'],'G':['BP','CH'],'BP':['HD'],'CH':['HD'],'HD':['HR','ECG','CP','EIA'],'CP':[],'EIA':[],'ECG': [], 'HR':[]}
        ,dataorder=['A','G','CP','BP','CH','ECG','HR','EIA','HD']
        ):

        self.relationship = relationship
        self.dataorder = dataorder
        self._factorize()
        self.bigtable = defaultdict(float);
        self.bigtable_order = dict(zip(self.dataorder,range(len(self.dataorder))))
        self.trainfile = trainfile
        self.testfile = testfile
        return;

    #First is always a non conditional
    # P(B,A) = P(B)* P(A|B) 
    #_factorise returns [['B'],['A','B']]
    def _factorize(self):   
        self.factor = self.topo_order()
        self.HD_infer_mask = [-1]*len(self.dataorder)
        r = dict(zip(self.dataorder,range(len(self.dataorder))))
        for each in self.factor:
            if 'HD' in each:
                for every in each:
                    self.HD_infer_mask[r[every]] = 1
        #print 'hd mask,dataorder', zip(self.HD_infer_mask,self.dataorder)
        return
        
    def _getfulltable(self):
        f = open(self.trainfile)
        for each in f.readlines():
            tup = ()
            temp = {}            
            for every in each.strip().split(','): tup = tup+ (float(every),)
            self.bigtable[tup]+=1
        f.close()
        return;
        
    def  topo_order(self):
        #assumes no cycles
        self.parents = defaultdict(lambda : [])
        marked ={}
        for each in self.relationship:
            self.parents[each]
            marked[each] = 0
            for eve in self.relationship[each]: 
                marked[eve] = 0
                self.parents[eve].append(each)
        explored = []
        
        pkey = self.parents.keys()
        shuffle(pkey) # to ensure correct implementation
        for each in pkey:
            if marked[each] ==1:
                continue
            # nil parent
            lastmarked = []    
            marked[each] = 1
            lastmarked.append(each)
            while len(lastmarked)>0:
                node = lastmarked[-1]
                expl = True
                for each in self.relationship[node]:
                    if marked[each] == 0:
                        expl = False
                        marked[each] = 1
                        lastmarked.append(each)
                if expl:
                    explored.append(lastmarked.pop())
        #print 'explored set %s', explored
        res = []
        for each in reversed(explored):
            l = [each]         
            l.extend(self.parents[each])
            res.append(l)
        return res
    
    def _learncpt(self):
        self.parameters = []
        for eachfactor in self.factor:
            self.parameters.append(defaultdict(float))
            for every in self.bigtable:
                res = ()
                for randomvar in eachfactor:
                    res = res + (every[self.bigtable_order[randomvar]],)
                    
                self.parameters[-1][res]+=self.bigtable[every]
        #parameter
        #Summing out the variations
        for eachtable in self.parameters:
            temp = defaultdict(float)
            for each in eachtable:
                temp[each[1:]] +=eachtable[each]
            
            for each in eachtable:
                #print eachtable[each], each, temp[each[1:]]
                eachtable[each]/=temp[each[1:]]
    #max likely learn                
    def learn(self):
        self._getfulltable()
        self._learncpt()
        assert(len(self.parameters) == len(self.factor))
        return;
        
    def get_accuracy(self):
        print 'reading test file %s' %self.testfile        
        f = open(self.testfile)
        pred = []
        gold = []
        for each in f.readlines():
            tup = []            
            #DATA ORDER IS ASSUMED
            for every in each.strip().split(','):
                tup.append(float(every))
            pred.append(self.infer_HD(tuple(tup[:-1])))
            gold.append(tup[-1])
            
        f.close()  
        #get accuracy
        
        correct = 0.0
        for p,g in  zip(pred,gold):
            if p==g:
                correct+=1
        return correct/len(gold)

    #infers HD given all others
    #HD is the last in dataorder
    def infer_HD(self,value):
        #P(OTHERS,HD)
        #Using marginals
        #self.dataorder = [-1,-1,'CP','BP','CH','ECG','HR','EIA','HD']
        value = list(value);
        newvalue = []
        
        for x,y in zip(self.HD_infer_mask,value):
            if x == 1:
                newvalue.append(y)
            else:
                newvalue.append(-1)
        newvalue.append(2)        
        
        
        hd_2 = self.marginal(tuple(newvalue))
        newvalue[-1]=1        
        hd_1 = self.marginal(tuple(newvalue))

        #verifying inference with reduction in terms and without 
        newvalue = tuple(value) + (2,)
        hd_check2 = self.jointinfer(newvalue)
        newvalue = tuple(value) + (1,)
        hd_check1 = self.jointinfer(newvalue)
        #print hd_check2
        #print '---v---',hd_check2*1.0/(hd_check2+hd_check1),hd_2*1.0/(hd_2+hd_1)
        if hd_2>hd_1: 
            assert(hd_check2>hd_check1)
            
        else:
            assert(hd_check2<=hd_check1)
        return 2 if hd_check2>hd_check1 else 1
        
    def marginal(self,value):
        from math import log,e
        #print '----------marginals value-----------'
        s = 1.0
        r = dict(zip(self.dataorder,value))
        for eachfactor,cpt in zip(self.factor,self.parameters):
            var = ()
            gogo = []
            chk  = True
            for every in eachfactor:
                if r[every] == -1:
                    chk = False
                    break;
                var = var+ (r[every],)
                gogo.append(every)
            if not chk:
                continue

            #print cpt[var], gogo, var, s
            #s+=log(cpt[var])
            s *= cpt[var]
        #print '----------marginal value ends ------- ', s
        return s

        
    #value is in dataorder
    def jointinfer(self,value):
        from math import log,e
        #print '----------value-----------'
        s = log(1.0)
        #s = 1.0
        r = dict(zip(self.dataorder,value))
        for eachfactor,cpt in zip(self.factor,self.parameters):
            var = ()
            gogo = []
            for every in eachfactor:
                var = var+ (r[every],)
                gogo.append(every)
            #print cpt[var], gogo, var, s
            s+=log(cpt[var])
            #s *= cpt[var]
        #print '----------value ends ------- ', s
        
        return e**s;

def print_cpt(key,factor, cpt):
    for i,each in enumerate(factor):
        if each == key:
            print key
            for every in cpt[i]:
                print every, cpt[i][every]

################################################################################################################
m = BayesNets()
m.learn()

# #vals need to be in data-order
x = [(1,2,3),(1,2),(1,2,3,4),(1,2),(1,2),(1,2),(1,2),(1,2),(1,2)]

a=[]
for each in product(*x):
    a.append(m.jointinfer(each))
assert(sum(a)-1.0 <0.001)

#self.factor = [['A'],['G'],['CH','G','A'],['BP','G'],
#        ['HD','CH','BP'],['HR','HD','A'],['ECG','HD'],['CP','HD'],['EIA','HD']]

print '------Table factors ---'
print m.factor
print 'A '
print_cpt(['A'],m.factor,m.parameters)
print 'BP|G'
print_cpt(['BP','G'],m.factor,m.parameters)
print 'HD|BP,CH'
print_cpt(['HD','BP','CH'],m.factor,m.parameters)
print '-----------------'
print_cpt(['HD','CH','BP'],m.factor,m.parameters)
print 'HR|HD,A'
print_cpt(['HR','A','HD'],m.factor,m.parameters)
print '-----------------'
print_cpt(['HR','HD','A'],m.factor,m.parameters)
print '-----------------------------'
print '5.a'

#self.dataorder = ['A','G','CP','BP','CH','ECG','HR','EIA','HD']
x = [2,2,4,1,1,1,1,1,1]
y = [2,2,4,1,1,1,1,1,1]
y[4] = 2

upx = m.jointinfer(x)
upy = m.jointinfer(y)
p = upx*1.0/(upx+upy)
print 'Results ', x, p, upx
print 'Results ', y, 1-p, upy
print 
#assert(False)
print '5.b'
# x1 = [2,1,1,1,2,1,2,2,1]
# x2 = [2,1,1,1,2,1,2,2,1]
# x2[1] = 2
# p = m.jointinfer(x1)+m.jointinfer(x2)

# y1 = [2,1,1,1,2,1,2,2,1]
# y1[3] = 2
# y2 = y1[:]
# y2[1] = 2
# q = m.jointinfer(y1)+m.jointinfer(y2)

# print zip(x1,x2), p*1.0/(p+q) , p
# print zip(y1,y2), q*1.0/(p+q) , q
print '----Reduced----'
x1 = [2,1,-1,1,2,-1,-1,-1,-1]
x2 = [2,2,-1,1,2,-1,-1,-1,-1]
p = m.marginal(x1)+ m.marginal(x2)
print p
y1 = [2,1,-1,2,2,-1,-1,-1,-1]
y2 = [2,2,-1,2,2,-1,-1,-1,-1]
q = m.marginal(y1)+ m.marginal(y2)

print zip(x1,x2), p*1.0/(p+q) , p
print zip(y1,y2), q*1.0/(p+q) , q

print '------GIVEN MODEL---------------------'
p = 'data-train-1.txt data-train-2.txt data-train-3.txt data-train-4.txt data-train-5.txt'.split()
q = 'data-test-1.txt data-test-2.txt data-test-3.txt data-test-4.txt data-test-5.txt'.split()
m = [BayesNets(train,test) for train,test in zip(p,q)]
acc = []
mean = 0.0
for each in m:
    each.learn()
    accuracy = each.get_accuracy()
    acc.append(accuracy)
mean = sum(acc)*1.0/len(acc)

sd = 0.0
for each in acc:
    sd += (each-mean)**2
sd/=len(acc)
print acc
print 'mean %f var %f , sd %f ' %( mean, sd , sd**0.5)


print '-----MY model-----'
mymodel = {'A':['CH'],'G':['BP','CH'],'BP':['HD'],'CH':['HD'],'HD':['HR','ECG','CP','EIA'],'CP':[],'EIA':[],'ECG': [], 'HR':[]}
#mymodel = {'A':['CH','HR'],'G':['BP','CH'],'BP':['HD'],'CH':['HD'],'HD':['HR','ECG','CP','EIA'],'CP':[],'EIA':[],'ECG': [], 'HR':[]}
m = [BayesNets(train,test,relationship=mymodel) for train,test in zip(p,q)]
acc = []
mean = 0.0
for each in m:
    each.learn()
    accuracy = each.get_accuracy()
    acc.append(accuracy)
mean = sum(acc)*1.0/len(acc)

sd = 0.0
for each in acc:
    sd += (each-mean)**2
sd/=len(acc)
print acc
print 'mean %f var %f , sd %f ' %( mean, sd , sd**0.5)

