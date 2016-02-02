from itertools import product
from math import log
from collections import defaultdict
# A->B

#Bayesian Nets
class model:

    def __init__(self):
        #self.relationship={'A':['B']}; 
        self._factorize()
        self.dataorder = ['A','G','CP','BP','CH','ECG','HR','EIA','HD']
        #self.dataorder = ['A','B'] #as it is present in the file
        self.bigtable = defaultdict(float);
        self.bigtable_order = dict(zip(self.dataorder,range(len(self.dataorder))))
        return;
    
    
    #First is always a non conditional
    # P(B,A) = P(B)* P(A|B)  [['B'],['A','B']]
    
    def _factorize(self):
        #self.factor = [['B'],['A','B']]
        #self.factor = [['A'],['B'],['C','A','B'],['D','C']]
        self.factor = [['A'],['G'],['CH','G','A'],['BP','G'],
        ['HD','CH','BP'],['HR','HD','A'],['ECG','HD'],['CP','HD'],['EIA','HD']]
        pass;
        
    def _getfulltable(self):
        f = open('data-train-1.txt')
        for each in f.readlines():
            tup = ()
            temp = {}
            
            for every in each.strip().split(','):
                tup = tup+ (float(every),)
            self.bigtable[tup]+=1
        for each in self.bigtable:
            print each,self.bigtable[each]
        
        f.close()
        pass;
        
    
    
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
                
                    

    
    def learn(self):
        #max likely learn
        #progressively assume
        self._getfulltable()
        self._learncpt()
        assert(len(self.parameters) == len(self.factor))
        return;
        
        

    def infer(self):
        
        return;
    
    
        
    #vals is in dataorder
    def jointinfer(self,value):
        s = 1.0
        r = dict(zip(self.dataorder,value))
        for eachfactor,cpt in zip(self.factor,self.parameters):
            var = ()
            for every in eachfactor:
                var = var+ (r[every],)
            print cpt[var], var
            s*=cpt[var]
        return s;


m = model()
m.learn()

#print m.jointinfer((0,1,5,2))*32, 3.0

    
# #vals need to be in data-order
x = [(1,2,3),(1,2),(1,2,3,4),(1,2),(1,2),(1,2),(1,2),(1,2),(1,2)]
#
#self.dataorder = ['A','G','CP','BP','CH','ECG','HR','EIA','HD']
a=[]
for each in product(*x):
    a.append(m.jointinfer(each))
assert(sum(a)-1.0 <0.001)

#self.factor = [['A'],['G'],['CH','G','A'],['BP','G'],
#        ['HD','CH','BP'],['HR','HD','A'],['ECG','HD'],['CP','HD'],['EIA','HD']]
print 'A '
print m.parameters[0]
print 'BP|G'
print m.parameters[3]
print 'HD|BP,CH'
print m.parameters[4]
print 'HR|HD,A'
print m.parameters[5]
