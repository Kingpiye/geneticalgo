import random
idealnumber = 76686
oldfit = 100000
currentfit = 100000

def fitness(idealnumber, genstring1):
    if genstring1>idealnumber:
        fit = genstring1-idealnumber
    else:
        fit = -1*(genstring1-idealnumber)
    currentfit = fit
    return currentfit

while currentfit>0:
    genstring1 = random.randrange(0,100000)
    print(genstring1)
    currentfit = fitness(idealnumber, genstring1)
    if currentfit < oldfit:
        oldfit = currentfit
    print(oldfit)



print(genstring1)