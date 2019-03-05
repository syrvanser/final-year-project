import multiprocessing as mp
import random, time

class MainClass:
    def __init__(self):
        self.value = 1
    def check(self, arg):
        time.sleep(random.uniform(0.01, 0.3))
        print(id(self),self.value,  arg)

def main():
    mc = MainClass()

    with mp.Pool(processes = 4) as pool:
        for i in range (8):
            pool.apply_async(mc.check, (i,))
            
        #temp = [pool.apply_async(mc.check, (i,)) for i in range(8)]
        #results = [t.get() for t in temp]

if __name__ == '__main__':
    main()




'''l = [2,-2,3,-5,1]

#n1 = max(l, key = (lambda s: s*s))

max_value = -1
n2 = l[0]
for s in l:
    value = s*s
    if value > max_value:
        max_value = value
        n2 = s
#print(n1)
print(n2)'''