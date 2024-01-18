import random,os,sys
import itertools

INTEL_TOTAL = 2313153

# 1) Generate a training set
# 2) Generate a training set
# 3) Run test/training set on relevant models
# r) Gather results and put in a meaningful format

train_sizes = [1000,10000,100000]
test_sizes = [INTEL_TOTAL]

def get_sample(k,ofile):

    with open("../datasets/real/intel/sensors.txt",'r') as f:
        with open(ofile,'w') as f2:
            for line in f:
                line = line.strip().split()
                if(len(line) < 8): continue

                for i in range(4,8):
                    if(line[i] == ""): continue
                line = [str(float(t)) for t in line[4:8]] 
                f2.write(' '.join(line) + '\n')

def get_rand_sample(k,ofile):
    l = random.sample(range(0,INTEL_TOTAL),k)
    l.sort()

    idx = -1 
    idx2 = 0
    idx3 = 0
    with open("../datasets/real/intel/sensors.txt",'r') as f:
        with open(ofile,'w') as f2:
            for line in f:
                idx = idx + 1
                if idx2 == len(l) or idx3 == k: break
                if idx == l[idx2]:
                    idx2 = idx2+1
                    line = line.strip().split()
                    if(len(line) < 8): continue

                    for i in range(4,8):
                        if(line[i] == ""): continue
                    line = [str(float(t)) for t in line[4:8]] 
                    f2.write(' '.join(line) + '\n')
                    idx3 = idx3 + 1


set_sizes = set(train_sizes) | set(test_sizes)
for ss in list(set_sizes):
    ofile = "../datasets/real/intel/sensors-{}.txt".format(ss)
    if os.path.isfile(ofile): continue
    get_rand_sample(ss,ofile)

experiments = [
    [1,"gaussian",1.5],
    [0.7,"mixture",1,0.1],
    [0.7,"mixture",2,0.05]
]

results = {
    "gaussian": [],
    "mixture":[],
    "histogram":[]
}

os.chdir("../dboost")
cmd = None
f = ""
for (train,test) in itertools.product(train_sizes,test_sizes):
    train_file = "../datasets/real/intel/sensors-{}.txt".format(train)
    test_file = "../datasets/real/intel/sensors-{}.txt".format(test)
    for e in experiments:
        cmd = None
        if e[1] == "gaussian":
            f = "../results/sensors_{}_stat{}_{}{}.out".format(*([train]+e))
            #f = "../results/sensors_{}_{}_stat{}_{}{}.out".format(*([train]+[test]+e))
            #if os.path.isfile(f): continue
            cmd = "./dboost-stdin.py -m --pr 100000 --minimal -F ' ' --train-with {} {} --statistical {} --{} {} -d fracpart -d unix2date_float > /tmp/tmp.out 2>{}".format(*([train_file]+[test_file]+e+[f]))
        elif e[1] == "mixture":
            f = "../results/sensors_{}_stat{}_{}{}_{}.out".format(*([train]+e))
            #f = "../results/sensors_{}_{}_stat{}_{}{}_{}.out".format(*([train]+[test]+e))
            #if os.path.isfile(f): continue
            cmd = "./dboost-stdin.py -m --pr 100000 --minimal -F ' ' --train-with {} {} --statistical {} --{} {} {} -d fracpart -d unix2date_float > /tmp/tmp.out 2>{}".format(*([train_file]+[test_file]+e+[f]))
        else: assert(False)
        #if cmd != None:
        print(cmd)
        os.system(cmd)

        with open(f,'r') as f2:
            for line in f:
                line = line.strip().split()
                if line[0] == "Runtime":
                    results[e[1]].append( (train,test,float(line[1])) )
 
