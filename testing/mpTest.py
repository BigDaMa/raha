import os.path
import pickle

MP_DIRECTORY = 'mp/flights/feature-vectors'
SP_DIRECTORY = 'sp/flights/feature-vectors'

mpDict = {strategy_file: pickle.load(open(os.path.join(MP_DIRECTORY, strategy_file), "rb")) for strategy_file in os.listdir(MP_DIRECTORY)}
spDict = {strategy_file: pickle.load(open(os.path.join(SP_DIRECTORY, strategy_file), "rb")) for strategy_file in os.listdir(SP_DIRECTORY)}


print([a for a in mpDict.keys() if a not in spDict.keys()])
print([a for a in spDict.keys() if a not in mpDict.keys()])

same = True
for fileName in mpDict.keys():
    if mpDict[fileName]['output'] != spDict[fileName]['output'] or mpDict[fileName]['name'] != spDict[fileName]['name']:
        print(mpDict[fileName])
        print(spDict[fileName])
        same = False

print("The directories and files are the same: ", same)


