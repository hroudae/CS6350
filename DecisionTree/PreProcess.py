##########
# Author: Evan Hrouda
# Purpose: Various functions to pre-process data before running the ID3
#          decision tree learning algorithm
##########

#####
# Author: Evan Hrouda
# Purpose: Convert numerical features to binary ones by sorting the attribute
#          over its median. The numerical feature will turn into a categorical
#          one with categories "smaller" and "larger". The numerical features
#          in the attribute dictionary should be empty
# Returns: the updated attribute dictionary and data
#####
def numerical2binary_MedianThreshold(data, attrDict):
    import statistics, copy

    attrDictCopy = copy.deepcopy(attrDict)
    for attr in attrDictCopy:
        # for all the numeric attributes
        if not attrDictCopy[attr]:
            # collect the integer data and find the median
            valuesList = []
            for example in data:
                valuesList.append(int(example[attr]))
            attrMedian = statistics.median(valuesList)
            
            # replace the attribute value with the new threshold value
            for example in data:
                if int(example[attr]) > attrMedian:
                    example[attr] = "larger"
                else:
                    example[attr] = "smaller"
            attrDictCopy[attr] = ["smaller", "larger"]

    return attrDictCopy, data

#####
# Author: Evan Hrouda
# Purpose: Find the majority value of an atrribute
#####
def findMajorityAttribute(data, attrDict, attrCol, unknown):
    countDict = {}
    for lbl in attrDict[attrCol]:
        countDict[lbl] = 0
    
    for example in data:
        if countDict[example[attrCol]] != unknown:
            countDict[example[attrCol]] = countDict[example[attrCol]] + 1

    vals = list(countDict.values())
    keys = list(countDict.keys())
    return keys[vals.index(max(vals))]

#####
# Author: Evan Hrouda
# Purpose: Replace unknown attributes with the majority value of that attribute
#####
def replaceUnknown_MajorityAttribute(data, attrDict, unknown):
    for attr in attrDict:
        majorityValue = findMajorityAttribute(data, attrDict, attr, unknown)

        for example in data:
            if example[attr] == unknown:
                example[attr] = majorityValue

    return data
