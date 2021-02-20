##########
# Author: Evan Hrouda
# Purpose: Various functions to pre-process data before running the ID3
#          decision tree learning algorithm
##########

#####
# Author: Evan Hrouda
# Purpose: Find the median of numerical attributes
#####
def numerical2binary_MedianThreshold(data, attrDict):
    import statistics

    medianList = {}
    for attr in attrDict:
        # for all the numeric attributes
        if not attrDict[attr]:
            # collect the integer data and find the median
            valuesList = []
            for example in data:
                valuesList.append(int(example[attr]))
            attrMedian = statistics.median(valuesList)
            medianList[attr] = attrMedian

    return medianList

#####
# Author: Evan Hrouda
# Purpose: replace all the numerical attributes with it's relation to the median
#          "larger" for those larger and "smaller" for those equal to or less than.
#####
def numerical2binary_MedianThreshold_Replace(data, attrDict, medianList):
    import copy
    attrDictCopy = copy.deepcopy(attrDict)

    for attr in medianList:
        # replace the attribute value with the new threshold value
        for example in data:
            if int(example[attr]) > medianList[attr]:
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
# Purpose: Find the majority value of that attributes
#####
def replaceUnknown_MajorityAttribute(data, attrDict, unknown):
    majorityAttrs = {}
    for attr in attrDict:
        majorityAttrs[attr] = findMajorityAttribute(data, attrDict, attr, unknown)

    return majorityAttrs

#####
# Author: Evan Hrouda
# Purpose: Replace unknown attributes with the majority value of that attribute
#####
def replaceUnknown_MajorityAttribute_Replace(data, majorityAttrs, unknown):
    for attr in majorityAttrs:
        for example in data:
            if example[attr] == unknown:
                example[attr] = majorityAttrs[attr]

    return data