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
def numerical2binary_MedianThreshold(data, cols, attrDict):
    import statistics 

    for attr in attrDict:
        # for all the numeric attributes
        if not attrDict[attr]:
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
            attrDict[attr] = ["smaller", "larger"]

    return attrDict, data
