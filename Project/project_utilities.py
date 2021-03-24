##########
# Author: Evan Hrouda
# Purpose: Various utilitiy functions used in the income prediction project
##########

#####
# Author: Evan Hrouda
# Purpose: Replace continuous values with labels corresponding to their decade
#####
def replaceContinuous_Decades(data, attr):
    newAttrValues = []
    for example in data:
        decade = f"{int(example[attr]) // 10}"
        example[attr] = decade
        if decade not in newAttrValues:
            newAttrValues.append(decade)
    return newAttrValues, data

def replaceContinuous_Median(data, attr):
    import statistics
    newAttrValues = []

    valuesList = []
    for example in data:
        valuesList.append(float(example[attr]))
    attrMedian = statistics.median(valuesList)
    for example in data:
        if float(example[attr]) > attrMedian:
            example[attr] = "larger"
        else:
            example[attr] = "smaller"
    newAttrValues = ["smaller", "larger"]

    return newAttrValues, data


def replaceContinuous_QuartilesAttr(data, attr):
    import statistics
    newAttrValues = []

    valuesList = []
    for example in data:
        valuesList.append(float(example[attr]))
    quartiles = statistics.quantiles(valuesList, n=4)
    for example in data:
        if int(example[attr]) < quartiles[0]:
            example[attr] = 'Q1'
        elif int(example[attr]) < quartiles[1]:
            example[attr] = 'Q2'
        elif int(example[attr]) < quartiles[2]:
            example[attr] = 'Q3'
        else:
            example[attr] = 'Q4'
    newAttrValues = ["Q1", "Q2", 'Q3', 'Q4']

    return newAttrValues, data


#####
# Author: Evan Hrouda
# Purpose: Perform One-Hot encoding on the data
#####
def OneHotEncoding(data, attrDict, labelCol, zero2neg):
    x = []
    y = []
    for example in data:
        xi = [1]
        for attr in attrDict:
            if attr == labelCol:
                if zero2neg:
                    y.append(2*float(example[attr]) - 1)
                else:
                    y.append(float(example[attr]))
                continue
            # this is already a continuos value, so just add it
            if not attrDict[attr]:
                xi.append(float(example[attr]))
            i = 0
            for featval in attrDict[attr]:
                if example[attr] != featval:
                    xi.append(0.0)
                else:
                    xi.append(1.0)
        x.append(xi)
    return x, y
