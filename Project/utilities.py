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
