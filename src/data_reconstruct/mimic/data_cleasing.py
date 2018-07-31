# coding=utf-8


# TODO 更为细致的清洗策略留待之后做
def diagnosis_normalize(icd_code):
    return icd_code[0:3]


# TODO 更为细致的清洗策略留待之后做
def procedure_normalize(procedure_code):
    return procedure_code[0:2]
