"""
写在前面：“双色球”每注投注号码由6个红色球号码和1个蓝色球号码组成。红色球号码从1--33中选择；
蓝色球号码从1--16中选择。

1、一等奖。红区6个号码和蓝区1个号码全中。
2、二等奖。红区6个号码全中，蓝区号码没有中。
3、三等奖。红区中了5个号码，蓝区号码也中了。
4、四等奖。红区中了5个号码，蓝区号码没中，或者红区中了4个号码，蓝区中了1个号码。
5、五等奖。红区中了4个号码，蓝区号码没中，或者红区中了3个号码，蓝区中了1个号码。
6、六等奖。红区中了2个号码，蓝区号码中了1个号码，或者红区中了2个号码，蓝区中了1个号码，
或者红区没有中号码，蓝区中了1个号码。
"""

import random
def luck_number():
        number = [x for x in range(1, 34)]
        last_number = [x for x in range(1, 17)]
        luck_number = []
        for x in range(6):
            num = number[random.randint(0, len(number) - 1)]
            luck_number.append(num)
            number = list(set(number).difference(set({num})))
        luck_number.sort()
        luck_number.append(last_number[random.randint(0, 15)])
        luck_str = ' '.join('0' + str(s) if len(str(s)) < 2 else str(s) for s in luck_number)
        print(luck_str)
        return luck_str


def luck_boy(num):
        """
        get luck number
        :param num: 需要生成几组号码
        :return: lsit
        """
        arr = []
        for x in range(int(num)):
            arr.append(luck_number())
        return arr


arr = luck_boy(3)
    # print(arr)