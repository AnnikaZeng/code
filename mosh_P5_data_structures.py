# letters = ["a", "b", "c"]
# for letter in enumerate(letters):  # enumerate 能够返回元素的index
#     print(letter[0], letter[1])

# for index, letter in enumerate(letters):
#     print(index, letter)

# letters.append("d")
# letters.insert(0, "-")
# letters.pop(0)
# letters.remove("b")
# del letters[0:3]
# letters.clear()

# print(letters)

# numbers = [3, 51, 2, 8, 6]
# # numbers.sort(reverse=True)
# print(sorted(numbers, reverse=True))
# print(numbers)


# items = [("product1", 10), ("product2", 9), ("product3", 12)]


# def sort_item(item):
#     return item[1]

# items.sort(key=sort_item)  # 为什么这里调用函数不需要加括号呢，还加上了key又是为什么？     sort可以遍历items的每一项
# sort 函数的 key 参数可以用来指定一个函数。不需要括号： 在传递函数作为参数时，你只是提供了函数的引用而不是调用它。函数名称本身就是一个对象，可以传递给其他函数。当你使用 key=sort_item 时，你只是告诉 sort 函数使用 sort_item 函数作为排序的标准，而没有实际调用它。

# items.sort(key=lambda item: item[1])  # 此操作则不需要再命名一个函数，匿名函数 lambda
# print(items)


# items = [("product1", 10), ("product2", 9), ("product3", 12)]

# for item in items:
#     prices.append(item[1])

# prices = list(map(lambda item: item[1], items))  # 为什么这里的list是用（）？
# prices = [item[1] foe item in items]  #与41行是等效的，更简洁，成为list comprehension
# print(prices)

# filtered = list(filter(lambda item: item[1] >= 10, items))
# filtered = [item for item in items if item[1] >= 10]
# print(filtered)

# list1 = [1, 2, 3]
# list2 = [10, 20, 30]

# print(list(zip("abc", list1, list2)))

# 只有条件的布尔值为True，if后面的语句才会被执行，所以一下代码是正确的，是为了将空列表取反，即检查列表是否为空
# if not[]: print("disable")

# stack
# browsing_session = []
# browsing_session.append(1)

# browsing_session.pop()
# if browsing_session:
#     browsing_session[-1]


# queues
# from collections import deque  # 这句是什么意思

# queue = deque([])
# queue.append(1)
# # queue.append(2)
# # queue.append(3)
# queue.popleft()
# print(queue)
# if not queue:
#     print("empty")


# tuple
# point = (1, 2) + (3, 4)
# point2 = (1, 2) * 3
# point3 = tuple([1, 2])
# point4 = tuple("Hello world")
# print(point3, "\n", point4)


from array import array

# numbers = array("i", [1, 2, 3])
# print(numbers[0])

# set
# numbers = [1, 1, 2, 3, 4]
# first = set(numbers)
# second = {1, 5}
# print(first | second)
# print(first & second)
# print(first - second)
# print(first ^ second)

# dict
# point = dict(x=1, y=2)
# point["z"] = 10
# del point["x"]

# print(point)
# print(point.get("a", 0))
# # for key in point:
# #     print(key, point[key])
# for key, value in point.items():
#     print(key, value)


# exercise！！！
from pprint import pprint  # 可以控制输出结果的形式

sentence = "This is a common interview question"  # 找到出现频率最高的字母
# set_char = set(sentence)     多余步骤
# print(set_char)
# count_dict = {char: 0 for char in set_char}   多余步骤
# print(count_dict)
# max_value = 0

# for char in list(sentence):
#     count_dict[char] += 1
#     if count_dict[char] > max_value:
#         max_value = count_dict[char]
#         max_key = char
# print(max_key, max_value)
# pprint(count_dict, width=1) #使输出结果好看些，以一列的形式输出

# GPT完善版
# count_dict = {}
# max_key, max_value = None, 0

# for char in sentence:
#     if char != " ":
#         count_dict[char] = count_dict.get(char, 0) + 1
#         if count_dict[char] > max_value:
#             max_key, max_value = char, count_dict[char]

# pprint(count_dict, width=1)
# print("Most frequent character:", max_key, "Frequency:", max_value)

# Code Answer
char_frequency = {}
for char in sentence:
    if char in char_frequency:
        char_frequency[char] += 1
    else:
        char_frequency[char] = 1  # 还是要再巩固是如何将数据存入和提出字典的，还是有些模糊。

char_frequency_sorted = sorted(
    char_frequency.items(), key=lambda kv: kv[1], reverse=True
)

print(char_frequency_sorted[0])
pprint(char_frequency_sorted)
