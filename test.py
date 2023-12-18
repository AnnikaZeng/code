"""Module providing a function printing python version."""
# \" 转义符\
# \'
# \\
# \n
import math

course_name = "  python programming"
print(len(course_name))
print(course_name[0])
print(course_name[0:3])
first = "kate"
last = "Annika"
full = f"{first} {last}"  # 字符串模板 f"{}{}"
print(full)
print(course_name)
print(course_name.upper())
print(course_name.title())  # 首字母大写
print(course_name.strip())  # removw the white spaces 首尾, lstrip,rstrip
print(course_name.find("Pro"))  # return the index
print("pro" in course_name)  # 返回布尔值

print(10 / 3)
print(10 // 3)
print(10 % 3)
print(10**3)

print(math.ceil(2.2))
