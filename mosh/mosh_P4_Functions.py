# def greet(first_name, last_name):
#     print(f"Hello, {first_name} {last_name}")
#     print("Welcome aboard")


# greet("Annika", "Zeng")


# def get_greeting(name):
#     return f"Hi {name}"


# print(get_greeting("Annika"))


# def increment(number, by):
#     return number + by


# print(increment(2, by=2))


# def increment(number, by=2):
#     return number + by


# print(increment(2, 5))


# def multiply(*numbers):  # 生成list
#     total = 1
#     for number in numbers:
#         total *= number
#     return total


# print("start")
# print(multiply(3, 5, 4, 5))


# def save_user(**user):  # 生成字典
#     print(user)


# save_user(id=1, name="Mosh", age=34)


def fizz_buzz(input):  # 这里的input只是变量名称，所以可以直接用来做运算，不需要转换
    if (input % 3 == 0) and (input % 5 == 0):
        return "fizzbuzz"
    if input % 3 == 0:
        return "Fizz"
    if input % 5 == 0:
        return "Buzz"

    return input


print(fizz_buzz(5))
