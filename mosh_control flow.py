# temperature = input("温度：")
# if int(temperature) > 30:
#     print("It's warm")
#     print("Drink water")
# elif int(temperature) > 20:
#     print("It's nice")
# else:
#     print("It's cold")
# print("Done")

# age = 22
# message = "Eligible" if age >= 18 else "Not eligible"
# print(message)
# ternary operator


# logical operators: and, or, not

# high_income = True
# good_credit = True
# student = False

# if high_income and good_credit and not student:
#     print("Eligible")

# age should be between 18 and 65
# age = 22
# if 18 <= age < 65:
#     print("eligible")

# for number in range(1, 10, 2):
#     print("Attempt", number, number * ".")

# for...else 循环
# successful = True
# for number in range(3):
#     print("Attempt")
#     if successful:
#         print("Successful")
#         break
# else:
#     print("Attempted 3 times and failed")


# command = ""
# while command.lower() != "quit":
#     command = input(">")
#     print("ECHO", command)


# while True:
#     command = input(">")
#     print("ECHO", command)
#     if command.lower() == "quit":
#         break

t = 0
for number in range(1, 10):
    while number % 2 == 0:
        t = t + 1
        print(number)
        break
print(f"We have {t} even numbers")
