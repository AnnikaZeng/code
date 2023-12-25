# class Point:
#     default_color = "red"

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __eq__(self, other):
#         return self.x == other.x and self.y == self.y

#     def __gt__(self, other):
#         return self.x > other.x and self.y > self.y

#     def __add__(self, other):
#         return Point(self.x + other.x, self.y + other.y)


#     def __str__(self):
#         return f"({self.x}, {self.y})"

#     @classmethod
#     def zero(cls):
#         return cls(0, 0)

#     def draw(self):
#         print(f"Point ({self.x},{self.y})")


# point = Point(10, 20)
# other = Point(1, 2)
# combined = point + other
# print(combined.x)
# another = Point.zero()
# # another.draw()
# # point.draw()
# print(point)

# print(isinstance(point, Point))  # 判断是否属于这个类


# create custom container
# class TagCloud:
#     def __init__(self):
#         self.__tags = {}

#     def add(self, tag):
#         self.__tags[tag.lower()] = self.__tags.get(tag.lower(), 0) + 1

#     def __setitem__(self, tag, count):
#         self.__tags[tag.lower()] = count

#     def __len__(self):
#         return len(self.__tags)

#     def __iter__(self):
#         return iter(self.__tags)


# cloud = TagCloud()
# cloud.add("Python")
# cloud.add("Python")
# cloud.add("Python")
# print(cloud.__dict__)
# print(cloud._TagCloud__tags)


# class Product:
#     def __init__(self, price):
#         self.price = price

#     @property
#     def price(self):
#         return self.__price

#     @price.setter
#     def price(self, value):
#         if value < 0:
#             raise ValueError("Price cannot be Negative")
#         self.__price = value


# product = Product(10)
# print(product.price)


# Animal: parent, base
# Mammal: child, sub
# class Animal:
#     def __init__(self):
#         print("Animal constructor")
#         self.age = 1

#     def eat(self):
#         print("eat")


# class Mammal(Animal):
#     def __init__(self):
#         super().__init__()
#         print("Mammal constructor")
#         self.weight = 2

#     def walk(self):
#         print("walk")


# class Fish(Animal):
#     def swim(self):
#         print("swim")


# m = Mammal()
# m.eat()
# print(m.age)
# print(m.weight)
# print(isinstance(m, Animal))
# print(issubclass(Mammal, object))

# from abc import ABC, abstractmethod


# class InValidOperationError(Exception):
#     pass


# class Stream(ABC):
#     def __init__(self):
#         self.opened = False

#     def open(self):
#         if self.opened:
#             raise InValidOperationError("Stream is already opened")
#         self.opened = True

#     def Close(self):
#         if not self.opened:
#             raise InValidOperationError("Stream is already closed")
#         self.opened = False

#     @abstractmethod
#     def read(self):
#         pass


# class FileStream(Stream):
#     def read(self):
#         print("reading data from a file")


# class NetworkStream(Stream):
#     def read(self):
#         print("reading data from a stream")


# class MemoryStream(Stream):
#     def read(self):
#         print("Reading data from a memory stream.")


# stream = MemoryStream()
# stream.read()


# illustrate the abstract base class and polymorphism
# from abc import ABC, abstractmethod


# class UIControl(ABC):
#     @abstractmethod
#     def draw(self):
#         pass


# class TextBox(UIControl):
#     def draw(self):
#         print("TextBox")


# class DropDownList(UIControl):
#     def draw(self):
#         print("DropDownList")


# def draw(controls):
#     for control in controls:
#         control.draw()


# ddl = DropDownList()
# textbox = TextBox()
# draw([ddl, textbox])


# extending built-in types
# class Text(str):
#     def duplicate(self):
#         return self + self


# class TrackableList(list):
#     def append(self, object):
#         print("Append called")
#         super().append(object)  # overrides the append method


# text = Text("Python")
# print(text.lower())
# print(text.duplicate())
# list = TrackableList()
# list.append("1")
# print(list)

# data classes
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p1 = Point(x=1, y=2)
p2 = Point(x=1, y=2)
print(p1 == p2)
