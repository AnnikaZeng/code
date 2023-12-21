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


class TagCloud:
    def __init__(self):
        self.tags = {}

    def add(self, tag):
        self.tags[tag.lower()] = self.tags.get(tag.lower(), 0) + 1

    def __setitem__(self, tag, count):
        self.tags[tag.lower()] = count

    def __len__(self):
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)


cloud = TagCloud()
cloud.add("Python")
cloud.add("Python")
cloud.add("Python")
print(cloud.tags)
