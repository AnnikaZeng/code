import requests
from bs4 import BeautifulSoup

url = "https://www.zhihu.com/search?type=content&q=python"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")
questions = soup.find_all("a", {"class": "Link--primary"})

for question in questions:
    print("标题：", question.text)
    print("链接：", "https://www.zhihu.com" + question["href"])
    print("-----------------")
