import httpx

url = "https://plan.polsl.pl/plan.php?type=2&id=343261191&cvsfile=true&wd=1"
resp = httpx.get(url)
resp.encoding = "utf-8"
text = resp.text
print(text[:400])
