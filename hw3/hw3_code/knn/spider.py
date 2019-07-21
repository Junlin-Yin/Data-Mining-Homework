import urllib.request as r

source_url = 'http://jwbinfosys.zju.edu.cn/CheckCode.aspx'
head = {}
head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
req = r.Request(source_url, headers = head)
N_train = 100
for i in range(N_train):
    response = r.urlopen(req)
    html = response.read()
    path = './checkcode/' + str(i) + '.aspx'
    with open(path, 'wb') as f:
        f.write(html)
    print(i, 'done')