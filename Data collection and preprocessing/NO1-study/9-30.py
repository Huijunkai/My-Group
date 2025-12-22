from bs4 import BeautifulSoup
import requests as req

head={
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

res = req.get('https://nn.zu.fang.com/', headers=head)

bs = BeautifulSoup(res.text, 'lxml')


#获取一页所有房屋信息dl标签元素
dls= bs.select('#listBox > div.houseList > dl')
#循环遍历dls列表，获取房屋信息
for dl in dls:


    # 解析小区名称
    name = dl.select('dd > p > a')[0].text
    #name2 = bs.select('#rentid_D09_1_02 > a')[0].get('title')

    # 解析配套信息
    info = dl.select('dd > p.font15.mt12.bold')[0].text
    print(info)
    # 地铁信息
    dt = dl.select('dd > div > p > span')[0].text
    print(dt)

    # 解析价格信息
    price = dl.select('dd > div.moreInfo > p')[0].text
    print(price)
    # 解析地址信息
    address = dl.select('dd ')[0].text
    print(address)
    # 解析封面图路径
    img = dl.select('dt > a > img')[0].get('data-original')
    print(img)
    # 解析房屋信息详情路径

    path = dl.select('dd > p > a')[0].get('href')
    print(path)
    fang_info = f"{name}+','+{info}+','+{dt}+','+{price}+','+{address}+','+{img}+','+{path}\n"
print(fang_info)