from bs4 import BeautifulSoup
import requests as req

head={
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

res = req.get('https://nn.zu.fang.com/', headers=head)

bs = BeautifulSoup(res.text, 'lxml')

# 获取一页所有房屋信息dl标签元素
dls = bs.select('#listBox > div.houseList > dl')

# 循环遍历dls列表，解析房屋信息
for dl in dls:

    # 解析小区名称
    name = dl.select('dd > p > a')[0].text
    #name2 = bs.select('#rentid_D09_1_02 > a')[0].get('title')

    # 解析配套信息
    info = dl.select('dd > p.font15.mt12.bold')[0].text
    print(info)

    # 地铁信息  dd > div > p > span
    # 解析地铁信息(如果没有地铁信息，则返回无地铁信息)
    if len(dl.select('dd > div > p > span')) > 0:
        dt = dl.select('dd > div > p > span')[0].text
    else:
        dt = "无地铁信息"
    print(dt)


    # 解析价格信息 #listBox > div.houseList > dl:nth-child(1) > dd > div.moreInfo > p > span
    price = dl.select('dd > div.moreInfo > p > span')[0].text
    print(price)

    # 解析地址信息  dd > p:nth-child(3)
    address = dl.select('dd > p:nth-child(3)')[0].text
    print(address)

    # 解析封面图路径
    # <img class="b-lazy lazyload" usertype="1ccc" data-original="//cdnsfb.soufunimg.com/viewimage/open/0/2025_9/21/M9/1/28f86073-b81d-4175-b28b-aa84276a2cca/275x207.jpg?t=1" src="//cdnsfb.soufunimg.com/viewimage/open/0/2025_9/21/M9/1/28f86073-b81d-4175-b28b-aa84276a2cca/275x207.jpg?t=1" onerror="imgiserror(this,'//cdnsfb.soufunimg.com/open/0/2025_9/21/M9/1/28f86073-b81d-4175-b28b-aa84276a2cca.jpg')" style="">
    # rentid_D09_10_01 > a > img
    # rentid_D09_11_01 > a > img
    img = dl.select('dt > a > img')[0].get('data-original')
    print(img)
    # 解析房屋信息详情路径
    # <a href="/chuzu/3_172665097_1.htm?channel=1,2&amp;source_page=1" data_channel="1,2" source_page="1" target="_blank" title="东葛华都1房1厅中装修1700元">东葛华都1房1厅中装修1700元</a>
    #rentid_D09_45_02 > a
    detail = dl.select('dt > a')[0].get('href')
    print(detail)

    # 把以上信息打印成一行，用逗号隔开
    fang_info = f"{name}+','+{info}+','+{dt}+','+{price}+','+{address}+','+{img}+'{detail}\n"
print(fang_info)

if __name__ == '__main__':
    for i in range(100):
        url =''
        