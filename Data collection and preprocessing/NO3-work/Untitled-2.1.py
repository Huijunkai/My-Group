import requests as req
# from lxml import etree
import json

head= {
    'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0',
    'Cookie' : 'cna=VZR0Ib5xGgoCAasl0YPhSuSy; sca=cbcabba2; atpsida=14223ad016458dc397884524_1760405103_3'
}

res = req.get('https://api.cntv.cn/newList/getMixListByPageId?serviceId=sannong&catg=0,1,2&id=PAGEqz3eMqOKpJn2mYy5v1jY220610&p=1&n=100&cb=TOPC1594283913340405', headers=head)
res.encoding = 'utf-8'

strData = res.text.replace('TOPC1594283913340405(','').replace(');','')
# 把字符串转换为字典
dataDic = json.loads(strData)
# print(dataDic)
# [{'keywords': '广西融水,烧鱼节,民俗,苗族', 'image': 'https://p3.img.cctvpic.com/photoworkspace/2025/10/13/2025101309415171672.jpg', 'focus_date': '1760319769961', 'url': 'https://xczx.cctv.com/2025/10/13/PHOAfgN0AmmgecpMQB7PiHzt251013.shtml', 'id': 'PHOAfgN0AmmgecpMQB7PiHzt251013', 'image2': 'https://p2.img.cctvpic.com/photoworkspace/2025/10/13/2025101309420362818.jpg', 'title': '广西融水：传统“烧鱼节”庆丰收', 'length': '7', 'page_id': 'PAGEqz3eMqOKpJn2mYy5v1jY220610', 'image3': 'https://p3.img.cctvpic.com/photoworkspace/2025/10/13/2025101309421897186.jpg', 'brief': '\u3000\u30002025年10月9日，广西融水苗族自治县红水乡高文苗寨举办第二十届传统“烧鱼节”，开展烧烤禾花鱼、跳苗舞、芦笙踩堂、拔河等活动，吸引附近乡镇各民族同胞和外地游客共庆丰收。兰堃摄（人民图片网）'}]
# 从字典中提取标题

