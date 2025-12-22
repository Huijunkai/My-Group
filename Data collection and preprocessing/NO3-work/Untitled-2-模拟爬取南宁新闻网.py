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

# 例子[{'brief': '\u3000\u30002025年10月9日，广西融水苗族自治县红水乡高文苗寨举办第二十届传统“烧鱼节”，开展烧烤禾花鱼、跳苗舞、芦笙踩堂、拔河等活动，吸引附近乡镇各民族同胞和外地游客共庆丰收。兰堃摄（人民图片 网）', 'image': 'https://p3.img.cctvpic.com/photoworkspace/2025/10/13/2025101309415171672.jpg', 'image3': 'https://p3.img.cctvpic.com/photoworkspace/2025/10/13/2025101309421897186.jpg', 'keywords': '广西融水,烧鱼节,民俗,苗族', 'length': '7', 'title': '广西融水：传统“烧鱼节”庆丰收', 'image2': 'https://p2.img.cctvpic.com/photoworkspace/2025/10/13/2025101309420362818.jpg', 'url': 'https://xczx.cctv.com/2025/10/13/PHOAfgN0AmmgecpMQB7PiHzt251013.shtml', 'page_id': 'PAGEqz3eMqOKpJn2mYy5v1jY220610', 'focus_date': '1760319769961', 'id': 'PHOAfgN0AmmgecpMQB7PiHzt251013'}
# 1.新闻标题，2.image链接，3.image2链接，4.image3链接，5.url链接，6.keywords关键词，7.brief简介，8.focus_date，9.id

for item in dataDic['data']['list']:
    (item['title'], item['image'], item['image2'], item['image3'], item['url'], item['keywords'], item['brief'], item['focus_date'], item['id'])
    # 对应上面的image2和image3进行判空，如果没有就存入'暂无此数据'
    if not item['image2']:
        item['image2'] = '暂无此数据'
    if not item['image3']:
        item['image3'] = '暂无此数据'
    
    # 写入文件和数据库
    with open('news_data.txt', 'w', encoding='utf-8') as f:
        for item in dataDic['data']['list']:
            f.write(f"第{dataDic['data']['list'].index(item) + 1}篇新闻\n")
            f.write(f"标题: {item['title']}\n")
            f.write(f"简介: {item['brief']}\n")
            f.write(f"关键词: {item['keywords']}\n")
            f.write(f"URL: {item['url']}\n")
            f.write(f"Image: {item['image']}\n")
            f.write(f"Image2: {item['image2']}\n")
            f.write(f"Image3: {item['image3']}\n")
            f.write(f"Focus Date: {item['focus_date']}\n")
            f.write(f"ID: {item['id']}\n")
            f.write("\n----------------\n\n")
    
    # 写入数据库
    import mysql.connector as mysql
    conn = mysql.connect(
        host='localhost',               # 主机名
        user='root',                    # 用户名
        password='121902',              # 密码
        database='cctv_nn_news'         # 数据库名
    )
    cursor = conn.cursor()
    for item in dataDic['data']['list']:
        # 如果有空就保存'暂无此数据'
        if not item['image2']:
            item['image2'] = '暂无此数据'
        if not item['image3']:
            item['image3'] = '暂无此数据'
            # 如果有重复的id就跳过
        cursor.execute("SELECT COUNT(*) FROM news WHERE id = %s", (item['id'],))
        result = cursor.fetchone()
        if result[0] > 0:
            continue
        sql = "INSERT INTO news (title, brief, keywords, url, image, image2, image3, focus_date, id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (item['title'], item['brief'], item['keywords'], item['url'], item['image'], item['image2'], item['image3'], item['focus_date'], item['id'])
        cursor.execute(sql, val)
        conn.commit()
    cursor.close()
    conn.close()
    

    # # =============写入文件或数据库================
    # # 将全部写入文件
    # with open('news_data.txt', 'w', encoding='utf-8') as f:
    #     f.write("标题:\n")
    #     for title :
    #         f.write(title + '\n')
    #     f.write("\n简介:\n")
    #     for brief in briefs:
    #         f.write(brief + '\n')
    #     f.write("\n关键词:\n")
    #     for keywords in keywords_list:
    #         f.write(keywords + '\n')
    #     f.write("\nURL:\n")
    #     for url in urls:
    #         f.write(url + '\n')
    #     f.write("\nImage:\n")
    #     for image in images:
    #         f.write(image + '\n')
    #     f.write("\nImage2:\n")
    #     for image2 in images2:
    #         f.write(image2 + '\n')
    #     f.write("\nImage3:\n")
    #     for image3 in images3:
    #         f.write(image3 + '\n')
    #     f.write("\nFocus Date:\n")
    #     for focus_date in focus_dates:
    #         f.write(focus_date + '\n')
    #     f.write("\nID:\n")
    #     for id_ in ids:
    #         f.write(id_ + '\n')
        

    # # 写入数据库
    # import mysql.connector as mysql
    # conn = mysql.connect(
    #     host='localhost',       # 主机名
    #     user='root',            # 用户名
    #     password='121902',      # 密码
    #     database='cctv_nn_news'         # 数据库名
    # )
    # cursor = conn.cursor()
    # for i in range(len(titles)):
    #     sql = "INSERT INTO news (title, brief, keywords, url, image, image2, image3, focus_date, id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    #     val = (titles[i], briefs[i], keywords_list[i], urls[i], images[i], images2[i], images3[i], focus_dates[i], ids[i])
    #     cursor.execute(sql, val)
    #     conn.commit()
    # cursor.close()