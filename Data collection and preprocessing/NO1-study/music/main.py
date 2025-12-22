import os
import  requests as req
filename = '2146002926.mp3'
savePath = './mp3s/'

# 判断目录是否存在，不存在就创建
if not os.path.exists(savePath):
    os.mkdir(savePath) # 创建目录

res = req.get('http://music.163.com/song/media/outer/url?id=2146002926.mp3')
with open(savePath+filename, 'bw') as ff:
    ff.write(res.content)