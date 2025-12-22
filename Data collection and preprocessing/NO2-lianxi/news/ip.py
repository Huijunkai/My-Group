# 生成100个代理地址，端口从7890到7989 
with open('F:/Source-code-management-repository/MyHome/Data-collection-and-preprocessing/NO2-lianxi/news/ip.txt', 'w') as f: 
    for port in range(7890, 7990): 
        f.write(f'http://localhost:{port}\n') 