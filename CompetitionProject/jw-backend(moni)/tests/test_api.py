import requests
import json
import sys

# 配置后端基础地址
BASE_URL = "http://localhost:3000"

def test_sync_and_query(username, password, semester):
    """
    测试同步接口（登录并同步指定学期的课表）
    """
    print(f"\n=== 测试同步接口 (学期: {semester}) ===")
    sync_url = f"{BASE_URL}/api/sync"
    payload = {
        "username": username,
        "password": password,
        "semester": semester
    }
    
    try:
        # 1. 调用同步接口
        print(f"正在请求: {sync_url} ...")
        response = requests.post(sync_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("同步成功!")
            print(f"消息: {result.get('message')}")
            print(f"学生姓名: {result.get('student', {}).get('name')}")
            print(f"同步课表数量: {result.get('timetableCount')}")
            
            # 2. 调用查询接口验证数据
            query_student_data(username, semester)
            
        else:
            print(f"同步失败! 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"请求发生异常: {e}")

def query_student_data(student_id, semester):
    """
    测试查询接口（获取指定学期的课表数据）
    """
    print(f"\n=== 测试查询接口 (学号: {student_id}, 学期: {semester}) ===")
    query_url = f"{BASE_URL}/api/student/{student_id}"
    params = {"semester": semester}
    
    try:
        response = requests.get(query_url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            courses = data.get('courses', [])
            
            print(f"查询成功! 找到 {len(courses)} 条课表记录。")
            
            if courses:
                print("\n前 3 条课表详情:")
                for course in courses[:3]:
                    print(f"- {course.get('name')} | 周{course.get('dayOfWeek')} | 第{course.get('period')}节 | {course.get('teacher')}")
        else:
            print(f"查询失败! 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"请求发生异常: {e}")

if __name__ == "__main__":
    # 可以通过命令行参数传入，或者直接在这里修改
    # 示例用法: python test_api.py 学号 密码 学期
    if len(sys.argv) >= 4:
        user = sys.argv[1]
        pwd = sys.argv[2]
        sem = sys.argv[3]
    else:
        # 默认测试数据（请根据实际情况修改）
        user = "your_student_id"
        pwd = "your_password"
        sem = "2023-2024-1"
        print("提示: 未提供命令行参数，使用默认配置。")
        print("用法: python test_api.py <学号> <密码> <学期>")

    test_sync_and_query(user, pwd, sem)
