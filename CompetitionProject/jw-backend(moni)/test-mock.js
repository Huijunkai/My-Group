const http = require('http');

function makeRequest(options, postData = null) {
    return new Promise((resolve, reject) => {
        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve({ status: res.statusCode, data: JSON.parse(data) });
                } catch (e) {
                    resolve({ status: res.statusCode, data: data });
                }
            });
        });
        
        req.on('error', reject);
        
        if (postData) {
            req.write(JSON.stringify(postData));
        }
        req.end();
    });
}

async function runTests() {
    console.log('========================================');
    console.log('🧪 模拟数据模式功能测试');
    console.log('========================================\n');
    
    const baseUrl = 'http://localhost:3000';
    let passed = 0;
    let failed = 0;
    
    async function test(name, fn) {
        try {
            await fn();
            console.log(`✅ ${name}`);
            passed++;
        } catch (error) {
            console.log(`❌ ${name}`);
            console.log(`   错误: ${error.message}`);
            failed++;
        }
    }

    console.log('📋 1. 模式检测\n');
    
    await test('获取模式信息', async () => {
        const res = await makeRequest(`${baseUrl}/api/mode/info`);
        if (!res.data.success || !res.data.isMock) throw new Error('模式检测失败');
        if (!res.data.availableTestAccounts || res.data.availableTestAccounts.length === 0) {
            throw new Error('没有可用测试账号');
        }
        console.log(`   当前模式: ${res.data.mode}`);
        console.log(`   测试账号数: ${res.data.availableTestAccounts.length} 个`);
    });

    console.log('\n📋 2. 教务系统认证\n');

    await test('登录成功 - 正确账号密码', async () => {
        const res = await makeRequest({
            hostname: 'localhost',
            port: 3000,
            path: '/api/sync',
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }, { username: '202101001', password: '123456' });
        
        if (!res.data.success) throw new Error(res.data.message);
        if (!res.data.info) throw new Error('缺少学生信息');
        if (!res.data.courses || res.data.courses.length === 0) throw new Error('缺少课程数据');
        console.log(`   学生姓名: ${res.data.info.name}`);
        console.log(`   课程数量: ${res.data.courses.length} 门`);
        console.log(`   成绩学期: ${Object.keys(res.data.grades).length} 个`);
        console.log(`   考试安排: ${res.data.exams.length} 门`);
    });

    await test('登录失败 - 密码错误', async () => {
        const res = await makeRequest({
            hostname: 'localhost',
            port: 3000,
            path: '/api/sync',
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }, { username: '202101001', password: 'wrongpassword' });
        
        if (res.data.success) throw new Error('应该返回失败');
        if (!res.data.message.includes('密码错误')) throw new Error('错误信息不正确');
    });

    await test('登录失败 - 用户不存在', async () => {
        const res = await makeRequest({
            hostname: 'localhost',
            port: 3000,
            path: '/api/sync',
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }, { username: '999999999', password: '123456' });
        
        if (res.data.success) throw new Error('应该返回失败');
        if (!res.data.message.includes('不存在')) throw new Error('错误信息不正确');
    });

    console.log('\n📋 3. 校园一信通\n');

    await test('一信通登录', async () => {
        const res = await makeRequest({
            hostname: 'localhost',
            port: 3000,
            path: '/api/xyyxt/login',
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }, { username: '202101001', password: '123456' });
        
        if (!res.data.success) throw new Error(res.data.message);
        if (!res.data.data.access_token) throw new Error('缺少access_token');
        console.log(`   Token类型: ${res.data.data.access_token.substring(0, 10)}...`);
    });

    const xyyxtLoginRes = await makeRequest({
        hostname: 'localhost',
        port: 3000,
        path: '/api/xyyxt/login',
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }, { username: '202101001', password: '123456' });

    await test('获取用户信息', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/userinfo?username=202101001`);
        if (!res.data || !res.data.name) throw new Error('缺少用户信息');
        console.log(`   用户名: ${res.data.name}`);
    });

    await test('获取余额', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/balance?username=202101001`);
        if (!res.data || !res.data.balance) throw new Error('缺少余额数据');
        console.log(`   余额: ${res.data.balance} 元`);
    });

    await test('获取交易记录', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/transactions?username=202101001&page=1&size=5`);
        if (!Array.isArray(res.data)) throw new Error('应该是数组');
        console.log(`   记录数: ${res.data.length} 条`);
    });

    console.log('\n📋 4. 宿舍信息\n');

    await test('获取南宁校区宿舍楼', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/buildings?username=202101001&areaId=nnxq`);
        if (!Array.isArray(res.data)) throw new Error('应该是数组');
        if (res.data.length === 0) throw new Error('宿舍楼列表为空');
        console.log(`   楼栋数: ${res.data.length} 栋`);
        console.log(`   示例: ${res.data[0].loudong_name}`);
    });

    await test('获取桂林校区宿舍楼', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/buildings?username=202101001&areaId=glxq`);
        if (!Array.isArray(res.data)) throw new Error('应该是数组');
        console.log(`   楼栋数: ${res.data.length} 栋`);
    });

    await test('获取房间列表', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/rooms?username=202101001&buildingId=4320&page=1&size=10`);
        if (!res.data || !Array.isArray(res.data.data)) throw new Error('房间数据格式错误');
        console.log(`   房间数: ${res.data.total} 个（当前页: ${res.data.data.length}）`);
    });

    await test('查询电费余额', async () => {
        const res = await makeRequest(`${baseUrl}/api/xyyxt/electricity?username=202101001&roomId=H4320101`);
        if (!res.data || !res.data.balance) throw new Error('缺少电费数据');
        console.log(`   房间: ${res.data.room_id}`);
        console.log(`   电费余额: ${res.data.balance} 元`);
    });

    console.log('\n📋 5. 版本信息\n');

    await test('获取版本信息', async () => {
        const res = await makeRequest(`${baseUrl}/api/version`);
        if (!res.data.name || !res.data.mode) throw new Error('版本信息不完整');
        console.log(`   服务名称: ${res.data.name}`);
        console.log(`   运行模式: ${res.data.mode}`);
        console.log(`   模拟数据: ${res.data.features.mockData ? '启用' : '禁用'}`);
    });

    console.log('\n========================================');
    console.log('✨ 测试完成！');
    console.log(`   通过: ${passed} 项`);
    console.log(`   失败: ${failed} 项`);
    console.log(`   总计: ${passed + failed} 项`);
    console.log('========================================\n');
    
    if (failed > 0) {
        process.exit(1);
    }
}

runTests().catch(error => {
    console.error('测试脚本执行失败:', error);
    process.exit(1);
});
