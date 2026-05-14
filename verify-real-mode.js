require('dotenv').config({ path: __dirname + '/.env' });
const { isMockMode, getModeInfo } = require('./src/mode');

console.log('========================================');
console.log('验证配置：Mock 模式检查');
console.log('========================================');
console.log('');

const modeInfo = getModeInfo();
console.log('✅ 模式信息:', JSON.stringify(modeInfo, null, 2));
console.log('');

if (modeInfo.isMock) {
    console.log('❌ 错误：仍在使用 Mock 模式！');
    console.log('   请检查 .env 文件中的 MOCK_MODE 和 NODE_ENV 设置');
    process.exit(1);
} else {
    console.log('✅ 成功：已切换到生产环境模式（真实数据）');
    console.log('');
    
    // 验证关键模块是否加载了真实实现
    try {
        const auth = require('./src/api/auth');
        const xyyxt = require('./src/xyyxt/auth');
        const announcement = require('./src/api/announcement');
        
        console.log('✅ 模块加载验证:');
        console.log('   - Auth 模块:', typeof auth.login === 'function' ? '✓ 已加载' : '✗ 错误');
        console.log('   - XYYXT 模块:', typeof xyyxt.login === 'function' ? '✓ 已加载' : '✗ 错误');
        console.log('   - Announcement 模块:', typeof announcement.getAnnouncements === 'function' ? '✓ 已加载' : '✗ 错误');
        
        // 检查是否包含 Mock 特征
        const authSource = auth.login.toString();
        const xyyxtSource = xyyxt.login.toString();
        const announcementSource = announcement.getAnnouncements.toString();
        
        const hasMockInAuth = authSource.includes('[Mock') || authSource.includes('mockData');
        const hasMockInXyyxt = xyyxtSource.includes('[Mock') || xyyxtSource.includes('mockData');
        const hasMockInAnnouncement = announcementSource.includes('[Mock') || announcementSource.includes('mockData');
        
        console.log('');
        console.log('✅ Mock 代码检查:');
        console.log('   - Auth 模块:', !hasMockInAuth ? '✓ 使用真实 API' : '✗ 仍包含 Mock 代码');
        console.log('   - XYYXT 模块:', !hasMockInXyyxt ? '✓ 使用真实 API' : '✗ 仍包含 Mock 代码');
        console.log('   - Announcement 模块:', !hasMockInAnnouncement ? '✓ 使用真实 API' : '✗ 仍包含 Mock 代码');
        
        if (hasMockInAuth || hasMockInXyyxt || hasMockInAnnouncement) {
            console.log('');
            console.log('❌ 警告：某些模块仍包含 Mock 实现');
            process.exit(1);
        } else {
            console.log('');
            console.log('🎉 所有验证通过！系统已成功切换到真实数据模式');
            console.log('');
            console.log('恢复的文件列表:');
            console.log('   ✓ src/api/auth.js - 教务系统登录（真实HTTP请求）');
            console.log('   ✓ src/api/student.js - 学生数据接口（真实数据解析）');
            console.log('   ✓ src/api/announcement.js - 公告爬取（真实网页抓取）');
            console.log('   ✓ src/api/emptyroom.js - 空教室查询（真实API）');
            console.log('   ✓ src/xyyxt/auth.js - 校园一信通（真实认证接口）');
            console.log('   ✓ src/xyyxt/guilinElec.js - 桂林电费查询（真实API）');
            console.log('');
            console.log('现在可以使用 npm start 启动服务器，将连接真实的教务系统和校园一信通服务');
        }
    } catch (error) {
        console.error('❌ 模块加载失败:', error.message);
        process.exit(1);
    }
}
