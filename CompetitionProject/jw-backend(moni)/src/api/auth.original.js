const { mockStudents } = require('../mockData');

async function login(username, password) {
    try {
        console.log(`[Mock Auth] 尝试登录: ${username}`);
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const student = mockStudents.find(s => s.studentId === username && s.password === password);
        
        if (student) {
            console.log(`[Mock Auth] 登录成功: ${student.name} (${student.studentId})`);
            
            const mockCookies = [
                `JSESSIONID=MOCK_${Date.now()}_${Math.random().toString(36).substr(2, 9)}; Path=/`,
                `studentId=${student.studentId}; Path=/`
            ];
            
            return {
                success: true,
                cookies: mockCookies,
                nextUrl: '/framework/xsMain.jsp',
                studentInfo: student
            };
        } else {
            const existsStudent = mockStudents.find(s => s.studentId === username);
            if (!existsStudent) {
                console.log(`[Mock Auth] 用户不存在: ${username}`);
                return { success: false, message: '该学号不存在' };
            } else {
                console.log(`[Mock Auth] 密码错误: ${username}`);
                return { success: false, message: '密码错误，请重新输入' };
            }
        }
    } catch (error) {
        console.error('[Mock Auth] 登录异常:', error.message);
        return { success: false, message: error.message };
    }
}

module.exports = { login };
