const { BASE_URL } = require('../utils/constants');
const { createInstance } = require('../utils/request');

/**
 * 登录教务系统
 * @param {string} username 
 * @param {string} password 
 */
async function login(username, password) {
    try {
        const instance = createInstance();
        
        // 1. 获取登录页面初始 Cookie
        const initialResponse = await instance.get(`${BASE_URL}/xk/LoginToXk`);
        const initialCookies = initialResponse.headers['set-cookie'] || [];

        // 2. 构造登录数据
        const encoded = Buffer.from(username).toString('base64') + '%%%' + Buffer.from(password).toString('base64');
        const postData = new URLSearchParams();
        postData.append('encoded', encoded);

        // 3. 尝试登录
        const loginInstance = createInstance(initialCookies, `${BASE_URL}/xk/LoginToXk`);
        const loginResponse = await loginInstance.post(`${BASE_URL}/xk/LoginToXk`, postData);

        if (loginResponse.status === 302 || (loginResponse.headers['location'] && loginResponse.headers['location'].includes('xsMain.jsp'))) {
            const finalCookies = loginResponse.headers['set-cookie'] || initialCookies;
            return {
                success: true,
                cookies: finalCookies,
                nextUrl: loginResponse.headers['location']
            };
        } else {
            return { success: false, message: '登录失败，请检查学号密码' };
        }
    } catch (error) {
        return { success: false, message: error.message };
    }
}

module.exports = { login };
