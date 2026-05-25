const axios = require('axios');
const { DEFAULT_HEADERS } = require('./constants');

/**
 * 格式化 Cookie 数组为字符串
 * @param {string[]} cookies 
 * @returns {string}
 */
function formatCookies(cookies) {
    if (!cookies) return '';
    return cookies.map(c => c.split(';')[0]).join('; ');
}

/**
 * 创建带有默认配置的 axios 实例
 * @param {string[]} cookies 
 * @param {string} referer 
 * @param {number} maxRedirects
 * @returns {import('axios').AxiosInstance}
 */
function createInstance(cookies = [], referer = '', maxRedirects = 0) {
    const headers = { ...DEFAULT_HEADERS };
    if (cookies.length > 0) {
        headers['Cookie'] = formatCookies(cookies);
    }
    if (referer) {
        headers['Referer'] = referer;
    }

    return axios.create({
        headers,
        // 默认 0：保留登录接口用 302 判断的逻辑
        // 某些页面（如课表）会 302 再 200，这种场景需要把 maxRedirects 传大一点
        maxRedirects,
        validateStatus: (status) => status >= 200 && status < 400
    });
}

module.exports = {
    formatCookies,
    createInstance
};
