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
 * @returns {import('axios').AxiosInstance}
 */
function createInstance(cookies = [], referer = '') {
    const headers = { ...DEFAULT_HEADERS };
    if (cookies.length > 0) {
        headers['Cookie'] = formatCookies(cookies);
    }
    if (referer) {
        headers['Referer'] = referer;
    }

    return axios.create({
        headers,
        maxRedirects: 0,
        validateStatus: (status) => status >= 200 && status < 400
    });
}

module.exports = {
    formatCookies,
    createInstance
};
