const { BASE_URL } = require('../utils/constants');
const { createInstance } = require('../utils/request');
const parser = require('../parser');
const cheerio = require('cheerio');

/**
 * 获取学生信息
 */
async function getStudentInfo(cookies) {
    try {
        const instance = createInstance(cookies, `${BASE_URL}/framework/xsMain.jsp`);
        const response = await instance.get(`${BASE_URL}/grxx/xsxx?Ves632DSdyV=NEW_XSD_XJCJ`);
        return parser.parseStudentInfo(response.data);
    } catch (error) {
        console.error('获取学生信息失败:', error.message);
        return null;
    }
}

/**
 * 获取课表
 */
async function getTimetable(cookies, semester = '') {
    try {
        // Railway 上课表页经常会 302 -> 200；axios 自动跟随时可能丢 Cookie/Referer 导致最终变回登录页
        // 这里改成“手动跟随 302”，确保每一步都带上 Cookie
        const maxHops = 5;
        let url = `${BASE_URL}/xskb/xskb_list.do`;
        
        // 如果指定了学期，带上学期参数
        if (semester) {
            url += `?xnxq01id=${encodeURIComponent(semester)}`;
        }

        let referer = `${BASE_URL}/framework/xsMain.jsp`;
        let response = null;

        for (let i = 0; i < maxHops; i++) {
            const instance = createInstance(cookies, referer, 0);
            response = await instance.get(url);

            // 302/303：继续跟随
            if ((response.status === 302 || response.status === 303) && response.headers && response.headers.location) {
                const location = response.headers.location;
                // 处理相对/绝对跳转
                if (location.startsWith('http://') || location.startsWith('https://')) {
                    url = location;
                } else if (location.startsWith('/')) {
                    url = `${BASE_URL}${location}`;
                } else {
                    // 少见情况：相对路径
                    const base = BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL;
                    url = `${base}/${location}`;
                }
                referer = url;
                continue;
            }
            break;
        }

        const html = response && response.data ? response.data : '';
        if (!html || typeof html !== 'string') return [];

        // 基本防呆：拿到的不是课表页（例如跳回登录/空页面）就直接返回空数组
        if (!html.includes('kbtable')) return [];

        return parser.parseTimetable(html);
    } catch (error) {
        console.error('获取课表信息失败:', error.message);
        return null;
    }
}

/**
 * 获取成绩
 */
async function getGrades(cookies, semester = '') {
    try {
        const instance = createInstance(cookies, `${BASE_URL}/kscj/cjcx_query?Ves632DSdyV=NEW_XSD_XJCJ`);
        const postData = new URLSearchParams({
            kksj: semester,
            kclbm: '',
            kcmc: '',
            xsfs: 'all',
            fxjs: '0'
        });
        const response = await instance.post(`${BASE_URL}/kscj/cjcx_list`, postData);
        return parser.parseGrades(response.data);
    } catch (error) {
        console.error('获取成绩信息失败:', error.message);
        return null;
    }
}

/**
 * 获取考试安排
 */
async function getExamSchedule(cookies) {
    try {
        const instance = createInstance(cookies, `${BASE_URL}/framework/xsMain.jsp`);
        const response = await instance.get(`${BASE_URL}/xsks/xsksap_list`);
        return parser.parseExams(response.data);
    } catch (error) {
        console.error('获取考试安排失败:', error.message);
        return null;
    }
}

/**
 * 获取学期计划
 */
async function getSemesterPlan(cookies) {
    try {
        const instance = createInstance(cookies, `${BASE_URL}/framework/xsMain.jsp`);
        const response = await instance.get(`${BASE_URL}/pyfa/pyfa_query`);
        return parser.parseSemesterPlan(response.data);
    } catch (error) {
        console.error('获取学期计划失败:', error.message);
        return null;
    }
}

/**
 * 获取学习进度
 */
async function getStudyProgress(cookies) {
    try {
        const queryPageUrl = `${BASE_URL}/xxwcqk/xxwcqk_idxOntx.do`;
        const progressUrl = `${BASE_URL}/xxwcqk/xxwcqkOnkctx.do`;
        
        const instance = createInstance(cookies, `${BASE_URL}/framework/xsMain.jsp`);
        
        // 1. 获取查询页面的表单参数
        const initialResponse = await instance.get(queryPageUrl);
        const $initial = cheerio.load(initialResponse.data);
        const postData = new URLSearchParams();
        
        $initial('form input').each((i, el) => {
            const name = $initial(el).attr('name');
            const value = $initial(el).attr('value') || '';
            if (name) postData.append(name, value);
        });

        // 2. 发送请求
        const progressInstance = createInstance(cookies, queryPageUrl);
        const response = await progressInstance.post(progressUrl, postData);
        return parser.parseStudyProgress(response.data);
    } catch (error) {
        console.error('获取学习完成情况失败:', error.message);
        return null;
    }
}

module.exports = {
    getStudentInfo,
    getTimetable,
    getGrades,
    getExamSchedule,
    getSemesterPlan,
    getStudyProgress
};
