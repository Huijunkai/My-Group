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
async function getTimetable(cookies) {
    try {
        const instance = createInstance(cookies, `${BASE_URL}/framework/xsMain.jsp`);
        const response = await instance.get(`${BASE_URL}/xskb/xskb_list.do`);
        return parser.parseTimetable(response.data);
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
