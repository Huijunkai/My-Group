const { login } = require('../src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule } = require('../src/api/student');
const { initDatabase } = require('../src/db');
const { syncStudent, syncCourses, syncGrades, syncExams } = require('../src/db/sync');
const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function question(query) {
    return new Promise(resolve => rl.question(query, resolve));
}

async function test() {
    try {
        // 1. 初始化数据库
        await initDatabase();

        const username = await question('请输入学号: ');
        const password = await question('请输入密码: ');

        console.log('正在登录...');
        const loginResult = await login(username, password);

        if (loginResult.success) {
            console.log('登录成功，开始同步数据...');
            const cookies = loginResult.cookies;

            // 获取并同步学生信息
            const studentInfo = await getStudentInfo(cookies);
            if (studentInfo) {
                console.log('同步学生信息:', studentInfo.name);
                await syncStudent(username, studentInfo);
            }

            // 获取并同步课表
            const timetable = await getTimetable(cookies);
            if (timetable) {
                console.log('同步课表，课程数量:', timetable.length);
                await syncCourses(username, timetable);
            }

            // 获取并同步成绩
            const grades = await getGrades(cookies);
            if (grades) {
                console.log('同步成绩，学期数量:', Object.keys(grades).length);
                await syncGrades(username, grades);
            }

            // 获取并同步考试安排
            const exams = await getExamSchedule(cookies);
            if (exams) {
                console.log('同步考试安排，数量:', exams.length);
                await syncExams(username, exams);
            }

            console.log('所有数据同步完成！');
        } else {
            console.error('登录失败:', loginResult.message);
        }
    } catch (error) {
        console.error('运行出错:', error);
    } finally {
        rl.close();
    }
}

test();
