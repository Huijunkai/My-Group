const jw = require('../src/index');

async function test() {
    console.log('开始测试拆分后的模块...');
    
    // 请替换为真实的学号和密码进行测试
    const username = '23490329';
    const password = '021219Hjk!';

    const loginResult = await jw.login(username, password);
    console.log('登录结果:', loginResult.success ? '成功' : '失败');

    if (loginResult.success) {
        const cookies = loginResult.cookies;

        // 1. 学生信息
        const info = await jw.getStudentInfo(cookies);
        console.log('学生姓名:', info ? info.name : '获取失败');

        // 2. 课表
        const timetable = await jw.getTimetable(cookies);
        console.log('课表课程数:', timetable ? timetable.length : '获取失败');

        // 3. 成绩
        const grades = await jw.getGrades(cookies);
        console.log('成绩学期数:', grades ? Object.keys(grades).length : '获取失败');

        // 4. 考试安排
        const exams = await jw.getExamSchedule(cookies);
        console.log('考试安排数:', exams ? exams.length : '获取失败');

        // 5. 学习进度
        const progress = await jw.getStudyProgress(cookies);
        console.log('学习进度条目:', progress ? progress.length : '获取失败');
    }
}

test();
