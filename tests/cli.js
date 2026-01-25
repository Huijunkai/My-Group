const readline = require('readline');
const jw = require('../src/index');
const { initDatabase } = require('../src/db');
const { syncStudent, syncCourses, syncGrades, syncExams } = require('../src/db/sync');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

let currentUser = {
    username: '',
    cookies: [],
    info: null
};

async function start() {
    // 初始化数据库
    await initDatabase();

    console.log('\n' + '='.repeat(40));
    console.log('   强智教务系统 命令行版 (v1.0)');
    console.log('='.repeat(40));
    
    const username = await question('请输入学号: ');
    const password = await question('请输入密码: ');

    currentUser.username = username;

    console.log('\n🚀 正在尝试登录...');
    const result = await jw.login(username, password);

    if (result.success) {
        currentUser.cookies = result.cookies;
        currentUser.info = await jw.getStudentInfo(result.cookies);
        
        console.log(`\n✅ 登录成功！欢迎你，${currentUser.info ? currentUser.info.name : '同学'}`);
        
        // 登录成功后立即启动静默全量同步
        if (currentUser.info) {
            console.log('⏳ 正在同步数据到云端...');
            syncAllData(true).then(() => {
                console.log('✅ 云端数据同步完成');
            }).catch(err => {
                console.error('❌ 云端同步失败:', err.message);
            });
            
            console.log('┌' + '-'.repeat(38) + '┐');
            console.log(`| 学院: ${currentUser.info.college.padEnd(20)}`);
            console.log(`| 专业: ${currentUser.info.major.padEnd(20)}`);
            console.log(`| 班级: ${currentUser.info.className.padEnd(20)}`);
            console.log('└' + '-'.repeat(38) + '┘');
        }
        
        await mainMenu();
    } else {
        console.log(`\n❌ 登录失败: ${result.message}`);
        const retry = await question('是否重试？(y/n): ');
        if (retry.toLowerCase() === 'y') start();
        else process.exit();
    }
}

async function mainMenu() {
    while (true) {
        console.log('\n' + '━'.repeat(20));
        console.log('  📜 功能菜单');
        console.log('  1. 📅 查周课表 (日历视图)');
        console.log('  2. 📊 查成绩 (按学期)');
        console.log('  3. 📝 查考试安排');
        console.log('  4. 📋 查学期计划');
        console.log('  5. 📈 查学习完成情况');
        console.log('  6. 🔄 同步所有数据到云端');
        console.log('  0. 🚪 退出系统');
        console.log('━'.repeat(20));
        
        const choice = await question('请选择功能 (0-6): ');

        switch (choice) {
            case '1': await showTimetable(); break;
            case '2': await showGrades(); break;
            case '3': await showExams(); break;
            case '4': await showPlans(); break;
            case '5': await showProgress(); break;
            case '6': await syncAllData(); break;
            case '0':
                console.log('\n感谢使用，再见！');
                process.exit();
            default:
                console.log('\n⚠️ 无效选择，请重新输入');
        }
    }
}

async function syncAllData(silent = false) {
    if (!silent) console.log('\n正在开始全量同步...');
    
    try {
        // 1. 同步基本信息
        if (currentUser.info) {
            await syncStudent(currentUser.username, currentUser.info);
        }

        // 2. 同步课表
        const timetable = await jw.getTimetable(currentUser.cookies);
        if (timetable) {
            await syncCourses(currentUser.username, timetable);
        }

        // 3. 同步成绩
        const grades = await jw.getGrades(currentUser.cookies);
        if (grades) {
            await syncGrades(currentUser.username, grades);
        }

        // 4. 同步考试
        const exams = await jw.getExamSchedule(currentUser.cookies);
        if (exams) {
            await syncExams(currentUser.username, exams);
        }

        if (!silent) console.log('\n✨ 所有数据已成功同步到云端数据库！');
    } catch (error) {
        if (!silent) console.error('\n❌ 同步过程中出错:', error.message);
        throw error;
    }
}

// --- 格式化输出函数 ---

async function showTimetable() {
    console.log('\n正在获取课表数据...');
    const data = await jw.getTimetable(currentUser.cookies);
    if (!data || data.length === 0) return console.log('❌ 未找到课程信息');
    
    // 异步静默同步，不阻塞 UI 显示
    syncCourses(currentUser.username, data).catch(() => {});

    console.log(`\n📅 [${data[0].semester}] 周课表视图`);
    console.log('='.repeat(80));

    const weekDays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
    
    weekDays.forEach(day => {
        const dayCourses = data.filter(c => c.dayOfWeek === day);
        
        if (dayCourses.length > 0) {
            console.log(`\n📌 ${day}`);
            console.log('-'.repeat(90));
            
            // 1. 深度排序逻辑
            dayCourses.sort((a, b) => {
                // 提取起始节次数字
                const secA = parseInt(a.weeks.match(/\[(\d+)/)?.[1] || 0);
                const secB = parseInt(b.weeks.match(/\[(\d+)/)?.[1] || 0);
                
                if (secA !== secB) return secA - secB; // 优先按节次排

                // 如果节次相同，按起始周数排
                const weekA = parseInt(a.weeks.match(/(\d+)/)?.[1] || 0);
                const weekB = parseInt(b.weeks.match(/(\d+)/)?.[1] || 0);
                return weekA - weekB;
            });

            // 2. 冲突检测与渲染
            for (let i = 0; i < dayCourses.length; i++) {
                const c = dayCourses[i];
                const sectionMatch = c.weeks.match(/\[\d+[-\d]*节\]/);
                const section = sectionMatch ? sectionMatch[0] : '[未知节次]';
                const weekRange = c.weeks.replace(section, '').trim();

                // 检查与上一门课是否冲突 (节次相同且周数有重叠)
                let conflictTag = '';
                if (i > 0) {
                    const prev = dayCourses[i - 1];
                    const prevSection = prev.weeks.match(/\[\d+[-\d]*节\]/)?.[0];
                    if (section === prevSection && section !== '[未知节次]') {
                        conflictTag = ' ⚠️ [冲突/合班]';
                    }
                }
                
                const output = `${section.padEnd(12)} | ${c.name.padEnd(25)} | ${c.location.padEnd(15)} | ${weekRange}`;
                console.log(conflictTag ? `\x1b[33m${output}${conflictTag}\x1b[0m` : output);
            }
        }
    });
    console.log('\n' + '='.repeat(80));
}

async function showGrades() {
    const data = await jw.getGrades(currentUser.cookies);
    if (!data) return console.log('❌ 获取成绩失败');

    // 异步静默同步
    syncGrades(currentUser.username, data).catch(() => {});

    const semesters = Object.keys(data);
    console.log('\n🎓 可选学期:', semesters.join(' | '));
    const target = await question('请输入学期名称 (直接回车查看全部): ');

    const displayData = target ? { [target]: data[target] } : data;

    for (const sem in displayData) {
        if (!displayData[sem]) continue;
        console.log(`\n【学期: ${sem}】`);
        console.log('-'.repeat(70));
        console.log('课程名称'.padEnd(25), '成绩'.padEnd(8), '学分'.padEnd(6), '绩点'.padEnd(6), '考核方式');
        console.log('-'.repeat(70));
        displayData[sem].forEach(g => {
            console.log(`${g.courseName.padEnd(25)} ${g.score.padEnd(8)} ${g.credit.padEnd(6)} ${g.gradePoint.padEnd(6)} ${g.examType}`);
        });
    }
}

async function showExams() {
    console.log('\n正在获取考试安排...');
    const data = await jw.getExamSchedule(currentUser.cookies);
    if (!data || data.length === 0) return console.log('📭 暂无考试安排');

    // 异步静默同步
    syncExams(currentUser.username, data).catch(() => {});

    console.log('\n📝 考试时间表');
    console.log('='.repeat(80));
    data.forEach((e, i) => {
        console.log(`${(i + 1).toString().padStart(2, '0')}. ${e.courseName.padEnd(25)}`);
        console.log(`    ⏰ 时间: ${e.examTime.padEnd(30)} 📍 地点: ${e.location}`);
        console.log(`    🪑 座位: ${e.seatNumber.padEnd(10)} 🏷️ 性质: ${e.examType}`);
        console.log('-'.repeat(80));
    });
}

async function showPlans() {
    const data = await jw.getSemesterPlan(currentUser.cookies);
    if (!data) return console.log('❌ 获取计划失败');

    const semesters = Object.keys(data);
    console.log('\n📋 可选学期:', semesters.join(' | '));
    const target = await question('请输入学期名称: ');

    if (data[target]) {
        console.log(`\n=== ${target} 教学计划 ===`);
        console.log('序号'.padEnd(4), '课程名称'.padEnd(25), '学分'.padEnd(6), '学时'.padEnd(6), '性质');
        console.log('-'.repeat(60));
        data[target].forEach((p, i) => {
            console.log(`${(i + 1).toString().padEnd(4)} ${p.courseName.padEnd(25)} ${p.credit.padEnd(6)} ${p.totalHours.padEnd(6)} ${p.courseType}`);
        });
    } else {
        console.log('❌ 未找到该学期计划');
    }
}

async function showProgress() {
    console.log('\n正在获取学习进度...');
    const data = await jw.getStudyProgress(currentUser.cookies);
    if (!data) return console.log('❌ 获取失败');

    console.log('\n📈 学习完成情况统计表');
    console.log('='.repeat(70));
    console.log('课程体系(属性)'.padEnd(18), '要求学分'.padEnd(10), '已修'.padEnd(8), '正修'.padEnd(8), '还需');
    console.log('-'.repeat(70));
    data.forEach(p => {
        console.log(`${p.category.padEnd(18)} ${p.requiredCredits.padEnd(10)} ${p.completedCredits.padEnd(8)} ${p.currentCredits.padEnd(8)} ${p.remainingCredits}`);
    });
    console.log('='.repeat(70));
}

start();