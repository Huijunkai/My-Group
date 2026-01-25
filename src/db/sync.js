const { Student, Course, Grade, Exam, Plan, Progress } = require('./models');
const { sequelize } = require('./index');

// 修正模型定义中的 sequelize 引用
// 之前在 models/index.js 中引用的是 ../index，现在我们要确保模型加载正确

/**
 * 同步学生信息
 */
async function syncStudent(studentId, info) {
    if (!studentId || !info) return;
    await Student.upsert({
        studentId,
        ...info,
        lastSync: new Date()
    });
}

/**
 * 同步课表
 */
async function syncCourses(studentId, courses) {
    if (!studentId || !courses || !Array.isArray(courses)) return;
    
    // 关键优化：
    // 1) period 已纳入主键，避免同名课程不同节次被覆盖
    // 2) 使用 bulkCreate + updateOnDuplicate 减少数据库往返，降低同步 500/超时概率
    const rows = [];
    for (const course of courses) {
        if (!course) continue;
        const period = (course.period || '').toString().trim();
        // period 是主键之一，没有则无法正确定位到课表格子，直接跳过
        if (!period) continue;
        rows.push({
            studentId,
            semester: course.semester,
            name: course.name,
            dayOfWeek: course.dayOfWeek,
            period: period,
            teacher: course.teacher,
            weeks: course.weeks,
            location: course.location,
            raw: course.raw
        });
    }
    if (rows.length === 0) return;

    await Course.bulkCreate(rows, {
        updateOnDuplicate: ['teacher', 'weeks', 'location', 'raw']
    });
}

/**
 * 同步成绩
 */
async function syncGrades(studentId, gradesGrouped) {
    if (!studentId || !gradesGrouped) return;

    // 将成绩字符串转换为可比较的数值（用于保留最高分）
    const scoreToNumber = (score) => {
        if (score === null || score === undefined) return -1;
        const s = String(score).trim();
        const n = parseFloat(s);
        if (!Number.isNaN(n)) return n;

        // 常见等级制映射（可按学校规则微调）
        const map = {
            '优秀': 95,
            '良好': 85,
            '中等': 75,
            '及格': 65,
            '合格': 60,
            '通过': 60,
            '不及格': 0,
            '未通过': 0,
            '缺考': -1,
            '缓考': -1
        };
        for (const key of Object.keys(map)) {
            if (s.includes(key)) return map[key];
        }
        return -1;
    };

    for (const semester in gradesGrouped) {
        for (const grade of gradesGrouped[semester]) {
            if (!grade || !grade.courseCode) continue;

            const where = { studentId, semester, courseCode: grade.courseCode };
            const existing = await Grade.findOne({ where });

            if (!existing) {
                await Grade.create({
                    studentId,
                    semester,
                    ...grade
                });
                continue;
            }

            // 同一学期同一课程编号（补考/重修会重复出现）——只保留最高成绩
            const oldScore = scoreToNumber(existing.score);
            const newScore = scoreToNumber(grade.score);
            if (newScore > oldScore) {
                await existing.update({
                    courseName: grade.courseName,
                    score: grade.score,
                    credit: grade.credit,
                    gradePoint: grade.gradePoint,
                    courseType: grade.courseType,
                    examType: grade.examType
                });
            }
        }
    }
}

/**
 * 同步考试安排
 */
async function syncExams(studentId, exams) {
    if (!studentId || !exams || !Array.isArray(exams)) return;
    
    for (const exam of exams) {
        await Exam.upsert({
            studentId,
            ...exam
        });
    }
}

/**
 * 同步学期计划
 */
async function syncPlans(studentId, plansGrouped) {
    if (!studentId || !plansGrouped) return;
    
    for (const semester in plansGrouped) {
        for (const plan of plansGrouped[semester]) {
            await Plan.upsert({
                studentId,
                semester,
                ...plan
            });
        }
    }
}

/**
 * 同步学习进度
 */
async function syncProgress(studentId, progressData) {
    if (!studentId || !progressData || !Array.isArray(progressData)) return;
    
    for (const item of progressData) {
        await Progress.upsert({
            studentId,
            ...item
        });
    }
}

module.exports = {
    syncStudent,
    syncCourses,
    syncGrades,
    syncExams,
    syncPlans,
    syncProgress
};