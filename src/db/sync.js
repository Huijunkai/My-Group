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
    
    // 简单起见，先删除该学生该学期的课表再重新插入（或者使用 upsert）
    // 这里假设 courses 中包含 semester 信息
    for (const course of courses) {
        // 只写入 Course 表需要的字段（避免模型字段收敛后出现未知字段报错）
        await Course.upsert({
            studentId,
            semester: course.semester,
            name: course.name,
            teacher: course.teacher,
            location: course.location,
            weeks: course.weeks,
            period: course.period,
            dayOfWeek: course.dayOfWeek,
            raw: course.raw
        });
    }
}

/**
 * 同步成绩
 */
async function syncGrades(studentId, gradesGrouped) {
    if (!studentId || !gradesGrouped) return;
    
    for (const semester in gradesGrouped) {
        for (const grade of gradesGrouped[semester]) {
            await Grade.upsert({
                studentId,
                semester,
                ...grade
            });
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