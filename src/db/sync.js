const { Student, Course, Grade, Exam } = require('./models');
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
        await Course.upsert({
            studentId,
            ...course
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

module.exports = {
    syncStudent,
    syncCourses,
    syncGrades,
    syncExams
};
