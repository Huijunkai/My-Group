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
    // 1) week + period 已纳入主键，避免同名课程不同周/不同节次被覆盖
    // 2) 使用 bulkCreate + updateOnDuplicate 减少数据库往返，降低同步 500/超时概率
    // 3) 按学期先清空旧缓存，避免周次变更后出现“旧周次残留”
    const semesterSet = new Set();
    for (const course of courses) {
        if (course && course.semester) semesterSet.add(course.semester);
    }
    for (const sem of semesterSet) {
        await Course.destroy({ where: { studentId, semester: sem } });
    }

    const rows = [];
    for (const course of courses) {
        if (!course) continue;
        const period = (course.period || '').toString().trim();
        // period 是主键之一，没有则无法正确定位到课表格子，直接跳过
        if (!period) continue;
        const week = Number.isFinite(course.week) ? course.week : parseInt(String(course.week || '0'), 10);
        if (!Number.isFinite(week)) continue;
        rows.push({
            studentId,
            semester: course.semester,
            name: course.name,
            dayOfWeek: course.dayOfWeek,
            week: week,
            period: period,
            teacher: course.teacher,
            weeks: course.weeks,
            location: course.location,
            raw: course.raw
        });
    }
    if (rows.length === 0) return;

    // PostgreSQL 限制：同一次 INSERT ... ON CONFLICT DO UPDATE 中，
    // 如果待插入数组里出现“相同主键”的重复行，会报错：
    // "ON CONFLICT DO UPDATE command cannot affect row a second time"
    // 因此这里必须先按主键去重（保留信息更完整的一条）。
    const dedupMap = new Map();
    for (const r of rows) {
        const key = [
            r.studentId,
            r.semester,
            r.name,
            r.dayOfWeek,
            String(r.week),
            r.period
        ].join('||');

        const prev = dedupMap.get(key);
        if (!prev) {
            dedupMap.set(key, r);
            continue;
        }

        // 合并策略：优先保留 raw 更长、location/teacher 非空的记录
        const prevRawLen = prev.raw ? String(prev.raw).length : 0;
        const nextRawLen = r.raw ? String(r.raw).length : 0;

        const merged = {
            ...prev,
            teacher: (prev.teacher && String(prev.teacher).trim()) ? prev.teacher : r.teacher,
            location: (prev.location && String(prev.location).trim()) ? prev.location : r.location,
            weeks: (prev.weeks && String(prev.weeks).trim()) ? prev.weeks : r.weeks,
            raw: nextRawLen > prevRawLen ? r.raw : prev.raw
        };

        dedupMap.set(key, merged);
    }

    const dedupRows = Array.from(dedupMap.values());

    await Course.bulkCreate(dedupRows, {
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