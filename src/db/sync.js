const { Student, Course, Grade, Exam, Plan, Progress } = require('./models');
const { sequelize } = require('./index');
const { 
    encryptStudentInfo, 
    encryptCourse, 
    encryptGrade, 
    encryptExam, 
    encryptPlan, 
    encryptProgress,
    encrypt 
} = require('../utils/encryption');

async function syncStudent(studentId, info) {
    if (!studentId || !info) return;
    
    const existing = await Student.findByPk(studentId);
    if (existing) {
        console.log(`syncStudent: 学生 ${studentId} 已存在，跳过更新`);
        return;
    }
    
    const encryptedInfo = encryptStudentInfo(info);
    
    await Student.create({
        studentId,
        ...encryptedInfo,
        lastSync: new Date()
    });
    
    console.log(`syncStudent: 新增学生 ${studentId}`);
}

async function syncCourses(studentId, courses) {
    if (!studentId || !courses || !Array.isArray(courses)) return;
    
    const semesterSet = new Set();
    for (const course of courses) {
        if (course && course.semester) semesterSet.add(course.semester);
    }
    
    for (const sem of semesterSet) {
        const existingCount = await Course.count({ where: { studentId, semester: sem } });
        if (existingCount > 0) {
            console.log(`syncCourses: 学期 ${sem} 课程已存在，跳过更新`);
            continue;
        }
        
        const rows = [];
        for (const course of courses) {
            if (!course || course.semester !== sem) continue;
            const period = (course.period || '').toString().trim();
            if (!period) continue;
            const week = Number.isFinite(course.week) ? course.week : parseInt(String(course.week || '0'), 10);
            if (!Number.isFinite(week)) continue;
            
            const encryptedCourse = encryptCourse(course);
            
            rows.push({
                studentId,
                semester: encryptedCourse.semester,
                name: encryptedCourse.name,
                dayOfWeek: encryptedCourse.dayOfWeek,
                week: week,
                period: period,
                teacher: encryptedCourse.teacher,
                weeks: encryptedCourse.weeks,
                location: encryptedCourse.location,
                courseType: encryptedCourse.courseType,
                raw: encryptedCourse.raw
            });
        }
        
        if (rows.length > 0) {
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
            await Course.bulkCreate(dedupRows);
            console.log(`syncCourses: 新增学期 ${sem} 课程 ${dedupRows.length} 条`);
        }
    }
}

async function syncGrades(studentId, gradesGrouped) {
    if (!studentId || !gradesGrouped) return [];
    
    const newGrades = [];
    
    for (const semester in gradesGrouped) {
        for (const grade of gradesGrouped[semester]) {
            if (!grade || !grade.courseCode) continue;

            const where = { studentId, semester, courseCode: grade.courseCode };
            const existing = await Grade.findOne({ where });

            if (existing) {
                console.log(`syncGrades: 成绩 ${semester}-${grade.courseCode} 已存在，跳过更新`);
                continue;
            }

            const encryptedGrade = encryptGrade(grade);

            await Grade.create({
                studentId,
                semester,
                ...encryptedGrade
            });
            
            console.log(`syncGrades: 新增成绩 ${semester}-${grade.courseCode}`);
            
            newGrades.push({
                success: true,
                grade: grade
            });
        }
    }
    
    return newGrades;
}

async function syncExams(studentId, exams) {
    if (!studentId || !exams || !Array.isArray(exams)) {
        return [];
    }
    
    const newExams = [];
    
    for (const exam of exams) {
        if (!exam || !exam.courseName || !exam.examTime) continue;
        
        const where = { studentId, courseName: exam.courseName, examTime: exam.examTime };
        const existing = await Exam.findOne({ where });
        
        if (existing) {
            console.log(`syncExams: 考试 ${exam.courseName} 已存在，跳过更新`);
            continue;
        }
        
        const encryptedExam = encryptExam(exam);
        await Exam.create({
            studentId,
            ...encryptedExam
        });
        
        console.log(`syncExams: 新增考试 ${exam.courseName}`);
        
        newExams.push({
            success: true,
            exam: exam
        });
    }
    
    return newExams;
}

async function syncPlans(studentId, plansGrouped) {
    if (!studentId || !plansGrouped) return;
    
    for (const semester in plansGrouped) {
        for (const plan of plansGrouped[semester]) {
            if (!plan || !plan.courseCode) continue;
            
            const where = { studentId, semester, courseCode: plan.courseCode };
            const existing = await Plan.findOne({ where });
            
            if (existing) {
                console.log(`syncPlans: 计划 ${semester}-${plan.courseCode} 已存在，跳过更新`);
                continue;
            }
            
            const encryptedPlan = encryptPlan(plan);
            await Plan.create({
                studentId,
                semester,
                ...encryptedPlan
            });
            
            console.log(`syncPlans: 新增计划 ${semester}-${plan.courseCode}`);
        }
    }
}

async function syncProgress(studentId, progressData) {
    if (!studentId || !progressData || !Array.isArray(progressData)) return;
    
    for (const item of progressData) {
        if (!item || !item.category) continue;
        
        const encryptedCategory = encrypt(item.category);
        const where = { studentId, category: encryptedCategory };
        const existing = await Progress.findOne({ where });
        
        if (existing) {
            console.log(`syncProgress: 进度 ${item.category} 已存在，跳过更新`);
            continue;
        }
        
        const encryptedItem = encryptProgress({
            studentId,
            ...item
        });
        
        await Progress.create(encryptedItem);
        console.log(`syncProgress: 新增进度 ${item.category}`);
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
