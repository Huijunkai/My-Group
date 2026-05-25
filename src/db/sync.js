const { Student, Course, Grade, Exam, Plan, Progress } = require('./models');
const { sequelize, checkHealth } = require('./index');
const { 
    encryptStudentInfo, 
    encryptCourse, 
    encryptGrade, 
    encryptExam, 
    encryptPlan, 
    encryptProgress,
    encrypt 
} = require('../utils/encryption');

const DB_MAX_RETRIES = 2;
const DB_RETRY_DELAY = 2000;

async function withDbRetry(fn, operationName) {
    let lastError;
    for (let attempt = 1; attempt <= DB_MAX_RETRIES; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            const isRetryable = error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT' ||
                error.name === 'SequelizeConnectionError' || 
                (error.original && error.original.code === 'PROTOCOL_CONNECTION_LOST');
            
            if (!isRetryable || attempt === DB_MAX_RETRIES) {
                throw error;
            }
            
            console.warn(`${operationName} 数据库操作失败 (第${attempt}次)，${DB_RETRY_DELAY}ms后重试...`);
            await new Promise(resolve => setTimeout(resolve, DB_RETRY_DELAY));
        }
    }
    throw lastError;
}

async function syncStudent(studentId, info, semesterStartDate = null) {
    if (!studentId || !info) return;
    
    return withDbRetry(async () => {
        const existing = await Student.findByPk(studentId);
        const hasSemesterDate = semesterStartDate && semesterStartDate.trim && semesterStartDate.trim().length > 0;
        
        if (existing) {
            const updateData = { lastSync: new Date() };
            if (hasSemesterDate) {
                updateData.semesterStartDate = semesterStartDate.trim();
            }
            await Student.update(updateData, { where: { studentId } });
            console.log(`syncStudent: 更新学生 ${studentId} lastSync${hasSemesterDate ? ` 和 semesterStartDate=${semesterStartDate.trim()}` : ' (无开学时间)'}`);
            return;
        }
        
        const encryptedInfo = encryptStudentInfo(info);
        
        await Student.create({
            studentId,
            ...encryptedInfo,
            lastSync: new Date(),
            semesterStartDate: hasSemesterDate ? semesterStartDate.trim() : null
        });
        
        console.log(`syncStudent: 新增学生 ${studentId}${hasSemesterDate ? ` (学期开始: ${semesterStartDate.trim()})` : ' (无开学时间)'}`);
    }, 'syncStudent');
}

async function syncCourses(studentId, courses) {
    if (!studentId || !courses || !Array.isArray(courses)) return;
    
    console.log(`syncCourses: 开始同步用户 ${studentId} 的课表，课程数: ${courses.length}`);
    
    return withDbRetry(async () => {
        const { decrypt } = require('../utils/encryption');
        
        const existingCourses = await Course.findAll({
            where: { studentId }
        });
        
        console.log(`syncCourses: 数据库中现有课程数: ${existingCourses.length}`);
        
        const existingMap = new Map();
        existingCourses.forEach(course => {
            const decryptedName = decrypt(course.name) || course.name;
            const key = `${decryptedName}_${course.dayOfWeek}_${course.period}`;
            
            if (!existingMap.has(key)) {
                existingMap.set(key, []);
            }
            existingMap.get(key).push(course);
        });
        
        const newMap = new Map();
        courses.forEach(course => {
            if (!course || !course.period) return;
            const period = course.period.toString().trim();
            if (!period) return;
            
            const key = `${course.courseName}_${course.dayOfWeek}_${period}`;
            
            if (!newMap.has(key)) {
                newMap.set(key, []);
            }
            newMap.get(key).push(course);
        });
        
        let newCount = 0;
        let updateCount = 0;
        let deleteCount = 0;
        
        for (const [key, newCourseList] of newMap) {
            const existingCourseList = existingMap.get(key) || [];
            
            if (existingCourseList.length === 0) {
                for (const newCourse of newCourseList) {
                    const encryptedCourse = encryptCourse(newCourse);
                    await Course.create({
                        studentId,
                        ...encryptedCourse
                    });
                    newCount++;
                }
            } else {
                const newWeeksSet = new Set();
                const newLocation = newCourseList[0].location || '';
                const newTeacher = newCourseList[0].teacher || '';
                
                newCourseList.forEach(c => {
                    if (c.weeks) {
                        const weeksList = String(c.weeks).split(',').map(w => parseInt(w.trim())).filter(w => !isNaN(w));
                        weeksList.forEach(w => newWeeksSet.add(w));
                    }
                    if (c.week) {
                        newWeeksSet.add(c.week);
                    }
                });
                
                const newWeeks = Array.from(newWeeksSet).sort((a, b) => a - b).join(',');
                
                const existing = existingCourseList[0];
                const existingLocation = decrypt(existing.location) || '';
                const existingWeeks = decrypt(existing.weeks) || '';
                
                if (newLocation !== existingLocation || newWeeks !== existingWeeks) {
                    await existing.update({
                        location: encrypt(newLocation),
                        weeks: newWeeks ? encrypt(newWeeks) : null,
                        week: newCourseList[0].week,
                        teacher: encrypt(newTeacher),
                        raw: JSON.stringify(newCourseList[0])
                    });
                    updateCount++;
                }
                
                for (let i = 1; i < existingCourseList.length; i++) {
                    await existingCourseList[i].destroy();
                    deleteCount++;
                }
                
                existingMap.delete(key);
            }
        }
        
        for (const [key, oldCourseList] of existingMap) {
            for (const oldCourse of oldCourseList) {
                await oldCourse.destroy();
                deleteCount++;
            }
        }
        
        console.log(`syncCourses: 同步完成 - 新增:${newCount}, 更新:${updateCount}, 删除:${deleteCount}`);
    }, 'syncCourses');
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
