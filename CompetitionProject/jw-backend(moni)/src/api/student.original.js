const { mockStudents, mockTimetable, mockGrades, mockExams, mockPlans, mockProgress } = require('../mockData');

async function getStudentInfo(cookies) {
    try {
        console.log('[Mock Student] 获取学生信息');
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        const studentId = extractStudentId(cookies);
        const student = mockStudents.find(s => s.studentId === studentId);
        
        if (student) {
            return {
                studentId: student.studentId,
                name: student.name,
                gender: student.gender,
                enrollmentYear: student.enrollmentYear,
                className: student.className,
                major: student.major,
                college: student.college
            };
        }
        
        return mockStudents[0];
    } catch (error) {
        console.error('[Mock Student] 获取学生信息失败:', error.message);
        return null;
    }
}

async function getTimetable(cookies, semester = '') {
    try {
        console.log(`[Mock Student] 获取课表 - 学期: ${semester || '当前学期'}`);
        
        await new Promise(resolve => setTimeout(resolve, 400));
        
        let timetable = [...mockTimetable];
        
        if (semester) {
            timetable = timetable.filter(c => c.semester === semester);
        }
        
        return timetable;
    } catch (error) {
        console.error('[Mock Student] 获取课表失败:', error.message);
        return [];
    }
}

async function getGrades(cookies, semester = '') {
    try {
        console.log(`[Mock Student] 获取成绩 - 学期: ${semester || '所有学期'}`);
        
        await new Promise(resolve => setTimeout(resolve, 400));
        
        if (semester && mockGrades[semester]) {
            return { [semester]: mockGrades[semester] };
        }
        
        return mockGrades;
    } catch (error) {
        console.error('[Mock Student] 获取成绩失败:', error.message);
        return null;
    }
}

async function getExamSchedule(cookies, semester = '') {
    try {
        console.log(`[Mock Student] 获取考试安排 - 学期: ${semester || '当前学期'}`);
        
        await new Promise(resolve => setTimeout(resolve, 350));
        
        return mockExams;
    } catch (error) {
        console.error('[Mock Student] 获取考试安排失败:', error.message);
        return null;
    }
}

async function getSemesterPlan(cookies) {
    try {
        console.log('[Mock Student] 获取培养计划');
        
        await new Promise(resolve => setTimeout(resolve, 380));
        
        return mockPlans;
    } catch (error) {
        console.error('[Mock Student] 获取培养计划失败:', error.message);
        return null;
    }
}

async function getStudyProgress(cookies) {
    try {
        console.log('[Mock Student] 获取学习进度');
        
        await new Promise(resolve => setTimeout(resolve, 320));
        
        return mockProgress;
    } catch (error) {
        console.error('[Mock Student] 获取学习进度失败:', error.message);
        return null;
    }
}

function extractStudentId(cookies) {
    if (!cookies) return null;
    
    if (Array.isArray(cookies)) {
        for (const cookie of cookies) {
            if (cookie.includes('studentId=')) {
                const match = cookie.match(/studentId=([^;]+)/);
                if (match) return match[1];
            }
        }
    } else if (typeof cookies === 'string') {
        const match = cookies.match(/studentId=([^;]+)/);
        if (match) return match[1];
    }
    
    return mockStudents[0].studentId;
}

module.exports = {
    getStudentInfo,
    getTimetable,
    getGrades,
    getExamSchedule,
    getSemesterPlan,
    getStudyProgress
};
