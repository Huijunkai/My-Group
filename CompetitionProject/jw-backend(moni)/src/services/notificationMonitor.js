const pushService = require('./pushService');
const { getGrades, getExamSchedule, getTimetable } = require('../api/student');
const { getAnnouncements } = require('../api/announcement');

const userGradeCache = new Map();
const userExamCache = new Map();
const userTimetableCache = new Map();
const knownAnnouncementIds = new Set();

const CHECK_INTERVAL = 30 * 60 * 1000;
const ANNOUNCEMENT_CHECK_INTERVAL = 10 * 60 * 1000;
const KEYWORDS = ['重修', '补考', '体质健康测试', '选课', '补修', '免修'];

function startMonitoring() {
    console.log('NotificationMonitor: Starting notification monitoring service...');
    
    setInterval(async () => {
        await checkAllUsers();
    }, CHECK_INTERVAL);
    
    setInterval(async () => {
        await checkAnnouncements();
    }, ANNOUNCEMENT_CHECK_INTERVAL);
    
    checkAllUsers();
    checkAnnouncements();
}

const monitoredUsers = new Map();

function registerUser(studentId, cookies, pushToken) {
    monitoredUsers.set(studentId, {
        cookies: cookies,
        pushToken: pushToken,
        registeredAt: Date.now()
    });
    console.log(`NotificationMonitor: User ${studentId} registered for monitoring`);
}

function unregisterUser(studentId) {
    monitoredUsers.delete(studentId);
    userGradeCache.delete(studentId);
    userExamCache.delete(studentId);
    userTimetableCache.delete(studentId);
    console.log(`NotificationMonitor: User ${studentId} unregistered from monitoring`);
}

async function checkAllUsers() {
    console.log(`NotificationMonitor: Checking ${monitoredUsers.size} users...`);
    
    for (const [studentId, userData] of monitoredUsers) {
        try {
            await Promise.all([
                checkGradeChanges(studentId, userData),
                checkExamChanges(studentId, userData),
                checkTimetableChanges(studentId, userData)
            ]);
        } catch (error) {
            console.error(`NotificationMonitor: Error checking user ${studentId}:`, error.message);
        }
    }
}

async function checkGradeChanges(studentId, userData) {
    try {
        const grades = await getGrades(userData.cookies);
        if (!grades || !grades.data) return;
        
        const currentGrades = JSON.stringify(grades.data);
        const cachedGrades = userGradeCache.get(studentId);
        
        if (cachedGrades) {
            const newGrades = findNewGrades(cachedGrades, currentGrades);
            
            if (newGrades.length > 0) {
                console.log(`NotificationMonitor: Found ${newGrades.length} new grades for ${studentId}`);
                
                for (const grade of newGrades) {
                    await pushService.notifyNewGrade(studentId, {
                        courseName: grade.courseName,
                        score: grade.score,
                        credit: grade.credit,
                        semester: grade.semester
                    });
                }
            }
        }
        
        userGradeCache.set(studentId, currentGrades);
    } catch (error) {
        console.error(`NotificationMonitor: Error checking grades for ${studentId}:`, error.message);
    }
}

function findNewGrades(oldGradesStr, newGradesStr) {
    const oldGrades = JSON.parse(oldGradesStr);
    const newGrades = JSON.parse(newGradesStr);
    
    const oldCourseSet = new Set(oldGrades.map(g => `${g.courseName}_${g.semester}`));
    
    return newGrades.filter(g => !oldCourseSet.has(`${g.courseName}_${g.semester}`));
}

async function checkExamChanges(studentId, userData) {
    try {
        const exams = await getExamSchedule(userData.cookies);
        if (!exams || !exams.data) return;
        
        const currentExams = JSON.stringify(exams.data);
        const cachedExams = userExamCache.get(studentId);
        
        if (cachedExams) {
            const newExams = findNewExams(cachedExams, currentExams);
            
            if (newExams.length > 0) {
                console.log(`NotificationMonitor: Found ${newExams.length} new exams for ${studentId}`);
                
                for (const exam of newExams) {
                    await pushService.notifyNewExam(studentId, {
                        courseName: exam.courseName,
                        examTime: exam.examTime,
                        location: exam.location
                    });
                }
            }
            
            await checkUpcomingExams(studentId, JSON.parse(currentExams));
        }
        
        userExamCache.set(studentId, currentExams);
    } catch (error) {
        console.error(`NotificationMonitor: Error checking exams for ${studentId}:`, error.message);
    }
}

function findNewExams(oldExamsStr, newExamsStr) {
    const oldExams = JSON.parse(oldExamsStr);
    const newExams = JSON.parse(newExamsStr);
    
    const oldExamSet = new Set(oldExams.map(e => `${e.courseName}_${e.examTime}`));
    
    return newExams.filter(e => !oldExamSet.has(`${e.courseName}_${e.examTime}`));
}

async function checkUpcomingExams(studentId, exams) {
    const now = new Date();
    const oneDayMs = 24 * 60 * 60 * 1000;
    
    for (const exam of exams) {
        try {
            const examDate = parseExamDate(exam.examTime);
            if (!examDate) continue;
            
            const timeDiff = examDate.getTime() - now.getTime();
            
            if (timeDiff > 0 && timeDiff <= oneDayMs) {
                const hoursLeft = Math.floor(timeDiff / (60 * 60 * 1000));
                
                if (hoursLeft <= 24 && hoursLeft > 23) {
                    await pushService.notifyExamReminder(studentId, {
                        courseName: exam.courseName,
                        examTime: exam.examTime,
                        location: exam.location,
                        reminderTime: '24小时后'
                    });
                }
            }
        } catch (e) {
            console.error('Error parsing exam date:', e);
        }
    }
}

function parseExamDate(examTimeStr) {
    try {
        const match = examTimeStr.match(/(\d{4})-(\d{2})-(\d{2})/);
        if (match) {
            return new Date(`${match[1]}-${match[2]}-${match[3]}`);
        }
        return null;
    } catch (e) {
        return null;
    }
}

async function checkTimetableChanges(studentId, userData) {
    try {
        const timetable = await getTimetable(userData.cookies);
        if (!timetable || !timetable.data) return;
        
        const currentTimetable = JSON.stringify(timetable.data);
        const cachedTimetable = userTimetableCache.get(studentId);
        
        if (cachedTimetable) {
            const changes = detectTimetableChanges(cachedTimetable, currentTimetable);
            
            if (changes.length > 0) {
                console.log(`NotificationMonitor: Found ${changes.length} timetable changes for ${studentId}`);
                
                for (const change of changes) {
                    await pushService.notifyCourseChange(studentId, {
                        type: change.type,
                        courseName: change.courseName,
                        message: change.message
                    });
                }
            }
        }
        
        userTimetableCache.set(studentId, currentTimetable);
    } catch (error) {
        console.error(`NotificationMonitor: Error checking timetable for ${studentId}:`, error.message);
    }
}

function detectTimetableChanges(oldTimetableStr, newTimetableStr) {
    const changes = [];
    const oldTimetable = JSON.parse(oldTimetableStr);
    const newTimetable = JSON.parse(newTimetableStr);
    
    const oldCourseMap = new Map();
    oldTimetable.forEach(course => {
        const key = `${course.courseName}_${course.week}_${course.dayOfWeek}_${course.startTime}`;
        oldCourseMap.set(key, course);
    });
    
    newTimetable.forEach(course => {
        const key = `${course.courseName}_${course.week}_${course.dayOfWeek}_${course.startTime}`;
        
        if (!oldCourseMap.has(key)) {
            changes.push({
                type: 'new',
                courseName: course.courseName,
                message: `新增课程：${course.courseName}，第${course.week}周 ${getDayName(course.dayOfWeek)} ${course.startTime}`
            });
        } else {
            const oldCourse = oldCourseMap.get(key);
            if (oldCourse.location !== course.location) {
                changes.push({
                    type: 'location_change',
                    courseName: course.courseName,
                    message: `${course.courseName} 教室变更：${oldCourse.location} → ${course.location}`
                });
            }
            oldCourseMap.delete(key);
        }
    });
    
    oldCourseMap.forEach((course, key) => {
        changes.push({
            type: 'cancelled',
            courseName: course.courseName,
            message: `课程取消：${course.courseName}，第${course.week}周 ${getDayName(course.dayOfWeek)}`
        });
    });
    
    return changes;
}

function getDayName(dayOfWeek) {
    const days = ['', '周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    return days[dayOfWeek] || '';
}

async function checkAnnouncements() {
    try {
        console.log('NotificationMonitor: Checking for new announcements...');
        const result = await getAnnouncements(20);
        
        if (!result || !result.announcements || !Array.isArray(result.announcements)) {
            console.log('NotificationMonitor: No announcements found or invalid format');
            return;
        }
        
        const announcements = result.announcements;
        
        if (announcements.length === 0) {
            return;
        }
        
        for (const announcement of announcements) {
            const announcementId = announcement.url || announcement.title;
            
            if (!knownAnnouncementIds.has(announcementId)) {
                knownAnnouncementIds.add(announcementId);
                
                for (const keyword of KEYWORDS) {
                    if (announcement.title.includes(keyword)) {
                        console.log(`NotificationMonitor: Found keyword "${keyword}" in announcement: ${announcement.title}`);
                        await pushService.notifyAnnouncement(announcement.title, keyword, announcement.url);
                        break;
                    }
                }
            }
        }
        
        if (knownAnnouncementIds.size > 100) {
            const idsArray = Array.from(knownAnnouncementIds);
            knownAnnouncementIds.clear();
            idsArray.slice(-50).forEach(id => knownAnnouncementIds.add(id));
        }
    } catch (error) {
        console.error('NotificationMonitor: Error checking announcements:', error.message);
    }
}

function getMonitoringStats() {
    return {
        monitoredUsers: monitoredUsers.size,
        cachedGrades: userGradeCache.size,
        cachedExams: userExamCache.size,
        cachedTimetables: userTimetableCache.size,
        knownAnnouncements: knownAnnouncementIds.size
    };
}

module.exports = {
    startMonitoring,
    registerUser,
    unregisterUser,
    checkAllUsers,
    checkAnnouncements,
    getMonitoringStats
};
