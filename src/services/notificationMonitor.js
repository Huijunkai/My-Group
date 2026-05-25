const pushService = require('./pushService');
const { getGrades, getExamSchedule, getTimetable } = require('../api/student');
const { getAnnouncements } = require('../api/announcement');

const userGradeCache = new Map();
const userExamCache = new Map();
const userTimetableCache = new Map();
const knownAnnouncementIds = new Set();
const examReminderSent = new Map();

const CHECK_INTERVAL = 30 * 60 * 1000;
const ANNOUNCEMENT_CHECK_INTERVAL = 10 * 60 * 1000;
const CACHE_CLEANUP_INTERVAL = 60 * 60 * 1000;
const CACHE_EXPIRY_MS = 24 * 60 * 60 * 1000;
const EXAM_REMINDER_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000;
const KEYWORDS = ['重修', '补考', '体质健康测试', '选课', '补修', '免修'];

function startMonitoring() {
    console.log('NotificationMonitor: Starting notification monitoring service...');
    
    setInterval(async () => {
        await checkAllUsers();
    }, CHECK_INTERVAL);
    
    setInterval(async () => {
        await checkAnnouncements();
    }, ANNOUNCEMENT_CHECK_INTERVAL);
    
    setInterval(() => {
        cleanupCaches();
    }, CACHE_CLEANUP_INTERVAL);
    
    checkAllUsers();
    checkAnnouncements();
}

function cleanupCaches() {
    console.log('[缓存清理] 开始清理过期缓存...');
    
    const now = Date.now();
    let cleanedGrades = 0;
    let cleanedExams = 0;
    let cleanedTimetables = 0;
    let cleanedReminders = 0;
    
    for (const [studentId, timestamp] of userGradeCache.entries()) {
        if (now - timestamp > CACHE_EXPIRY_MS) {
            userGradeCache.delete(studentId);
            cleanedGrades++;
        }
    }
    
    for (const [studentId, timestamp] of userExamCache.entries()) {
        if (now - timestamp > CACHE_EXPIRY_MS) {
            userExamCache.delete(studentId);
            cleanedExams++;
        }
    }
    
    for (const [studentId, timestamp] of userTimetableCache.entries()) {
        if (now - timestamp > CACHE_EXPIRY_MS) {
            userTimetableCache.delete(studentId);
            cleanedTimetables++;
        }
    }
    
    for (const [key, timestamp] of examReminderSent.entries()) {
        if (now - timestamp > EXAM_REMINDER_EXPIRY_MS) {
            examReminderSent.delete(key);
            cleanedReminders++;
        }
    }
    
    console.log(`[缓存清理] 完成 - 成绩:${cleanedGrades} 考试:${cleanedExams} 课表:${cleanedTimetables} 提醒:${cleanedReminders}`);
    console.log(`[缓存状态] 当前大小 - 成绩:${userGradeCache.size} 考试:${userExamCache.size} 课表:${userTimetableCache.size} 提醒:${examReminderSent.size}`);
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
        console.log(`[成绩检查] 开始检查用户 ${studentId} 的成绩...`);
        const grades = await getGrades(userData.cookies);
        if (!grades || !grades.data) {
            console.log(`[成绩检查] 用户 ${studentId} 无成绩数据`);
            return;
        }
        
        const currentGrades = JSON.stringify(grades.data);
        const cachedGrades = userGradeCache.get(studentId);
        
        if (cachedGrades) {
            const newGrades = findNewGrades(cachedGrades, currentGrades);
            
            if (newGrades.length > 0) {
                console.log(`[成绩检查] 发现 ${newGrades.length} 条新成绩 for ${studentId}`);
                
                let successCount = 0;
                let failCount = 0;
                
                for (const grade of newGrades) {
                    const result = await pushService.notifyNewGrade(studentId, {
                        courseName: grade.courseName,
                        score: grade.score,
                        credit: grade.credit,
                        semester: grade.semester
                    });
                    
                    if (result.success) {
                        successCount++;
                        console.log(`[成绩推送] ✅ ${grade.courseName}: ${grade.score}分`);
                    } else {
                        failCount++;
                        console.error(`[成绩推送] ❌ ${grade.courseName}: ${result.message}`);
                    }
                }
                
                console.log(`[成绩推送] 完成 - 成功:${successCount} 失败:${failCount}`);
            } else {
                console.log(`[成绩检查] 用户 ${studentId} 无新成绩`);
            }
        } else {
            console.log(`[成绩检查] 用户 ${studentId} 首次检查，缓存成绩数据`);
        }
        
        userGradeCache.set(studentId, currentGrades);
    } catch (error) {
        console.error(`[成绩检查] 用户 ${studentId} 检查失败:`, error.message);
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
        console.log(`[考试检查] 开始检查用户 ${studentId} 的考试安排...`);
        const exams = await getExamSchedule(userData.cookies);
        if (!exams || !exams.data) {
            console.log(`[考试检查] 用户 ${studentId} 无考试数据`);
            return;
        }
        
        const currentExams = JSON.stringify(exams.data);
        const cachedExams = userExamCache.get(studentId);
        
        if (cachedExams) {
            const newExams = findNewExams(cachedExams, currentExams);
            
            if (newExams.length > 0) {
                console.log(`[考试检查] 发现 ${newExams.length} 个新考试 for ${studentId}`);
                
                let successCount = 0;
                let failCount = 0;
                
                for (const exam of newExams) {
                    const result = await pushService.notifyNewExam(studentId, {
                        courseName: exam.courseName,
                        examTime: exam.examTime,
                        location: exam.location
                    });
                    
                    if (result.success) {
                        successCount++;
                        console.log(`[考试推送] ✅ ${exam.courseName}: ${exam.examTime}`);
                    } else {
                        failCount++;
                        console.error(`[考试推送] ❌ ${exam.courseName}: ${result.message}`);
                    }
                }
                
                console.log(`[考试推送] 完成 - 成功:${successCount} 失败:${failCount}`);
            } else {
                console.log(`[考试检查] 用户 ${studentId} 无新考试`);
            }
            
            await checkUpcomingExams(studentId, JSON.parse(currentExams));
        } else {
            console.log(`[考试检查] 用户 ${studentId} 首次检查，缓存考试数据`);
        }
        
        userExamCache.set(studentId, currentExams);
    } catch (error) {
        console.error(`[考试检查] 用户 ${studentId} 检查失败:`, error.message);
    }
}

function findNewExams(oldExamsStr, newExamsStr) {
    const oldExams = JSON.parse(oldExamsStr);
    const newExams = JSON.parse(newExamsStr);
    
    const oldExamSet = new Set(oldExams.map(e => `${e.courseName}_${e.examTime}`));
    
    return newExams.filter(e => !oldExamSet.has(`${e.courseName}_${e.examTime}`));
}

async function checkUpcomingExams(studentId, exams) {
    console.log(`[考前提醒] 检查用户 ${studentId} 的即将到来的考试...`);
    
    const now = new Date();
    const oneDayMs = 24 * 60 * 60 * 1000;
    let reminderCount = 0;
    let skipCount = 0;
    
    for (const exam of exams) {
        try {
            const examDate = parseExamDate(exam.examTime);
            if (!examDate) {
                console.log(`[考前提醒] 无法解析考试时间: ${exam.courseName} - ${exam.examTime}`);
                continue;
            }
            
            const timeDiff = examDate.getTime() - now.getTime();
            
            if (timeDiff > 0 && timeDiff <= oneDayMs) {
                const reminderKey = `${studentId}_${exam.courseName}_${exam.examTime}`;
                
                if (examReminderSent.has(reminderKey)) {
                    console.log(`[考前提醒] ⏭️ 跳过已发送: ${exam.courseName}`);
                    skipCount++;
                    continue;
                }
                
                const result = await pushService.notifyExamReminder(studentId, {
                    courseName: exam.courseName,
                    examTime: exam.examTime,
                    location: exam.location,
                    reminderTime: '24小时后'
                });
                
                if (result.success) {
                    examReminderSent.set(reminderKey, Date.now());
                    reminderCount++;
                    console.log(`[考前提醒] ✅ 已发送: ${exam.courseName} - ${exam.examTime}`);
                } else {
                    console.error(`[考前提醒] ❌ 发送失败: ${exam.courseName} - ${result.message}`);
                }
            }
        } catch (e) {
            console.error('[考前提醒] 处理考试时出错:', e);
        }
    }
    
    if (reminderCount > 0 || skipCount > 0) {
        console.log(`[考前提醒] 完成 - 发送:${reminderCount} 跳过:${skipCount}`);
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
        console.log(`[课表检查] 开始检查用户 ${studentId} 的课表...`);
        const timetable = await getTimetable(userData.cookies);
        if (!timetable || !timetable.data) {
            console.log(`[课表检查] 用户 ${studentId} 无课表数据`);
            return;
        }
        
        const currentTimetable = JSON.stringify(timetable.data);
        const cachedTimetable = userTimetableCache.get(studentId);
        
        if (cachedTimetable) {
            const changes = detectTimetableChanges(cachedTimetable, currentTimetable);
            
            if (changes.length > 0) {
                console.log(`[课表检查] 发现 ${changes.length} 处课表变更 for ${studentId}`);
                
                let successCount = 0;
                let failCount = 0;
                
                for (const change of changes) {
                    const result = await pushService.notifyCourseChange(studentId, {
                        type: change.type,
                        courseName: change.courseName,
                        message: change.message
                    });
                    
                    if (result.success) {
                        successCount++;
                        console.log(`[课表推送] ✅ ${change.type}: ${change.courseName}`);
                    } else {
                        failCount++;
                        console.error(`[课表推送] ❌ ${change.type}: ${change.courseName} - ${result.message}`);
                    }
                    
                    if (change.type === 'new' && change.courseInfo) {
                        await checkImmediateBeforeClassReminder(studentId, change.courseInfo, userData);
                    }
                }
                
                console.log(`[课表推送] 完成 - 成功:${successCount} 失败:${failCount}`);
            } else {
                console.log(`[课表检查] 用户 ${studentId} 课表无变更`);
            }
        } else {
            console.log(`[课表检查] 用户 ${studentId} 首次检查，缓存课表数据`);
        }
        
        userTimetableCache.set(studentId, currentTimetable);
    } catch (error) {
        console.error(`[课表检查] 用户 ${studentId} 检查失败:`, error.message);
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
                message: `新增课程：${course.courseName}，第${course.week}周 ${getDayName(course.dayOfWeek)} ${course.startTime}`,
                courseInfo: course
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

async function checkImmediateBeforeClassReminder(studentId, courseInfo, userData) {
    try {
        console.log(`[新增课程提醒] 检查是否需要立即发送课前提醒: ${courseInfo.courseName}`);
        
        const now = new Date();
        const currentDay = now.getDay();
        const currentDayOfWeek = currentDay === 0 ? 7 : currentDay;
        const currentHour = now.getHours();
        const currentMinute = now.getMinutes();
        const currentTotalMinutes = currentHour * 60 + currentMinute;
        
        if (courseInfo.dayOfWeek !== currentDayOfWeek) {
            console.log(`[新增课程提醒] 课程不在今天，跳过立即提醒`);
            return;
        }
        
        const courseReminderPush = require('./courseReminderPush');
        const userConfig = await courseReminderPush.getOrCreateUserConfig(studentId);
        
        if (!userConfig.enabled || !userConfig.remindBeforeClass) {
            console.log(`[新增课程提醒] 用户未开启课前提醒`);
            return;
        }
        
        const courseStartMinutes = getCourseStartMinutesFromPeriod(courseInfo.startTime);
        const minutesUntilClass = courseStartMinutes - currentTotalMinutes;
        
        if (minutesUntilClass <= 0) {
            console.log(`[新增课程提醒] 课程已经开始或已结束，跳过`);
            return;
        }
        
        if (minutesUntilClass > userConfig.beforeClassMinutes) {
            console.log(`[新增课程提醒] 课程距离上课还有 ${minutesUntilClass} 分钟，超过提醒阈值 ${userConfig.beforeClassMinutes} 分钟，跳过`);
            return;
        }
        
        console.log(`[新增课程提醒] ✅ 课程在提醒窗口内，立即发送课前提醒`);
        
        const result = await pushService.notifyCourseReminder(studentId, {
            type: 'before_class',
            courseName: courseInfo.courseName,
            location: courseInfo.location,
            startTime: courseInfo.startTime,
            message: `${courseInfo.courseName} 将在${minutesUntilClass}分钟后开始\n地点: ${courseInfo.location || '待定'}\n时间: ${courseInfo.startTime || '-'}`
        }, userData.pushToken);
        
        if (result.success) {
            console.log(`[新增课程提醒] ✅ 课前提醒发送成功: ${courseInfo.courseName}`);
        } else {
            console.error(`[新增课程提醒] ❌ 课前提醒发送失败: ${result.message}`);
        }
    } catch (error) {
        console.error(`[新增课程提醒] 检查失败:`, error.message);
    }
}

function getCourseStartMinutesFromPeriod(period) {
    const scheduleMap = {
        '1-2': 510,
        '3-4': 625,
        '5-6': 870,
        '7-8': 975,
        '9-10': 1100,
        '11-12': 1205
    };
    
    if (scheduleMap[period]) {
        return scheduleMap[period];
    }
    
    if (period && period.includes(':')) {
        const parts = period.split(':');
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }
    
    return 510;
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

        const today = new Date();
        const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
        
        for (const announcement of announcements) {
            const announcementId = announcement.url || announcement.title;
            
            if (!knownAnnouncementIds.has(announcementId)) {
                knownAnnouncementIds.add(announcementId);
                
                if (announcement.date && announcement.date !== todayStr) {
                    console.log(`NotificationMonitor: 跳过非当日公告 (${announcement.date}): ${announcement.title}`);
                    continue;
                }
                
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
