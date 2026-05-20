const pushService = require('./pushService');
const { UserPushToken, Course, Student, CourseReminderConfig } = require('../db/models');
const { decryptCourse, decrypt } = require('../utils/encryption');
const nodeCron = require('node-cron');

const DAY_NAMES = ['', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
const DAY_NAMES_SHORT = ['', '周一', '周二', '周三', '周四', '周五', '周六', '周日'];

const DEFAULT_SEMESTER_START = '2025-03-03';

let beforeClassCron = null;
let tomorrowCron = null;

const REMINDER_CONFIG = {
    beforeClassMinutes: 15,
    tomorrowHour: 21,
    tomorrowMinute: 0,
    enabled: process.env.COURSE_REMINDER_ENABLED !== 'false'
};

const beforeClassSentCache = new Map();
const tomorrowSentCache = new Map();
const CACHE_EXPIRY_MINUTES = 120;

async function getOrCreateUserConfig(studentId) {
    let config = await CourseReminderConfig.findOne({ where: { studentId } });
    
    if (!config) {
        let semesterStartDate = null;
        let currentWeek = 1;
        
        try {
            const student = await Student.findByPk(studentId);
            if (student && student.semesterStartDate) {
                semesterStartDate = student.semesterStartDate;
                currentWeek = calculateCurrentWeek(semesterStartDate);
                console.log(`[课程提醒] 从Student表获取开学时间: ${semesterStartDate}, 当前周: ${currentWeek}`);
            }
        } catch (e) {
            console.log(`[课程提醒] 获取Student表开学时间失败: ${e.message}`);
        }
        
        config = await CourseReminderConfig.create({
            studentId,
            enabled: true,
            semesterStartDate: semesterStartDate,
            currentWeek: currentWeek,
            beforeClassMinutes: 15,
            remindBeforeClass: true,
            remindTomorrowCourse: true,
            tomorrowHour: 21,
            tomorrowMinute: 0
        });
        console.log(`[课程提醒] 为用户 ${studentId} 创建配置, 开学时间: ${semesterStartDate || '未设置'}, 当前周: ${currentWeek}`);
    } else if (!config.semesterStartDate) {
        try {
            const student = await Student.findByPk(studentId);
            if (student && student.semesterStartDate) {
                const currentWeek = calculateCurrentWeek(student.semesterStartDate);
                await config.update({
                    semesterStartDate: student.semesterStartDate,
                    currentWeek: currentWeek,
                    updatedAt: new Date()
                });
                console.log(`[课程提醒] 从Student表同步开学时间到配置: ${student.semesterStartDate}, 当前周: ${currentWeek}`);
            }
        } catch (e) {
            console.log(`[课程提醒] 同步Student表开学时间失败: ${e.message}`);
        }
    }
    
    return config;
}

async function updateUserConfig(studentId, updates) {
    const config = await getOrCreateUserConfig(studentId);
    
    console.log(`[课程提醒] 更新用户 ${studentId} 配置前: beforeClassMinutes=${config.beforeClassMinutes}, tomorrowHour=${config.tomorrowHour}, tomorrowMinute=${config.tomorrowMinute}`);
    
    await config.update({
        ...updates,
        updatedAt: new Date()
    });
    
    if (updates.semesterStartDate) {
        const currentWeek = calculateCurrentWeek(updates.semesterStartDate);
        await config.update({ currentWeek });
    }
    
    await config.reload();
    
    console.log(`[课程提醒] 更新用户 ${studentId} 配置后: beforeClassMinutes=${config.beforeClassMinutes}, tomorrowHour=${config.tomorrowHour}, tomorrowMinute=${config.tomorrowMinute}, semesterStartDate=${config.semesterStartDate}`);
    
    if (updates.tomorrowHour !== undefined || updates.tomorrowMinute !== undefined) {
        console.log(`[课程提醒] 用户 ${studentId} 明日提醒时间已更新为 ${config.tomorrowHour}:${String(config.tomorrowMinute).padStart(2, '0')}，将在该时间自动推送`);
    }
    
    if (updates.beforeClassMinutes !== undefined) {
        console.log(`[课程提醒] 用户 ${studentId} 课前提醒时间已更新为 ${config.beforeClassMinutes}分钟`);
    }
    
    return config;
}

function getCurrentWeekInfo() {
    const now = new Date();
    const dayOfWeek = now.getDay() === 0 ? 7 : now.getDay();
    return { now, dayOfWeek };
}

function getTodayDateString() {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

function cleanupExpiredCache(cache) {
    const now = Date.now();
    const expiryMs = CACHE_EXPIRY_MINUTES * 60 * 1000;
    let cleaned = 0;
    
    for (const [key, timestamp] of cache.entries()) {
        if (now - timestamp > expiryMs) {
            cache.delete(key);
            cleaned++;
        }
    }
    
    if (cleaned > 0) {
        console.log(`[去重缓存] 清理了 ${cleaned} 条过期记录`);
    }
}

function isBeforeClassAlreadySent(studentId, courseName, period, todayStr) {
    const key = `${studentId}_${todayStr}_${courseName}_${period}`;
    return beforeClassSentCache.has(key);
}

function markBeforeClassAsSent(studentId, courseName, period, todayStr) {
    const key = `${studentId}_${todayStr}_${courseName}_${period}`;
    beforeClassSentCache.set(key, Date.now());
    console.log(`[去重记录] 课前提醒已记录: ${key}`);
}

function isTomorrowAlreadySent(studentId, tomorrowDayOfWeek, todayStr) {
    const key = `${studentId}_${todayStr}_${tomorrowDayOfWeek}`;
    return tomorrowSentCache.has(key);
}

function markTomorrowAsSent(studentId, tomorrowDayOfWeek, todayStr) {
    const key = `${studentId}_${todayStr}_${tomorrowDayOfWeek}`;
    tomorrowSentCache.set(key, Date.now());
    console.log(`[去重记录] 明日提醒已记录: ${key}`);
}

function calculateCurrentWeek(semesterStartDate) {
    const startDate = new Date(semesterStartDate);
    const now = new Date();
    const diffTime = now.getTime() - startDate.getTime();
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const currentWeek = Math.floor(diffDays / 7) + 1;
    return Math.max(1, Math.min(25, currentWeek));
}

function getCurrentSemester(semesterStartDate) {
    const startDate = new Date(semesterStartDate);
    const year = startDate.getFullYear();
    const month = startDate.getMonth() + 1;
    
    if (month >= 2 && month <= 7) {
        return `${year - 1}-${year}-2`;
    } else if (month >= 8 && month <= 12) {
        return `${year}-${year + 1}-1`;
    } else {
        return `${year - 1}-${year}-1`;
    }
}

function parseWeeksString(weeksStr) {
    if (!weeksStr) return [];
    
    const decrypted = decrypt(weeksStr);
    const str = decrypted || weeksStr;
    
    const weeks = new Set();
    
    const rangeMatch = str.match(/(\d+)-(\d+)/);
    if (rangeMatch) {
        const start = parseInt(rangeMatch[1]);
        const end = parseInt(rangeMatch[2]);
        for (let i = start; i <= end; i++) {
            weeks.add(i);
        }
    }
    
    const singleWeeks = str.match(/\d+/g);
    if (singleWeeks) {
        singleWeeks.forEach(w => weeks.add(parseInt(w)));
    }
    
    return Array.from(weeks).sort((a, b) => a - b);
}

function isCourseInCurrentWeek(course, currentWeek) {
    if (course.weeks) {
        const weekList = parseWeeksString(course.weeks);
        return weekList.includes(currentWeek);
    }
    
    if (course.week) {
        return course.week === currentWeek;
    }
    
    return true;
}

async function getSemesterStartDate(studentId) {
    try {
        const config = await CourseReminderConfig.findOne({ where: { studentId } });
        if (config && config.semesterStartDate) {
            console.log(`[课程提醒] 用户 ${studentId} 使用配置表中的开学时间: ${config.semesterStartDate}`);
            return config.semesterStartDate;
        }
        
        try {
            const student = await Student.findByPk(studentId);
            if (student && student.semesterStartDate) {
                console.log(`[课程提醒] 用户 ${studentId} 使用Student表中的开学时间: ${student.semesterStartDate}`);
                return student.semesterStartDate;
            }
        } catch (e) {
            console.log(`[课程提醒] Student表查询失败，跳过: ${e.message}`);
        }
        
        console.log(`[课程提醒] 用户 ${studentId} 使用默认开学时间: ${DEFAULT_SEMESTER_START}`);
        return DEFAULT_SEMESTER_START;
    } catch (e) {
        console.error('[课程提醒] 获取学期开始日期失败:', e.message);
        return DEFAULT_SEMESTER_START;
    }
}

function parseTimeToMinutes(timeStr) {
    if (!timeStr || typeof timeStr !== 'string') return 0;
    const parts = timeStr.split(':');
    if (parts.length < 2) return 0;
    return parseInt(parts[0]) * 60 + parseInt(parts[1]);
}

function getCourseStartMinutes(period) {
    const scheduleMap = {
        '1-2': 510,
        '3-4': 625,
        '5-6': 870,
        '7-8': 975,
        '9-10': 1100,
        '11-12': 1205
    };
    
    if (scheduleMap[period]) {
        console.log(`[时间映射] period=${period} -> ${Math.floor(scheduleMap[period]/60)}:${String(scheduleMap[period]%60).padStart(2,'0')} (${scheduleMap[period]}分钟)`);
        return scheduleMap[period];
    }
    
    if (period && period.includes('-')) {
        const first = parseInt(period.split('-')[0]);
        const timeSlots = {
            1: 510,
            2: 560,
            3: 625,
            4: 675,
            5: 870,
            6: 920,
            7: 975,
            8: 1025,
            9: 1100,
            10: 1150,
            11: 1205,
            12: 1255
        };
        
        const minutes = timeSlots[first] || 510;
        console.log(`[时间映射] period=${period} (slot=${first}) -> ${Math.floor(minutes/60)}:${String(minutes%60).padStart(2,'0')} (${minutes}分钟)`);
        return minutes;
    }
    
    console.log(`[时间映射] period=${period} 无法解析，使用默认值 08:30`);
    return 510;
}

async function sendBeforeClassReminders() {
    console.log('[课程提醒] ========== 课前提醒扫描开始 ==========');
    console.log('[课程提醒] enabled:', REMINDER_CONFIG.enabled);
    
    if (!REMINDER_CONFIG.enabled) {
        console.log('[课程提醒] 服务已禁用，跳过');
        return;
    }

    cleanupExpiredCache(beforeClassSentCache);

    const { now, dayOfWeek } = getCurrentWeekInfo();
    const currentMinutes = now.getHours() * 60 + now.getMinutes();
    const dayOfWeekChinese = DAY_NAMES[dayOfWeek];
    const todayStr = getTodayDateString();

    console.log(`[课程提醒] 当前时间: ${now.toLocaleString()}`);
    console.log(`[课程提醒] dayOfWeek=${dayOfWeek} (${dayOfWeekChinese}), currentMinutes=${currentMinutes} (${now.getHours()}:${now.getMinutes()})`);
    console.log(`[课程提醒] 今日日期: ${todayStr}`);
    console.log(`[课程提醒] 去重缓存大小: ${beforeClassSentCache.size}`);

    try {
        const activeUsers = await UserPushToken.findAll({
            where: { isActive: true }
        });

        console.log(`[课程提醒] 活跃推送用户数: ${activeUsers ? activeUsers.length : 0}`);

        if (!activeUsers || activeUsers.length === 0) {
            console.log('[课程提醒] 无活跃推送用户，跳过');
            return;
        }

        let sentCount = 0;
        let checkedCount = 0;
        let skippedCount = 0;

        for (const user of activeUsers) {
            try {
                console.log(`[课程提醒] 检查用户 ${user.studentId} 的课程...`);
                
                const userConfig = await getOrCreateUserConfig(user.studentId);
                console.log(`[课程提醒] 用户 ${user.studentId} 配置信息:`);
                console.log(`[课程提醒]   - 启用状态: ${userConfig.enabled}`);
                console.log(`[课程提醒]   - 开学时间: ${userConfig.semesterStartDate || '未设置'}`);
                console.log(`[课程提醒]   - 当前周次: ${userConfig.currentWeek}`);
                console.log(`[课程提醒]   - 课前提醒: ${userConfig.remindBeforeClass ? '开启' : '关闭'}, 提前 ${userConfig.beforeClassMinutes} 分钟`);
                console.log(`[课程提醒]   - 明日提醒: ${userConfig.remindTomorrowCourse ? '开启' : '关闭'}, 时间 ${userConfig.tomorrowHour}:${String(userConfig.tomorrowMinute).padStart(2, '0')}`);
                
                if (!userConfig.enabled) {
                    console.log(`[课程提醒] 用户 ${user.studentId} 已禁用提醒，跳过`);
                    continue;
                }
                
                const semesterStart = await getSemesterStartDate(user.studentId);
                const currentWeek = calculateCurrentWeek(semesterStart);
                const currentSemester = getCurrentSemester(semesterStart);
                console.log(`[课程提醒] 用户 ${user.studentId} 学期开始: ${semesterStart}, 当前周: ${currentWeek}, 计算学期: ${currentSemester}`);
                
                const allCourses = await Course.findAll({
                    where: {
                        studentId: user.studentId,
                        dayOfWeek: dayOfWeekChinese
                    },
                    attributes: ['semester', 'dayOfWeek'],
                    group: ['semester']
                });
                console.log(`[课程提醒] 用户 ${user.studentId} 数据库中所有学期: ${allCourses.map(c => c.semester).join(', ')}`);
                
                const courses = await Course.findAll({
                    where: {
                        studentId: user.studentId,
                        dayOfWeek: dayOfWeekChinese,
                        semester: currentSemester
                    }
                });

                console.log(`[课程提醒] 用户 ${user.studentId} 数据库中${dayOfWeekChinese}课程总数: ${courses.length} (学期: ${currentSemester})`);

                const seenCourses = new Set();
                const validCourses = [];
                
                for (const course of courses) {
                    checkedCount++;
                    
                    if (!isCourseInCurrentWeek(course, currentWeek)) {
                        continue;
                    }
                    
                    const decryptedCourse = decryptCourse(course.get({ plain: true }));
                    const courseName = decryptedCourse.name || course.name;
                    const location = decryptedCourse.location || course.location;
                    
                    const courseKey = `${courseName}_${course.period}`;
                    if (seenCourses.has(courseKey)) {
                        continue;
                    }
                    seenCourses.add(courseKey);
                    
                    const courseStartMinutes = getCourseStartMinutes(course.period);
                    const minutesUntilClass = courseStartMinutes - currentMinutes;
                    
                    validCourses.push({
                        courseName,
                        location,
                        period: course.period,
                        startMinutes: courseStartMinutes,
                        minutesUntil: minutesUntilClass
                    });
                }

                console.log(`[课程提醒] 用户 ${user.studentId} 当前第${currentWeek}周有效课程: ${validCourses.length}节`);
                
                for (const course of validCourses) {
                    console.log(`[课程提醒] 课程: ${course.courseName}, period: ${course.period}, startMinutes: ${course.startMinutes}, minutesUntil: ${course.minutesUntil}`);

                    if (userConfig.remindBeforeClass && course.minutesUntil > 0 && course.minutesUntil <= userConfig.beforeClassMinutes) {
                        if (isBeforeClassAlreadySent(user.studentId, course.courseName, course.period, todayStr)) {
                            console.log(`[课程提醒] ⏭️ 跳过重复推送: ${course.courseName} (${course.period}) 今日已发送`);
                            skippedCount++;
                            continue;
                        }
                        
                        console.log(`[课程提醒] >>> 触发课前提醒: ${course.courseName}, ${course.minutesUntil}分钟后开始 (阈值: ${userConfig.beforeClassMinutes}分钟)`);
                        
                        const result = await pushService.notifyCourseReminder(user.studentId, {
                            type: 'before_class',
                            courseName: course.courseName,
                            location: course.location,
                            startTime: course.period,
                            message: `${course.courseName} 将在${course.minutesUntil}分钟后开始\n地点: ${course.location || '待定'}\n时间: ${course.period || '-'}`
                        }, user.pushToken);
                        
                        console.log(`[课程提醒] 推送结果:`, result);
                        
                        markBeforeClassAsSent(user.studentId, course.courseName, course.period, todayStr);
                        sentCount++;
                    } else if (course.minutesUntil > 0 && course.minutesUntil <= userConfig.beforeClassMinutes) {
                        console.log(`[课程提醒] 课程 ${course.courseName} 在提醒范围内，但用户未开启课前提醒`);
                    }
                }
            } catch (e) {
                console.error(`[课程提醒] 用户 ${user.studentId} 课前提醒失败:`, e.message);
            }
        }

        console.log(`[课程提醒] 课前提醒扫描完成 检查:${checkedCount} 发送:${sentCount} 跳过:${skippedCount}`);
    } catch (error) {
        console.error('[课程提醒] 课前提醒异常:', error.message, error.stack);
    }
}

async function sendTomorrowCourseReminders() {
    console.log('[课程提醒] ========== 明日课程提醒扫描开始 ==========');
    console.log('[课程提醒] enabled:', REMINDER_CONFIG.enabled);
    
    if (!REMINDER_CONFIG.enabled) {
        console.log('[课程提醒] 服务已禁用，跳过');
        return;
    }

    cleanupExpiredCache(tomorrowSentCache);

    const { now, dayOfWeek } = getCurrentWeekInfo();
    const tomorrowDayOfWeek = dayOfWeek === 7 ? 1 : dayOfWeek + 1;
    const tomorrowDayName = DAY_NAMES_SHORT[tomorrowDayOfWeek];
    const tomorrowDayNameChinese = DAY_NAMES[tomorrowDayOfWeek];
    const todayStr = getTodayDateString();
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    const currentTotalMinutes = currentHour * 60 + currentMinute;

    console.log(`[课程提醒] 当前时间: ${now.toLocaleString()}`);
    console.log(`[课程提醒] 当前dayOfWeek=${dayOfWeek} (${DAY_NAMES[dayOfWeek]}), 明日dayOfWeek=${tomorrowDayOfWeek} (${tomorrowDayNameChinese})`);
    console.log(`[课程提醒] 今日日期: ${todayStr}`);
    console.log(`[课程提醒] 当前分钟数: ${currentTotalMinutes} (${currentHour}:${String(currentMinute).padStart(2, '0')})`);
    console.log(`[课程提醒] 明日去重缓存大小: ${tomorrowSentCache.size}`);

    try {
        const activeUsers = await UserPushToken.findAll({
            where: { isActive: true }
        });

        console.log(`[课程提醒] 活跃推送用户数: ${activeUsers ? activeUsers.length : 0}`);

        if (!activeUsers || activeUsers.length === 0) {
            console.log('[课程提醒] 无活跃推送用户，跳过');
            return;
        }

        let sentCount = 0;
        let skippedCount = 0;
        let timeNotMatchCount = 0;

        for (const user of activeUsers) {
            try {
                console.log(`[课程提醒] 检查用户 ${user.studentId} 明日课程...`);
                
                const userConfig = await getOrCreateUserConfig(user.studentId);
                console.log(`[课程提醒] 用户 ${user.studentId} 配置信息:`);
                console.log(`[课程提醒]   - 启用状态: ${userConfig.enabled}`);
                console.log(`[课程提醒]   - 开学时间: ${userConfig.semesterStartDate || '未设置'}`);
                console.log(`[课程提醒]   - 当前周次: ${userConfig.currentWeek}`);
                console.log(`[课程提醒]   - 明日提醒: ${userConfig.remindTomorrowCourse ? '开启' : '关闭'}, 时间 ${userConfig.tomorrowHour}:${String(userConfig.tomorrowMinute).padStart(2, '0')}`);
                
                if (!userConfig.enabled) {
                    console.log(`[课程提醒] 用户 ${user.studentId} 已禁用提醒，跳过`);
                    continue;
                }
                
                if (!userConfig.remindTomorrowCourse) {
                    console.log(`[课程提醒] 用户 ${user.studentId} 未开启明日提醒，跳过`);
                    continue;
                }
                
                const userConfigTime = userConfig.tomorrowHour * 60 + userConfig.tomorrowMinute;
                if (currentTotalMinutes !== userConfigTime) {
                    console.log(`[课程提醒] ⏰ 用户 ${user.studentId} 设定时间 ${userConfig.tomorrowHour}:${String(userConfig.tomorrowMinute).padStart(2, '0')} (${userConfigTime}分钟) ≠ 当前时间 ${currentHour}:${String(currentMinute).padStart(2, '0')} (${currentTotalMinutes}分钟)，跳过`);
                    timeNotMatchCount++;
                    continue;
                }
                
                console.log(`[课程提醒] ✅ 用户 ${user.studentId} 设定时间 ${userConfig.tomorrowHour}:${String(userConfig.tomorrowMinute).padStart(2, '0')} 匹配当前时间，开始处理`);
                
                if (isTomorrowAlreadySent(user.studentId, tomorrowDayOfWeek, todayStr)) {
                    console.log(`[课程提醒] ⏭️ 用户 ${user.studentId} 今日已发送过明日(${tomorrowDayName})课程提醒，跳过`);
                    skippedCount++;
                    continue;
                }
                
                const semesterStart = await getSemesterStartDate(user.studentId);
                const currentWeek = calculateCurrentWeek(semesterStart);
                const currentSemester = getCurrentSemester(semesterStart);
                console.log(`[课程提醒] 用户 ${user.studentId} 学期开始: ${semesterStart}, 当前周: ${currentWeek}, 学期: ${currentSemester}`);
                
                const courses = await Course.findAll({
                    where: {
                        studentId: user.studentId,
                        dayOfWeek: tomorrowDayNameChinese,
                        semester: currentSemester
                    }
                });

                console.log(`[课程提醒] 用户 ${user.studentId} 数据库中${tomorrowDayNameChinese}课程总数: ${courses.length} (学期: ${currentSemester})`);

                if (courses.length === 0) continue;

                const seenCourses = new Set();
                const uniqueCourses = [];
                
                for (const c of courses) {
                    if (!isCourseInCurrentWeek(c, currentWeek)) {
                        continue;
                    }
                    
                    const decrypted = decryptCourse(c.get({ plain: true }));
                    const name = decrypted.name || c.name;
                    const key = `${name}_${c.period}`;
                    if (!seenCourses.has(key)) {
                        seenCourses.add(key);
                        uniqueCourses.push({ ...c, decryptedName: name, decryptedLocation: decrypted.location || c.location });
                    }
                }

                console.log(`[课程提醒] 用户 ${user.studentId} 当前第${currentWeek}周有效课程: ${uniqueCourses.length}节`);

                if (uniqueCourses.length === 0) continue;

                console.log(`[课程提醒] 明日课程列表:`);
                uniqueCourses.forEach((c, i) => {
                    console.log(`[课程提醒]   ${i + 1}. ${c.decryptedName} | ${c.period || '-'} | ${c.decryptedLocation || '待定'}`);
                });

                const courseList = uniqueCourses.map(c =>
                    `${c.decryptedName} (${c.period || '-'}) @ ${c.decryptedLocation || '待定'}`
                ).join('\n');

                console.log(`[课程提醒] >>> 触发明日提醒: ${user.studentId}, ${uniqueCourses.length}门课程`);
                
                const result = await pushService.notifyCourseReminder(user.studentId, {
                    type: 'tomorrow',
                    message: `明天${tomorrowDayName}有 ${uniqueCourses.length} 门课程:\n${courseList}`
                }, user.pushToken);
                
                console.log(`[课程提醒] 推送结果:`, result);
                
                markTomorrowAsSent(user.studentId, tomorrowDayOfWeek, todayStr);
                sentCount++;
            } catch (e) {
                console.error(`[课程提醒] 用户 ${user.studentId} 明日提醒失败:`, e.message);
            }
        }

        console.log(`[课程提醒] 明日课程提醒扫描完成 发送:${sentCount} 跳过:${skippedCount} 时间不匹配:${timeNotMatchCount}`);
    } catch (error) {
        console.error('[课程提醒] 明日课程提醒异常:', error.message, error.stack);
    }
}

function start() {
    console.log('[课程提醒] ========================================');
    console.log('[课程提醒] 课程提醒推送服务启动中...');
    console.log('[课程提醒] ========================================');
    
    if (!REMINDER_CONFIG.enabled) {
        console.log('[课程提醒] 课程提醒推送服务已禁用 (COURSE_REMINDER_ENABLED=false)');
        return;
    }

    console.log(`[课程提醒] 配置信息:`);
    console.log(`[课程提醒]   - 课前提醒: 每分钟检查，提前${REMINDER_CONFIG.beforeClassMinutes}分钟`);
    console.log(`[课程提醒]   - 明日提醒: 每分钟检查，按用户设定时间推送`);
    console.log(`[课程提醒]   - enabled: ${REMINDER_CONFIG.enabled}`);

    try {
        beforeClassCron = nodeCron.schedule('* * * * * 1-5', async () => {
            await sendBeforeClassReminders();
        });
        console.log('[课程提醒] beforeClassCron 创建成功');
    } catch (e) {
        console.error('[课程提醒] beforeClassCron 创建失败:', e.message);
    }

    try {
        tomorrowCron = nodeCron.schedule('* * * * *', async () => {
            await sendTomorrowCourseReminders();
        });
        console.log('[课程提醒] tomorrowCron 创建成功 (每分钟检查，按用户配置时间推送)');
    } catch (e) {
        console.error('[课程提醒] tomorrowCron 创建失败:', e.message);
    }

    console.log('[课程提醒] ========================================');
    console.log('[课程提醒] 定时任务已启动');
    console.log('[课程提醒] beforeClassCron:', beforeClassCron ? '运行中' : '未启动');
    console.log('[课程提醒] tomorrowCron:', tomorrowCron ? '运行中' : '未启动');
    console.log('[课程提醒] ========================================');
}

function stop() {
    if (beforeClassCron) {
        beforeClassCron.stop();
        beforeClassCron = null;
    }
    if (tomorrowCron) {
        tomorrowCron.stop();
        tomorrowCron = null;
    }
    console.log('[课程提醒] 服务已停止');
}

function getStatus() {
    return {
        enabled: REMINDER_CONFIG.enabled,
        beforeClassMinutes: REMINDER_CONFIG.beforeClassMinutes,
        tomorrowTime: `${REMINDER_CONFIG.tomorrowHour}:${String(REMINDER_CONFIG.tomorrowMinute).padStart(2, '0')}`,
        running: !!beforeClassCron
    };
}

function updateConfig(config) {
    console.log('[课程提醒] 收到配置更新请求:', JSON.stringify(config));
    
    const oldBeforeClassMinutes = REMINDER_CONFIG.beforeClassMinutes;
    const oldTomorrowHour = REMINDER_CONFIG.tomorrowHour;
    const oldTomorrowMinute = REMINDER_CONFIG.tomorrowMinute;
    
    if (config.beforeClassMinutes !== undefined) {
        console.log(`[课程提醒] 课前提醒时间: ${oldBeforeClassMinutes}分钟 -> ${config.beforeClassMinutes}分钟`);
    }
    if (config.tomorrowHour !== undefined) {
        console.log(`[课程提醒] 明日提醒小时: ${oldTomorrowHour} -> ${config.tomorrowHour}`);
    }
    if (config.tomorrowMinute !== undefined) {
        console.log(`[课程提醒] 明日提醒分钟: ${oldTomorrowMinute} -> ${config.tomorrowMinute}`);
    }
    if (config.enabled !== undefined) {
        REMINDER_CONFIG.enabled = config.enabled;
        console.log(`[课程提醒] 启用状态: ${config.enabled}`);
    }
    
    console.log('[课程提醒] 当前配置:', JSON.stringify({
        enabled: REMINDER_CONFIG.enabled,
        beforeClassMinutes: REMINDER_CONFIG.beforeClassMinutes,
        tomorrowTime: `${REMINDER_CONFIG.tomorrowHour}:${String(REMINDER_CONFIG.tomorrowMinute).padStart(2, '0')}`
    }));
}

module.exports = {
    start,
    stop,
    getStatus,
    updateConfig,
    sendBeforeClassReminders,
    sendTomorrowCourseReminders,
    getOrCreateUserConfig,
    updateUserConfig
};
