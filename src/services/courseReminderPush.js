const pushService = require('./pushService');
const { UserPushToken, Course } = require('../db/models');
const nodeCron = require('node-cron');

const DAY_NAMES = ['', '周一', '周二', '周三', '周四', '周五', '周六', '周日'];

let beforeClassCron = null;
let tomorrowCron = null;

const REMINDER_CONFIG = {
    beforeClassMinutes: 15,
    tomorrowHour: 21,
    tomorrowMinute: 0,
    enabled: process.env.COURSE_REMINDER_ENABLED !== 'false'
};

function getCurrentWeekInfo() {
    const now = new Date();
    const dayOfWeek = now.getDay() === 0 ? 7 : now.getDay();
    return { now, dayOfWeek };
}

function parseTimeToMinutes(timeStr) {
    if (!timeStr || typeof timeStr !== 'string') return 0;
    const parts = timeStr.split(':');
    if (parts.length < 2) return 0;
    return parseInt(parts[0]) * 60 + parseInt(parts[1]);
}

function getCourseStartMinutes(period) {
    const scheduleMap = {
        '1-2': 480,
        '3-4': 550,
        '5-6': 660,
        '7-8': 730,
        '9-10': 840,
        '11-12': 910
    };
    if (scheduleMap[period]) return scheduleMap[period];
    if (period && period.includes('-')) {
        const first = parseInt(period.split('-')[0]);
        return 480 + (first - 1) * 70 + (first > 2 ? 20 : 0) + (first > 4 ? 70 : 0) + (first > 6 ? 20 : 0) + (first > 8 ? 70 : 0) + (first > 10 ? 20 : 0);
    }
    return 480;
}

async function sendBeforeClassReminders() {
    if (!REMINDER_CONFIG.enabled) return;

    const { now, dayOfWeek } = getCurrentWeekInfo();
    const currentMinutes = now.getHours() * 60 + now.getMinutes();

    console.log(`[课程提醒] 课前提醒扫描 dayOfWeek=${dayOfWeek} currentMinutes=${currentMinutes}`);

    try {
        const activeUsers = await UserPushToken.findAll({
            where: { isActive: true }
        });

        if (!activeUsers || activeUsers.length === 0) {
            console.log('[课程提醒] 无活跃推送用户');
            return;
        }

        let sentCount = 0;

        for (const user of activeUsers) {
            try {
                const courses = await Course.findAll({
                    where: {
                        studentId: user.studentId,
                        dayOfWeek: String(dayOfWeek)
                    }
                });

                for (const course of courses) {
                    const courseStartMinutes = getCourseStartMinutes(course.period);
                    const minutesUntilClass = courseStartMinutes - currentMinutes;

                    if (minutesUntilClass > 0 && minutesUntilClass <= REMINDER_CONFIG.beforeClassMinutes) {
                        await pushService.notifyCourseReminder(user.studentId, {
                            type: 'before_class',
                            courseName: course.name,
                            location: course.location,
                            startTime: course.period,
                            message: `${course.name} 将在${minutesUntilClass}分钟后开始\n地点: ${course.location || '待定'}\n时间: ${course.period || '-'}`
                        });
                        sentCount++;
                    }
                }
            } catch (e) {
                console.error(`[课程提醒] 用户 ${user.studentId} 课前提醒失败:`, e.message);
            }
        }

        console.log(`[课程提醒] 课前提醒完成 发送:${sentCount}`);
    } catch (error) {
        console.error('[课程提醒] 课前提醒异常:', error.message);
    }
}

async function sendTomorrowCourseReminders() {
    if (!REMINDER_CONFIG.enabled) return;

    const { now, dayOfWeek } = getCurrentWeekInfo();
    const tomorrowDayOfWeek = dayOfWeek === 7 ? 1 : dayOfWeek + 1;
    const tomorrowDayName = DAY_NAMES[tomorrowDayOfWeek];

    console.log(`[课程提醒] 明日课程提醒 tomorrow=${tomorrowDayName}`);

    try {
        const activeUsers = await UserPushToken.findAll({
            where: { isActive: true }
        });

        if (!activeUsers || activeUsers.length === 0) {
            console.log('[课程提醒] 无活跃推送用户');
            return;
        }

        let sentCount = 0;

        for (const user of activeUsers) {
            try {
                const courses = await Course.findAll({
                    where: {
                        studentId: user.studentId,
                        dayOfWeek: String(tomorrowDayOfWeek)
                    }
                });

                if (courses.length === 0) continue;

                const courseList = courses.map(c =>
                    `${c.name} (${c.period || '-'}) @ ${c.location || '待定'}`
                ).join('\n');

                await pushService.notifyCourseReminder(user.studentId, {
                    type: 'tomorrow',
                    message: `明天${tomorrowDayName}有 ${courses.length} 门课程:\n${courseList}`
                });
                sentCount++;
            } catch (e) {
                console.error(`[课程提醒] 用户 ${user.studentId} 明日提醒失败:`, e.message);
            }
        }

        console.log(`[课程提醒] 明日课程提醒完成 发送:${sentCount}`);
    } catch (error) {
        console.error('[课程提醒] 明日课程提醒异常:', error.message);
    }
}

function start() {
    if (!REMINDER_CONFIG.enabled) {
        console.log('[课程提醒] 课程提醒推送服务已禁用');
        return;
    }

    console.log(`[课程提醒] 启动课程提醒推送服务`);
    console.log(`[课程提醒] 课前提醒: 每分钟检查，提前${REMINDER_CONFIG.beforeClassMinutes}分钟`);
    console.log(`[课程提醒] 明日提醒: 每天${REMINDER_CONFIG.tomorrowHour}:${String(REMINDER_CONFIG.tomorrowMinute).padStart(2, '0')}`);

    beforeClassCron = nodeCron.schedule('* * * * 1-5', async () => {
        await sendBeforeClassReminders();
    });

    const tomorrowCronExpr = `${REMINDER_CONFIG.tomorrowMinute} ${REMINDER_CONFIG.tomorrowHour} * * *`;
    tomorrowCron = nodeCron.schedule(tomorrowCronExpr, async () => {
        await sendTomorrowCourseReminders();
    });

    console.log('[课程提醒] 定时任务已启动');
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
    if (config.beforeClassMinutes !== undefined) {
        REMINDER_CONFIG.beforeClassMinutes = config.beforeClassMinutes;
    }
    if (config.tomorrowHour !== undefined) {
        REMINDER_CONFIG.tomorrowHour = config.tomorrowHour;
    }
    if (config.tomorrowMinute !== undefined) {
        REMINDER_CONFIG.tomorrowMinute = config.tomorrowMinute;
    }
    if (config.enabled !== undefined) {
        REMINDER_CONFIG.enabled = config.enabled;
    }
    console.log('[课程提醒] 配置已更新:', REMINDER_CONFIG);
}

module.exports = {
    start,
    stop,
    getStatus,
    updateConfig,
    sendBeforeClassReminders,
    sendTomorrowCourseReminders
};
