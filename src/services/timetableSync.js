const { getTimetable } = require('../api/student');
const { login } = require('../api/auth');
const { Course, Student, UserPushToken, CourseReminderConfig } = require('../db/models');
const { decrypt, encrypt } = require('../utils/encryption');
const pushService = require('./pushService');
const fs = require('fs');
const path = require('path');

const SYNC_INTERVAL = 30 * 60 * 1000;
const MAX_RETRY = 3;
const CREDENTIALS_FILE = path.join(__dirname, '../../data/credentials.json');

const userCredentials = new Map();
const syncStatus = new Map();

function ensureDataDir() {
    const dataDir = path.dirname(CREDENTIALS_FILE);
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
    }
}

function loadCredentialsFromFile() {
    try {
        ensureDataDir();
        if (fs.existsSync(CREDENTIALS_FILE)) {
            const data = fs.readFileSync(CREDENTIALS_FILE, 'utf8');
            const credentials = JSON.parse(data);
            
            for (const [studentId, cred] of Object.entries(credentials)) {
                userCredentials.set(studentId, cred);
            }
            
            console.log(`[课表同步] 从文件加载了 ${userCredentials.size} 个用户的凭据`);
        }
    } catch (error) {
        console.error('[课表同步] 加载凭据文件失败:', error.message);
    }
}

function saveCredentialsToFile() {
    try {
        ensureDataDir();
        const credentials = {};
        
        for (const [studentId, cred] of userCredentials.entries()) {
            credentials[studentId] = cred;
        }
        
        fs.writeFileSync(CREDENTIALS_FILE, JSON.stringify(credentials, null, 2), 'utf8');
        console.log(`[课表同步] 保存了 ${userCredentials.size} 个用户的凭据到文件`);
    } catch (error) {
        console.error('[课表同步] 保存凭据文件失败:', error.message);
    }
}

function startTimetableSync() {
    console.log('[课表同步] 启动课表自动同步服务...');
    console.log(`[课表同步] 同步间隔: ${SYNC_INTERVAL / 60000} 分钟`);
    
    loadCredentialsFromFile();
    
    setTimeout(async () => {
        await cleanupDuplicateCourses();
    }, 5000);
    
    setInterval(async () => {
        await syncAllUsers();
    }, SYNC_INTERVAL);
    
    setInterval(async () => {
        await cleanupDuplicateCourses();
    }, 6 * 60 * 60 * 1000);
    
    setTimeout(async () => {
        await syncAllUsers();
    }, 10000);
}

async function cleanupDuplicateCourses() {
    try {
        console.log('[清理重复] ========== 开始清理重复课程记录 ==========');
        
        const allCourses = await Course.findAll();
        
        const courseMap = new Map();
        allCourses.forEach(course => {
            const decryptedName = decrypt(course.name) || course.name;
            const key = `${course.studentId}_${decryptedName}_${course.dayOfWeek}_${course.period}`;
            
            if (!courseMap.has(key)) {
                courseMap.set(key, []);
            }
            courseMap.get(key).push(course);
        });
        
        let deletedCount = 0;
        let updatedCount = 0;
        
        for (const [key, courses] of courseMap) {
            if (courses.length > 1) {
                console.log(`[清理重复] 发现重复记录: ${key}, 数量: ${courses.length}`);
                
                const allWeeks = new Set();
                courses.forEach(c => {
                    const decryptedWeeks = decrypt(c.weeks) || '';
                    if (decryptedWeeks) {
                        decryptedWeeks.split(',').forEach(w => {
                            const week = parseInt(w.trim());
                            if (!isNaN(week)) allWeeks.add(week);
                        });
                    }
                    if (c.week) allWeeks.add(c.week);
                });
                
                const mergedWeeks = Array.from(allWeeks).sort((a, b) => a - b).join(',');
                const primary = courses[0];
                
                await primary.update({
                    weeks: mergedWeeks ? encrypt(mergedWeeks) : null,
                    week: Array.from(allWeeks)[0] || primary.week
                });
                updatedCount++;
                
                for (let i = 1; i < courses.length; i++) {
                    await courses[i].destroy();
                    deletedCount++;
                }
            }
        }
        
        console.log(`[清理重复] ========== 清理完成 ========== 更新:${updatedCount}, 删除:${deletedCount}`);
    } catch (error) {
        console.error('[清理重复] 清理失败:', error.message);
    }
}

async function syncAllUsers() {
    try {
        console.log('[课表同步] ========== 开始同步所有用户课表 ==========');
        
        const activeUsers = await UserPushToken.findAll({
            where: { isActive: true }
        });
        
        if (!activeUsers || activeUsers.length === 0) {
            console.log('[课表同步] 无活跃用户，跳过同步');
            return;
        }
        
        console.log(`[课表同步] 发现 ${activeUsers.length} 个活跃用户`);
        
        let successCount = 0;
        let failCount = 0;
        let skipCount = 0;
        
        for (const user of activeUsers) {
            try {
                const result = await syncUserTimetable(user.studentId);
                
                if (result.success) {
                    successCount++;
                } else if (result.skipped) {
                    skipCount++;
                } else {
                    failCount++;
                }
            } catch (error) {
                console.error(`[课表同步] 用户 ${user.studentId} 同步异常:`, error.message);
                failCount++;
            }
        }
        
        console.log(`[课表同步] ========== 同步完成 ==========`);
        console.log(`[课表同步] 成功: ${successCount}, 失败: ${failCount}, 跳过: ${skipCount}`);
    } catch (error) {
        console.error('[课表同步] 同步服务异常:', error.message);
    }
}

async function syncUserTimetable(studentId) {
    console.log(`[课表同步] 开始同步用户 ${studentId} 的课表...`);
    
    const credentials = userCredentials.get(studentId);
    
    if (!credentials) {
        console.log(`[课表同步] 用户 ${studentId} 未保存登录凭据，跳过同步`);
        return { success: false, skipped: true, message: '未保存登录凭据' };
    }
    
    const lastSync = syncStatus.get(studentId);
    const now = Date.now();
    
    if (lastSync && now - lastSync < 10 * 60 * 1000) {
        console.log(`[课表同步] 用户 ${studentId} 最近已同步，跳过`);
        return { success: false, skipped: true, message: '最近已同步' };
    }
    
    try {
        let cookies = credentials.cookies;
        const decryptedPassword = decrypt(credentials.password);
        
        console.log(`[课表同步] 尝试获取课表...`);
        let timetable = await getTimetable(cookies);
        
        if (!timetable || !timetable.data || timetable.data.length === 0) {
            console.log(`[课表同步] Cookies可能已失效，尝试重新登录...`);
            
            const loginResult = await login(credentials.username, decryptedPassword);
            
            if (!loginResult.success) {
                console.error(`[课表同步] 重新登录失败: ${loginResult.message}`);
                return { success: false, message: '登录失败' };
            }
            
            cookies = loginResult.cookies;
            userCredentials.set(studentId, {
                ...credentials,
                cookies: cookies
            });
            
            timetable = await getTimetable(cookies);
            
            if (!timetable || !timetable.data) {
                console.error(`[课表同步] 重新登录后仍无法获取课表`);
                return { success: false, message: '无法获取课表' };
            }
        }
        
        console.log(`[课表同步] 获取到 ${timetable.data.length} 条课程数据`);
        
        const changes = await updateDatabaseCourses(studentId, timetable.data);
        
        if (changes.length > 0) {
            console.log(`[课表同步] 发现 ${changes.length} 处课表变更`);
            
            for (const change of changes) {
                await pushService.notifyCourseChange(studentId, {
                    type: change.type,
                    courseName: change.courseName,
                    message: change.message
                });
            }
        }
        
        syncStatus.set(studentId, now);
        
        console.log(`[课表同步] ✅ 用户 ${studentId} 课表同步成功`);
        return { success: true, changes: changes.length };
        
    } catch (error) {
        console.error(`[课表同步] 同步失败:`, error.message);
        return { success: false, message: error.message };
    }
}

async function updateDatabaseCourses(studentId, newCourses) {
    const changes = [];
    
    try {
        console.log(`[课表同步] 开始更新数据库课程，新课程数: ${newCourses.length}`);
        
        const existingCourses = await Course.findAll({
            where: { studentId }
        });
        
        console.log(`[课表同步] 数据库中现有课程数: ${existingCourses.length}`);
        
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
        newCourses.forEach(course => {
            const key = `${course.courseName}_${course.dayOfWeek}_${course.period || course.startTime}`;
            
            if (!newMap.has(key)) {
                newMap.set(key, []);
            }
            newMap.get(key).push(course);
        });
        
        for (const [key, newCourseList] of newMap) {
            const existingCourseList = existingMap.get(key) || [];
            
            if (existingCourseList.length === 0) {
                for (const newCourse of newCourseList) {
                    await Course.create({
                        studentId,
                        semester: newCourse.semester,
                        name: encrypt(newCourse.courseName),
                        dayOfWeek: newCourse.dayOfWeek,
                        week: newCourse.week,
                        period: newCourse.period || newCourse.startTime,
                        teacher: encrypt(newCourse.teacher || ''),
                        weeks: newCourse.weeks ? encrypt(newCourse.weeks) : null,
                        location: encrypt(newCourse.location || ''),
                        courseType: newCourse.courseType,
                        raw: JSON.stringify(newCourse)
                    });
                    
                    changes.push({
                        type: 'new',
                        courseName: newCourse.courseName,
                        message: `新增课程：${newCourse.courseName}`
                    });
                }
            } else {
                const newWeeksSet = new Set();
                const newLocation = newCourseList[0].location || '';
                const newTeacher = newCourseList[0].teacher || '';
                const newCourseType = newCourseList[0].courseType;
                const newSemester = newCourseList[0].semester;
                
                newCourseList.forEach(c => {
                    if (c.weeks) {
                        const weeksList = c.weeks.split(',').map(w => parseInt(w.trim())).filter(w => !isNaN(w));
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
                
                const hasChanges = newLocation !== existingLocation || newWeeks !== existingWeeks;
                
                if (hasChanges) {
                    await existing.update({
                        location: encrypt(newLocation),
                        weeks: newWeeks ? encrypt(newWeeks) : null,
                        week: newCourseList[0].week,
                        teacher: encrypt(newTeacher),
                        courseType: newCourseType,
                        semester: newSemester,
                        raw: JSON.stringify(newCourseList[0])
                    });
                    
                    if (newLocation !== existingLocation) {
                        changes.push({
                            type: 'location_change',
                            courseName: newCourseList[0].courseName,
                            message: `${newCourseList[0].courseName} 教室变更：${existingLocation} → ${newLocation}`
                        });
                    }
                    
                    if (newWeeks !== existingWeeks) {
                        changes.push({
                            type: 'weeks_change',
                            courseName: newCourseList[0].courseName,
                            message: `${newCourseList[0].courseName} 周次变更：${existingWeeks} → ${newWeeks}`
                        });
                    }
                }
                
                for (let i = 1; i < existingCourseList.length; i++) {
                    await existingCourseList[i].destroy();
                    console.log(`[课表同步] 删除重复课程记录: id=${existingCourseList[i].id}`);
                }
                
                existingMap.delete(key);
            }
        }
        
        for (const [key, oldCourseList] of existingMap) {
            for (const oldCourse of oldCourseList) {
                await oldCourse.destroy();
                
                changes.push({
                    type: 'cancelled',
                    courseName: decrypt(oldCourse.name),
                    message: `课程取消：${decrypt(oldCourse.name)}`
                });
            }
        }
        
        console.log(`[课表同步] 数据库更新完成，变更数: ${changes.length}`);
        
    } catch (error) {
        console.error('[课表同步] 更新数据库失败:', error.message);
    }
    
    return changes;
}

function saveUserCredentials(studentId, username, password, cookies) {
    userCredentials.set(studentId, {
        username,
        password: encrypt(password),
        cookies,
        savedAt: Date.now()
    });
    
    saveCredentialsToFile();
    
    console.log(`[课表同步] 已保存用户 ${studentId} 的登录凭据`);
}

function removeUserCredentials(studentId) {
    userCredentials.delete(studentId);
    syncStatus.delete(studentId);
    
    saveCredentialsToFile();
    
    console.log(`[课表同步] 已移除用户 ${studentId} 的登录凭据`);
}

function getSyncStatus() {
    return {
        totalUsers: userCredentials.size,
        lastSyncTimes: Object.fromEntries(syncStatus)
    };
}

module.exports = {
    startTimetableSync,
    syncAllUsers,
    syncUserTimetable,
    saveUserCredentials,
    removeUserCredentials,
    getSyncStatus,
    userCredentials,
    cleanupDuplicateCourses
};
