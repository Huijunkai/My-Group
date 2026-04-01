const pushService = require('./pushService');
const { UserPushToken, Grade, Exam, Course } = require('../db/models');
const { decrypt } = require('../utils/encryption');

async function getUserPushToken(studentId) {
    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });
        return userToken ? userToken.pushToken : null;
    } catch (error) {
        console.error('Failed to get user push token:', error);
        return null;
    }
}

async function notifyNewGradeRealtime(studentId, gradeInfo) {
    const token = await getUserPushToken(studentId);
    if (!token) {
        console.log(`No push token for student ${studentId}, skipping grade notification`);
        return { success: false, message: 'No push token' };
    }

    const decryptedCourseName = decrypt(gradeInfo.courseName);
    const decryptedScore = decrypt(gradeInfo.score);

    return await pushService.sendPushNotification(
        token,
        '新成绩发布',
        `${decryptedCourseName}: ${decryptedScore}分`,
        'new_grade',
        {
            courseName: decryptedCourseName,
            score: decryptedScore,
            credit: gradeInfo.credit,
            semester: gradeInfo.semester
        }
    );
}

async function notifyNewExamRealtime(studentId, examInfo) {
    const token = await getUserPushToken(studentId);
    if (!token) {
        console.log(`No push token for student ${studentId}, skipping exam notification`);
        return { success: false, message: 'No push token' };
    }

    const decryptedCourseName = decrypt(examInfo.courseName);
    const decryptedLocation = decrypt(examInfo.location);

    return await pushService.sendPushNotification(
        token,
        '新考试安排',
        `${decryptedCourseName} - ${examInfo.examTime}`,
        'new_exam',
        {
            courseName: decryptedCourseName,
            examTime: examInfo.examTime,
            location: decryptedLocation
        }
    );
}

async function notifyCourseChangeRealtime(studentId, changeInfo) {
    const token = await getUserPushToken(studentId);
    if (!token) {
        console.log(`No push token for student ${studentId}, skipping course change notification`);
        return { success: false, message: 'No push token' };
    }

    return await pushService.sendPushNotification(
        token,
        '课程变动通知',
        changeInfo.message || '您的课表有新的变动，请及时查看',
        'course_change',
        {
            changeType: changeInfo.type,
            courseName: changeInfo.courseName
        }
    );
}

async function checkAndNotifyNewGrades(studentId, newGrades) {
    const results = [];
    
    for (const grade of newGrades) {
        const result = await notifyNewGradeRealtime(studentId, grade);
        results.push({
            courseCode: grade.courseCode,
            success: result.success
        });
    }
    
    return results;
}

async function checkAndNotifyNewExams(studentId, newExams) {
    const results = [];
    
    for (const exam of newExams) {
        const result = await notifyNewExamRealtime(studentId, exam);
        results.push({
            courseName: exam.courseName,
            success: result.success
        });
    }
    
    return results;
}

async function checkAndNotifyCourseChanges(studentId, changes) {
    const results = [];
    
    for (const change of changes) {
        const result = await notifyCourseChangeRealtime(studentId, change);
        results.push({
            type: change.type,
            success: result.success
        });
    }
    
    return results;
}

async function getActivePushUsers() {
    try {
        const users = await UserPushToken.findAll({
            where: { isActive: true }
        });
        return users.map(u => ({
            studentId: u.studentId,
            pushToken: u.pushToken,
            lastActiveAt: u.lastActiveAt
        }));
    } catch (error) {
        console.error('Failed to get active push users:', error);
        return [];
    }
}

module.exports = {
    getUserPushToken,
    notifyNewGradeRealtime,
    notifyNewExamRealtime,
    notifyCourseChangeRealtime,
    checkAndNotifyNewGrades,
    checkAndNotifyNewExams,
    checkAndNotifyCourseChanges,
    getActivePushUsers
};
