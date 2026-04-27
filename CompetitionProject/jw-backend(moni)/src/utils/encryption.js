const crypto = require('crypto');

const ALGORITHM = 'aes-256-cbc';
const KEY_LENGTH = 32;
const IV_LENGTH = 16;

const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || 'NNLG-HarmonyOS-2024-Secret-Key!!';
const ENCRYPTION_IV = process.env.ENCRYPTION_IV || 'NNLG-InitVector16';

let key = null;
let iv = null;

function getKey() {
    if (!key) {
        key = crypto.scryptSync(ENCRYPTION_KEY, 'salt', KEY_LENGTH);
    }
    return key;
}

function getIv() {
    if (!iv) {
        const ivBase = ENCRYPTION_IV.padEnd(IV_LENGTH, '0').slice(0, IV_LENGTH);
        iv = Buffer.from(ivBase, 'utf8');
    }
    return iv;
}

function encrypt(text) {
    if (text === null || text === undefined) {
        return text;
    }
    
    const textStr = String(text);
    if (textStr === '') {
        return textStr;
    }
    
    try {
        const cipher = crypto.createCipheriv(ALGORITHM, getKey(), getIv());
        let encrypted = cipher.update(textStr, 'utf8', 'base64');
        encrypted += cipher.final('base64');
        return 'ENC:' + encrypted;
    } catch (error) {
        console.error('Encryption error:', error.message);
        return textStr;
    }
}

function decrypt(encryptedText) {
    if (encryptedText === null || encryptedText === undefined) {
        return encryptedText;
    }
    
    const textStr = String(encryptedText);
    if (textStr === '' || !textStr.startsWith('ENC:')) {
        return textStr;
    }
    
    const encryptedData = textStr.slice(4);
    
    try {
        const decipher = crypto.createDecipheriv(ALGORITHM, getKey(), getIv());
        let decrypted = decipher.update(encryptedData, 'base64', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    } catch (error) {
        console.error('Decryption error:', error.message);
        return textStr;
    }
}

function encryptStudentInfo(info) {
    if (!info) return info;
    
    return {
        ...info,
        name: encrypt(info.name),
        gender: encrypt(info.gender),
        enrollmentYear: encrypt(info.enrollmentYear),
        className: encrypt(info.className),
        major: encrypt(info.major),
        college: encrypt(info.college)
    };
}

function decryptStudentInfo(info) {
    if (!info) return info;
    
    return {
        ...info,
        name: decrypt(info.name),
        gender: decrypt(info.gender),
        enrollmentYear: decrypt(info.enrollmentYear),
        className: decrypt(info.className),
        major: decrypt(info.major),
        college: decrypt(info.college)
    };
}

function encryptCourse(course) {
    if (!course) return course;
    
    return {
        ...course,
        name: encrypt(course.name),
        teacher: encrypt(course.teacher),
        location: encrypt(course.location),
        weeks: encrypt(course.weeks),
        courseType: encrypt(course.courseType),
        raw: encrypt(course.raw)
    };
}

function decryptCourse(course) {
    if (!course) return course;
    
    return {
        ...course,
        name: decrypt(course.name),
        teacher: decrypt(course.teacher),
        location: decrypt(course.location),
        weeks: decrypt(course.weeks),
        courseType: decrypt(course.courseType),
        raw: decrypt(course.raw)
    };
}

function encryptGrade(grade) {
    if (!grade) return grade;
    
    return {
        ...grade,
        courseName: encrypt(grade.courseName),
        score: encrypt(grade.score),
        credit: encrypt(grade.credit),
        gradePoint: encrypt(grade.gradePoint),
        courseType: encrypt(grade.courseType),
        examType: encrypt(grade.examType)
    };
}

function decryptGrade(grade) {
    if (!grade) return grade;
    
    return {
        ...grade,
        courseName: decrypt(grade.courseName),
        score: decrypt(grade.score),
        credit: decrypt(grade.credit),
        gradePoint: decrypt(grade.gradePoint),
        courseType: decrypt(grade.courseType),
        examType: decrypt(grade.examType)
    };
}

function encryptExam(exam) {
    if (!exam) return exam;
    
    return {
        ...exam,
        courseName: encrypt(exam.courseName),
        location: encrypt(exam.location),
        seatNumber: encrypt(exam.seatNumber),
        examType: encrypt(exam.examType),
        status: encrypt(exam.status)
    };
}

function decryptExam(exam) {
    if (!exam) return exam;
    
    return {
        ...exam,
        courseName: decrypt(exam.courseName),
        location: decrypt(exam.location),
        seatNumber: decrypt(exam.seatNumber),
        examType: decrypt(exam.examType),
        status: decrypt(exam.status)
    };
}

function encryptPlan(plan) {
    if (!plan) return plan;
    
    return {
        ...plan,
        courseName: encrypt(plan.courseName),
        teachingUnit: encrypt(plan.teachingUnit),
        credit: encrypt(plan.credit),
        totalHours: encrypt(plan.totalHours),
        examType: encrypt(plan.examType),
        courseAttribute: encrypt(plan.courseAttribute),
        isExam: encrypt(plan.isExam)
    };
}

function decryptPlan(plan) {
    if (!plan) return plan;
    
    return {
        ...plan,
        courseName: decrypt(plan.courseName),
        teachingUnit: decrypt(plan.teachingUnit),
        credit: decrypt(plan.credit),
        totalHours: decrypt(plan.totalHours),
        examType: decrypt(plan.examType),
        courseAttribute: decrypt(plan.courseAttribute),
        isExam: decrypt(plan.isExam)
    };
}

function encryptProgress(progress) {
    if (!progress) return progress;
    
    return {
        ...progress,
        category: encrypt(progress.category),
        requiredCredits: encrypt(progress.requiredCredits),
        completedCredits: encrypt(progress.completedCredits),
        currentCredits: encrypt(progress.currentCredits),
        remainingCredits: encrypt(progress.remainingCredits)
    };
}

function decryptProgress(progress) {
    if (!progress) return progress;
    
    return {
        ...progress,
        category: decrypt(progress.category),
        requiredCredits: decrypt(progress.requiredCredits),
        completedCredits: decrypt(progress.completedCredits),
        currentCredits: decrypt(progress.currentCredits),
        remainingCredits: decrypt(progress.remainingCredits)
    };
}

function encryptElectricityReminder(setting) {
    if (!setting) return setting;
    
    return {
        ...setting,
        electricityAccount: encrypt(setting.electricityAccount),
        roomId: encrypt(setting.roomId),
        campusId: encrypt(setting.campusId),
        buildingId: encrypt(setting.buildingId)
    };
}

function decryptElectricityReminder(setting) {
    if (!setting) return setting;
    
    return {
        ...setting,
        electricityAccount: decrypt(setting.electricityAccount),
        roomId: decrypt(setting.roomId),
        campusId: decrypt(setting.campusId),
        buildingId: decrypt(setting.buildingId)
    };
}

function getEncryptionKeyBase64() {
    return getKey().toString('base64');
}

function getIvBase64() {
    return getIv().toString('base64');
}

module.exports = {
    encrypt,
    decrypt,
    encryptStudentInfo,
    decryptStudentInfo,
    encryptCourse,
    decryptCourse,
    encryptGrade,
    decryptGrade,
    encryptExam,
    decryptExam,
    encryptPlan,
    decryptPlan,
    encryptProgress,
    decryptProgress,
    encryptElectricityReminder,
    decryptElectricityReminder,
    getEncryptionKeyBase64,
    getIvBase64,
    ALGORITHM
};
