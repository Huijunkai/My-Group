const { DataTypes } = require('sequelize');
const { sequelize } = require('../index');

// 学生基本信息表
const Student = sequelize.define('Student', {
    studentId: { type: DataTypes.STRING, primaryKey: true }, // 学号作为主键
    name: DataTypes.STRING,
    gender: DataTypes.STRING,
    enrollmentYear: DataTypes.STRING,
    className: DataTypes.STRING,
    major: DataTypes.STRING,
    college: DataTypes.STRING,
    lastSync: { type: DataTypes.DATE, defaultValue: DataTypes.NOW }
});

// 课程表
const Course = sequelize.define('Course', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    name: { type: DataTypes.STRING, primaryKey: true },
    dayOfWeek: { type: DataTypes.STRING, primaryKey: true },
    teacher: DataTypes.STRING,
    weeks: DataTypes.STRING, // 原始周次字符串，如 "1-16" 或 "1-8,10-16"
    startWeek: DataTypes.INTEGER, // 起始周
    endWeek: DataTypes.INTEGER,   // 结束周
    isOdd: DataTypes.BOOLEAN,     // 是否单周
    isEven: DataTypes.BOOLEAN,    // 是否双周
    period: DataTypes.STRING,     // 节次，如 "1-2"
    startPeriod: DataTypes.INTEGER, // 起始节次
    endPeriod: DataTypes.INTEGER,   // 结束节次
    location: DataTypes.STRING,
    raw: DataTypes.TEXT
});

// 成绩表
const Grade = sequelize.define('Grade', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    courseCode: { type: DataTypes.STRING, primaryKey: true },
    courseName: DataTypes.STRING,
    score: DataTypes.STRING,
    credit: DataTypes.STRING,
    gradePoint: DataTypes.STRING,
    courseType: DataTypes.STRING,
    examType: DataTypes.STRING
});

// 考试安排表
const Exam = sequelize.define('Exam', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    courseName: { type: DataTypes.STRING, primaryKey: true },
    examTime: { type: DataTypes.STRING, primaryKey: true },
    location: DataTypes.STRING,
    seatNumber: DataTypes.STRING,
    examType: DataTypes.STRING,
    status: DataTypes.STRING
});

// 学期计划表
const Plan = sequelize.define('Plan', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    courseCode: { type: DataTypes.STRING, primaryKey: true },
    courseName: DataTypes.STRING,
    credit: DataTypes.STRING,
    totalHours: DataTypes.STRING,
    courseType: DataTypes.STRING,
    examType: DataTypes.STRING
});

// 学习进度表
const Progress = sequelize.define('Progress', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    category: { type: DataTypes.STRING, primaryKey: true }, // 课程体系
    requiredCredits: DataTypes.STRING,
    completedCredits: DataTypes.STRING,
    currentCredits: DataTypes.STRING,
    remainingCredits: DataTypes.STRING
});

module.exports = {
    Student,
    Course,
    Grade,
    Exam,
    Plan,
    Progress
};