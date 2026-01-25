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
    // 说明：必须把 period 纳入主键，否则同一天同名不同节次会被 upsert 覆盖
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    name: { type: DataTypes.STRING, primaryKey: true },
    dayOfWeek: { type: DataTypes.STRING, primaryKey: true },
    period: { type: DataTypes.STRING, primaryKey: true }, // 节次字符串，如 "01-02节"
    teacher: DataTypes.STRING,
    weeks: DataTypes.STRING, // 周次字符串，如 "1-16周" / "1-8,10-16周(单)"
    location: DataTypes.STRING,
    // 兼容字段：前端仍可能用 raw 做解析兜底
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